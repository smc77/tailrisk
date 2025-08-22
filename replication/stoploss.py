# Monte Carlo with modified stop rule:
# Instead of a 1-month cooldown, go to cash UNTIL cumulative recovery since stop >= K * sigma_ann(at stop).
# Also include transaction costs (per-side) applied when we switch exposure (1->0 or 0->1).
#
# Keep the same fat-tailed AR(1) setup and the same two experiments:
#   (A) μ=+2.5% ann, vary ρ in [-0.5, 0.5]
#   (B) ρ=0.25, vary μ in [-10%, +10%]
#
# t-costs: tc_bps per SIDE (default 5 bps = 0.0005). Costs are subtracted from the return
# in the first month when the new exposure takes effect.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(812)

# ------------------- Core settings -------------------
years = 10
months = years * 12
sigma_annual = 0.05
sigma_m = sigma_annual / np.sqrt(12.0)
nu = 4.0
scale_t = np.sqrt(nu/(nu-2.0))
paths = 6000
K_vol = 2.0
tc_bps = 0.0005  # 5 bps per trade (one-way)

# ------------------- Helpers -------------------
def simulate_AR_t(mu_annual, rho, paths, months):
    mu_m = mu_annual / 12.0
    Zt = rng.standard_t(df=nu, size=(paths, months)) / scale_t
    eps = np.sqrt(1.0 - rho**2) * sigma_m * Zt
    R = np.empty_like(Zt)
    R[:, 0] = mu_m + eps[:, 0]
    for t in range(1, months):
        R[:, t] = mu_m + rho * (R[:, t-1] - mu_m) + eps[:, t]
    return R

def equity_from_returns(R):
    W = np.cumprod(1.0 + R, axis=1)
    W = np.c_[np.ones((R.shape[0], 1)), W]
    return W, R.copy()

def rolling_ann_vol(returns, win=12, lam=None):
    n = returns.shape[1]
    out = np.full_like(returns, np.nan, dtype=float)
    if lam is None:
        for t in range(n):
            lo = max(0, t - win + 1)
            window = returns[:, lo:t+1]
            s = np.std(window, axis=1, ddof=1 if (t-lo+1) > 1 else 0)
            out[:, t] = s * np.sqrt(12.0)
    else:
        alpha = 1.0 - lam
        v = returns[:, [0]]**2
        out[:, 0] = np.sqrt(v[:, 0]) * np.sqrt(12.0)
        for t in range(1, n):
            v = lam * v + alpha * (returns[:, [t]]**2)
            out[:, t] = np.sqrt(v[:, 0]) * np.sqrt(12.0)
    return out

def apply_vol_stop_until_recover(R, K=2.0, tc=0.0005, use_ewma=False):
    """
    Exposure starts at 1.
    If at end of month t, DD_t > K * sigma_ann_t, then from month t+1 onward exposure = 0
    until cumulative return from stop point >= K * sigma_ann_t (measured at stop time).
    One-way transaction cost 'tc' is applied in the first month the new exposure takes effect.
    """
    paths, n = R.shape
    # Price path with t index 0..n (include t=0)
    P = np.cumprod(1.0 + R, axis=1)
    P = np.c_[np.ones((paths,1)), P]
    P_max = np.maximum.accumulate(P, axis=1)
    vol_ann = rolling_ann_vol(R, win=12, lam=0.94 if use_ewma else None)

    expo = np.ones((paths, n), dtype=float)  # exposure applied to month t return
    trade_cost = np.zeros((paths, n), dtype=float)

    # State per path
    stopped = np.zeros(paths, dtype=bool)
    stop_price = np.ones(paths)   # price at stop (P_t at stop time)
    req_recovery = np.zeros(paths) # required cumulative return to re-enter (K * sigma_ann at stop)

    for t in range(1, n+1):  # price index t corresponds to return index t-1
        # End-of-month drawdown & threshold based on current price
        DD_t = 1.0 - P[:, t] / P_max[:, t]
        thr_t = K * vol_ann[:, t-1]

        # If currently stopped, we keep expo=0; else expo=1 by default (set last loop)
        # Check re-entry condition for paths that are stopped:
        if t >= 1:
            cum_gain = (P[:, t] / stop_price) - 1.0
            reenter = stopped & (cum_gain >= req_recovery)
            if np.any(reenter):
                # Re-enter next month: set expo for next month (t) but our array expo is for month t (index t), so we need to mark the *next* period.
                if t < n:  # apply for month t (index t) which is next period
                    expo[reenter, t] = 1.0  # will be overwritten by default next iter; set explicit for clarity
                    trade_cost[reenter, t] -= tc
                # Update states
                stopped[reenter] = False
                # reset placeholders
                stop_price[reenter] = 1.0
                req_recovery[reenter] = 0.0

        # Check new stop trigger based on end-of-month state (applies from next month)
        trigger = (~stopped) & (DD_t > thr_t)
        if np.any(trigger):
            # From next month (t), exposure becomes 0 and we pay a trading cost at that entry to cash
            if t < n:
                expo[trigger, t] = 0.0
                trade_cost[trigger, t] -= tc
            # Set stop state variables
            stopped[trigger] = True
            stop_price[trigger] = P[trigger, t]
            req_recovery[trigger] = thr_t[trigger]

        # For current month t-1, exposure is whatever was set previously; no change here.

        # Ensure exposure state propagates forward if stopped
        if t < n:
            expo[stopped, t] = 0.0  # keep at 0 while stopped

    # Strategy returns: exposure * returns + trade costs when switches occur
    strat_ret = expo * R + trade_cost
    W = np.cumprod(1.0 + strat_ret, axis=1)
    W = np.c_[np.ones((paths,1)), W]
    return strat_ret, W

def sharpe_annual_from_monthly(ret):
    m = np.mean(ret, axis=1)
    s = np.std(ret, axis=1, ddof=1)
    return np.where(s>0, (m / s) * np.sqrt(12.0), np.nan)

def max_drawdown(W):
    M = np.maximum.accumulate(W, axis=1)
    DD = W / M - 1.0
    return DD.min(axis=1)

def summarize_perf(R_bh, W_bh, R_sl, W_sl):
    sr_bh = sharpe_annual_from_monthly(R_bh)
    sr_sl = sharpe_annual_from_monthly(R_sl)
    mdd_bh = max_drawdown(W_bh)
    mdd_sl = max_drawdown(W_sl)
    return {
        "SR_bh_mean": np.nanmean(sr_bh),
        "SR_sl_mean": np.nanmean(sr_sl),
        "MDD_bh_median": np.median(mdd_bh),
        "MDD_sl_median": np.median(mdd_sl),
    }

# ------------------- Experiment A: vary rho -------------------
rho_grid = np.linspace(-0.5, 0.5, 21)
mu_ann_A = 0.025

rows_A = []
for rho in rho_grid:
    R = simulate_AR_t(mu_ann_A, rho, paths, months)
    W_bh, R_bh = equity_from_returns(R)
    R_sl, W_sl = apply_vol_stop_until_recover(R, K=K_vol, tc=tc_bps, use_ewma=False)
    s = summarize_perf(R_bh, W_bh, R_sl, W_sl)
    rows_A.append([rho, s["SR_bh_mean"], s["SR_sl_mean"], s["MDD_bh_median"], s["MDD_sl_median"]])
dfA = pd.DataFrame(rows_A, columns=["rho", "SR_bh", "SR_stop", "MDD_bh", "MDD_stop"])

# ------------------- Experiment B: vary mu -------------------
mu_grid = np.linspace(-0.10, 0.10, 21)
rho_B = 0.25

rows_B = []
for mu_ann in mu_grid:
    R = simulate_AR_t(mu_ann, rho_B, paths, months)
    W_bh, R_bh = equity_from_returns(R)
    R_sl, W_sl = apply_vol_stop_until_recover(R, K=K_vol, tc=tc_bps, use_ewma=False)
    s = summarize_perf(R_bh, W_bh, R_sl, W_sl)
    rows_B.append([mu_ann, s["SR_bh_mean"], s["SR_sl_mean"], s["MDD_bh_median"], s["MDD_sl_median"]])
dfB = pd.DataFrame(rows_B, columns=["mu_ann", "SR_bh", "SR_stop", "MDD_bh", "MDD_stop"])

# ------------------- Plots -------------------
# A: Sharpe vs rho
plt.figure()
plt.plot(dfA["rho"], dfA["SR_bh"], 'o-', label="Buy & Hold")
plt.plot(dfA["rho"], dfA["SR_stop"], 's--', label=f"Vol Stop (K={K_vol}, until recover by K·σ; tc={int(tc_bps*1e4)} bps/side)")
plt.xlabel("AR(1) ρ")
plt.ylabel("Sharpe (annualized)")
plt.title("Experiment A: Sharpe vs AR(1) ρ (μ=+2.5% ann, σ=5% ann)")
plt.legend()
plt.tight_layout()
plt.show()

# A: MDD vs rho
plt.figure()
plt.plot(dfA["rho"], -100*dfA["MDD_bh"], 'o-', label="Buy & Hold")
plt.plot(dfA["rho"], -100*dfA["MDD_stop"], 's--', label=f"Vol Stop (K={K_vol})")
plt.xlabel("AR(1) ρ")
plt.ylabel("Median Max Drawdown (%)")
plt.title("Experiment A: Max Drawdown vs AR(1) ρ (μ=+2.5% ann, σ=5% ann)")
plt.legend()
plt.tight_layout()
plt.show()

# B: Sharpe vs mu
plt.figure()
plt.plot(100*dfB["mu_ann"], dfB["SR_bh"], 'o-', label="Buy & Hold")
plt.plot(100*dfB["mu_ann"], dfB["SR_stop"], 's--', label=f"Vol Stop (K={K_vol}, until recover; tc={int(tc_bps*1e4)} bps/side)")
plt.xlabel("Annual μ (%)")
plt.ylabel("Sharpe (annualized)")
plt.title("Experiment B: Sharpe vs μ (ρ=0.25, σ=5% ann)")
plt.legend()
plt.tight_layout()
plt.show()

# B: MDD vs mu
plt.figure()
plt.plot(100*dfB["mu_ann"], -100*dfB["MDD_bh"], 'o-', label="Buy & Hold")
plt.plot(100*dfB["mu_ann"], -100*dfB["MDD_stop"], 's--', label=f"Vol Stop (K={K_vol})")
plt.xlabel("Annual μ (%)")
plt.ylabel("Median Max Drawdown (%)")
plt.title("Experiment B: Max Drawdown vs μ (ρ=0.25, σ=5% ann)")
plt.legend()
plt.tight_layout()
plt.show()

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Experiment A (until recovery + t-costs)", dfA.round(4))
caas_jupyter_tools.display_dataframe_to_user("Experiment B (until recovery + t-costs)", dfB.round(4))




# Daily Monte Carlo — separate exit (K) and entry (E) thresholds, log grids, two experiments
# Settings: AR(1)-t daily, ρ=0.25, μ=2.5% p.a., σ=5% p.a., ν=4
# Runtime tuned: 400 paths, 80 bootstrap reps, ~10 years horizon
# Output: Sharpe vs threshold, MDD vs threshold with 95% bootstrap CIs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(20250822)

# ---------------- Settings ----------------
years = 10
days_per_year = 252
N = years * days_per_year
mu_ann = 0.025
rho = 0.25
sigma_ann = 0.05
sigma_d = sigma_ann / np.sqrt(days_per_year)
mu_d = mu_ann / days_per_year
nu = 4.0
scale_t = np.sqrt(nu/(nu-2.0))

paths = 400
tc_bps = 0.0005
B = 80
alpha = 0.05

# Log grids
high_seg = np.logspace(np.log10(3.0), np.log10(0.5), 5)
low_seg  = np.logspace(np.log10(0.5), np.log10(0.05), 8)
grid = np.unique(np.r_[high_seg, low_seg])[::-1]

# ---------------- Helpers ----------------
def simulate_AR_t_daily(mu_d, sigma_d, rho, paths, N):
    Z = rng.standard_t(df=nu, size=(paths, N)) / scale_t
    eps = np.sqrt(1.0 - rho**2) * sigma_d * Z
    R = np.empty_like(Z)
    R[:, 0] = mu_d + eps[:, 0]
    for t in range(1, N):
        R[:, t] = mu_d + rho*(R[:, t-1] - mu_d) + eps[:, t]
    return R

def wealth_from_returns(R):
    W = np.cumprod(1.0 + R, axis=1)
    return np.c_[np.ones((R.shape[0],1)), W]

def rolling_ann_vol_paths(R, win=252):
    P, T = R.shape
    out = np.empty_like(R, dtype=float)
    for i in range(P):
        r = R[i]
        for t in range(T):
            lo = max(0, t - win + 1)
            window = r[lo:t+1]
            s = np.std(window, ddof=1 if (t-lo+1)>1 else 0)
            out[i,t] = s * np.sqrt(days_per_year)
    return out

def apply_stop_runninglow_daily_single(P, M, vol_ann, K_exit, E_entry, tc):
    N = len(vol_ann)
    expo = np.ones(N, dtype=float)
    costs = np.zeros(N, dtype=float)
    stopped = False
    sigma_stop = 0.0
    running_low = 1.0
    for t in range(1, N+1):
        if stopped:
            running_low = min(running_low, P[t])
            target = running_low * (1.0 + E_entry * sigma_stop)
            if P[t] >= target:
                if t < N:
                    expo[t] = 1.0
                    if tc > 0: costs[t] -= tc
                stopped = False
        thr = K_exit * vol_ann[t-1]
        DD_t = 1.0 - P[t] / M[t]
        if (not stopped) and (DD_t > thr):
            if t < N:
                expo[t] = 0.0
                if tc > 0: costs[t] -= tc
            stopped = True
            sigma_stop = vol_ann[t-1]
            running_low = P[t]
        if stopped and t < N:
            expo[t] = 0.0
    return expo, costs

def sharpe_ann_from_daily(ret):
    m = np.mean(ret, axis=1)
    s = np.std(ret, axis=1, ddof=1)
    return np.where(s>0, (m/s)*np.sqrt(days_per_year), np.nan)

def max_drawdown(W):
    M = np.maximum.accumulate(W, axis=1)
    DD = W/M - 1.0
    return DD.min(axis=1)

def bootstrap_ci(values, stat="mean", B=80, alpha=0.05):
    rng_loc = np.random.default_rng(12345)
    n = len(values)
    stats = np.empty(B)
    for b in range(B):
        idx = rng_loc.integers(0, n, size=n)
        sample = values[idx]
        stats[b] = np.nanmean(sample) if stat=="mean" else np.nanmedian(sample)
    lo = np.nanpercentile(stats, 100*alpha/2)
    hi = np.nanpercentile(stats, 100*(1-alpha/2))
    return lo, hi

def summarize(R_bh, W_bh, R_g, W_g, R_n, W_n):
    sr_bh = sharpe_ann_from_daily(R_bh)
    sr_g  = sharpe_ann_from_daily(R_g)
    sr_n  = sharpe_ann_from_daily(R_n)
    mdd_bh = max_drawdown(W_bh)
    mdd_g  = max_drawdown(W_g)
    mdd_n  = max_drawdown(W_n)
    return {
        "SR_bh": float(np.nanmean(sr_bh)),
        "SR_g":  float(np.nanmean(sr_g)),
        "SR_n":  float(np.nanmean(sr_n)),
        "SR_bh_ci": bootstrap_ci(sr_bh, "mean", B=B, alpha=alpha),
        "SR_g_ci":  bootstrap_ci(sr_g, "mean", B=B, alpha=alpha),
        "SR_n_ci":  bootstrap_ci(sr_n, "mean", B=B, alpha=alpha),
        "MDD_bh": float(np.nanmedian(mdd_bh)),
        "MDD_g":  float(np.nanmedian(mdd_g)),
        "MDD_n":  float(np.nanmedian(mdd_n)),
        "MDD_bh_ci": bootstrap_ci(mdd_bh, "median", B=B, alpha=alpha),
        "MDD_g_ci":  bootstrap_ci(mdd_g, "median", B=B, alpha=alpha),
        "MDD_n_ci":  bootstrap_ci(mdd_n, "median", B=B, alpha=alpha),
    }

def plot_with_ci(ax, x, y, ci, label, ls):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ylo = np.asarray([c[0] for c in ci], dtype=float)
    yhi = np.asarray([c[1] for c in ci], dtype=float)
    ax.plot(x, y, ls, label=label, linewidth=2)
    ax.fill_between(x, ylo, yhi, alpha=0.15)

# ---------------- Simulate base paths ----------------
R = simulate_AR_t_daily(mu_d, sigma_d, rho, paths, N)
W_bh = wealth_from_returns(R)
M_bh = np.maximum.accumulate(W_bh, axis=1)
vol_ann = rolling_ann_vol_paths(R, win=252)

# Precompute BH stats
SR_bh_paths = sharpe_ann_from_daily(R)
MDD_bh_paths = max_drawdown(W_bh)
SR_bh_mean = float(np.nanmean(SR_bh_paths))
MDD_bh_med = float(np.nanmedian(MDD_bh_paths))
SR_bh_ci = bootstrap_ci(SR_bh_paths, "mean", B=B, alpha=alpha)
MDD_bh_ci = bootstrap_ci(MDD_bh_paths, "median", B=B, alpha=alpha)

# ---------------- Experiment 1: vary E, K=1 ----------------
rowsE = []
for E in grid:
    expo_g = np.ones_like(R); costs_g = np.zeros_like(R)
    expo_n = np.ones_like(R); costs_n = np.zeros_like(R)
    for i in range(paths):
        e_g, c_g = apply_stop_runninglow_daily_single(W_bh[i], M_bh[i], vol_ann[i], K_exit=1.0, E_entry=E, tc=0.0)
        e_n, c_n = apply_stop_runninglow_daily_single(W_bh[i], M_bh[i], vol_ann[i], K_exit=1.0, E_entry=E, tc=tc_bps)
        expo_g[i], costs_g[i] = e_g, c_g
        expo_n[i], costs_n[i] = e_n, c_n
    R_g = expo_g * R + costs_g; R_n = expo_n * R + costs_n
    W_g = wealth_from_returns(R_g); W_n = wealth_from_returns(R_n)
    s = summarize(R, W_bh, R_g, W_g, R_n, W_n)
    rowsE.append({"E": E, **s})

dfE = pd.DataFrame(rowsE)

# ---------------- Experiment 2: vary K, E=1 ----------------
rowsK = []
for K in grid:
    expo_g = np.ones_like(R); costs_g = np.zeros_like(R)
    expo_n = np.ones_like(R); costs_n = np.zeros_like(R)
    for i in range(paths):
        e_g, c_g = apply_stop_runninglow_daily_single(W_bh[i], M_bh[i], vol_ann[i], K_exit=K, E_entry=1.0, tc=0.0)
        e_n, c_n = apply_stop_runninglow_daily_single(W_bh[i], M_bh[i], vol_ann[i], K_exit=K, E_entry=1.0, tc=tc_bps)
        expo_g[i], costs_g[i] = e_g, c_g
        expo_n[i], costs_n[i] = e_n, c_n
    R_g = expo_g * R + costs_g; R_n = expo_n * R + costs_n
    W_g = wealth_from_returns(R_g); W_n = wealth_from_returns(R_n)
    s = summarize(R, W_bh, R_g, W_g, R_n, W_n)
    rowsK.append({"K": K, **s})

dfK = pd.DataFrame(rowsK)

# ---------------- Plots ----------------
fig, axs = plt.subplots(2,2, figsize=(13,10))

# Exp 1 (vary E, K=1)
plot_with_ci(axs[0,0], dfE["E"], dfE["SR_bh"], [SR_bh_ci]*len(dfE), "Buy & Hold", "o-")
plot_with_ci(axs[0,0], dfE["E"], dfE["SR_g"],  list(dfE["SR_g_ci"]), "Stop (gross)", "s--")
plot_with_ci(axs[0,0], dfE["E"], dfE["SR_n"],  list(dfE["SR_n_ci"]), "Stop (net)",   "d:")
axs[0,0].set_xscale("log"); axs[0,0].invert_xaxis()
axs[0,0].set_title("Sharpe vs E (K=1 fixed) — Daily"); axs[0,0].set_xlabel("E (σ units)"); axs[0,0].set_ylabel("Sharpe"); axs[0,0].legend()

plot_with_ci(axs[0,1], dfE["E"], -100*dfE["MDD_bh"], [(-100*MDD_bh_ci[0], -100*MDD_bh_ci[1])]*len(dfE), "Buy & Hold", "o-")
plot_with_ci(axs[0,1], dfE["E"], -100*dfE["MDD_g"],  [(-100*lo,-100*hi) for (lo,hi) in dfE["MDD_g_ci"]], "Stop (gross)", "s--")
plot_with_ci(axs[0,1], dfE["E"], -100*dfE["MDD_n"],  [(-100*lo,-100*hi) for (lo,hi) in dfE["MDD_n_ci"]], "Stop (net)",   "d:")
axs[0,1].set_xscale("log"); axs[0,1].invert_xaxis()
axs[0,1].set_title("Median Max Drawdown vs E (K=1 fixed) — Daily"); axs[0,1].set_xlabel("E (σ units)"); axs[0,1].set_ylabel("MDD (%)")

# Exp 2 (vary K, E=1)
plot_with_ci(axs[1,0], dfK["K"], dfK["SR_bh"], [SR_bh_ci]*len(dfK), "Buy & Hold", "o-")
plot_with_ci(axs[1,0], dfK["K"], dfK["SR_g"],  list(dfK["SR_g_ci"]), "Stop (gross)", "s--")
plot_with_ci(axs[1,0], dfK["K"], dfK["SR_n"],  list(dfK["SR_n_ci"]), "Stop (net)",   "d:")
axs[1,0].set_xscale("log"); axs[1,0].invert_xaxis()
axs[1,0].set_title("Sharpe vs K (E=1 fixed) — Daily"); axs[1,0].set_xlabel("K (σ units)"); axs[1,0].set_ylabel("Sharpe"); axs[1,0].legend()

plot_with_ci(axs[1,1], dfK["K"], -100*dfK["MDD_bh"], [(-100*MDD_bh_ci[0], -100*MDD_bh_ci[1])]*len(dfK), "Buy & Hold", "o-")
plot_with_ci(axs[1,1], dfK["K"], -100*dfK["MDD_g"],  [(-100*lo,-100*hi) for (lo,hi) in dfK["MDD_g_ci"]], "Stop (gross)", "s--")
plot_with_ci(axs[1,1], dfK["K"], -100*dfK["MDD_n"],  [(-100*lo,-100*hi) for (lo,hi) in dfK["MDD_n_ci"]], "Stop (net)",   "d:")
axs[1,1].set_xscale("log"); axs[1,1].invert_xaxis()
axs[1,1].set_title("Median Max Drawdown vs K (E=1 fixed) — Daily"); axs[1,1].set_xlabel("K (σ units)"); axs[1,1].set_ylabel("MDD (%)")

plt.tight_layout()
plt.show()

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Experiment 1 (vary E, K=1) — daily, log grid", dfE.round(4))
caas_jupyter_tools.display_dataframe_to_user("Experiment 2 (vary K, E=1) — daily, log grid", dfK.round(4))

#
#
#

# Daily rebalancing with re-entry relative to a RUNNING LOW (after the stop), plus target lines
# Settings: μ=2.5% p.a., ρ=0.25 (daily), σ=5% p.a., Student-t(ν=4), K=0.5, tc=5 bps/side
# Top: wealth, peak, stop markers, re-entry markers, two target lines:
#      1) static stop-price target: stop_price * (1 + K * sigma_at_stop)
#      2) dynamic running-low target: running_low * (1 + K * sigma_at_stop)
# Bottom: cumulative P&L for BH, stop (gross), stop (net)

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(20250822)

# --- Parameters ---
years = 10
days_per_year = 252
N = years * days_per_year
mu_ann = 0.025
rho = 0.25
sigma_ann = 0.05
sigma_d = sigma_ann / np.sqrt(days_per_year)
mu_d = mu_ann / days_per_year
nu = 4.0
scale_t = np.sqrt(nu/(nu-2.0))
K = 0.5
tc = 0.0005

# --- Simulate daily AR(1)-t returns (one path) ---
Zt = rng.standard_t(df=nu, size=N) / scale_t  # unit variance
eps = np.sqrt(1.0 - rho**2) * sigma_d * Zt
R = np.empty(N)
R[0] = mu_d + eps[0]
for t in range(1, N):
    R[t] = mu_d + rho*(R[t-1] - mu_d) + eps[t]

# --- Rolling annualized vol (252d window) ---
def rolling_ann_vol_1d(r, win=252):
    out = np.empty_like(r, dtype=float)
    for t in range(len(r)):
        lo = max(0, t - win + 1)
        window = r[lo:t+1]
        ddof = 1 if (t-lo+1) > 1 else 0
        s = np.std(window, ddof=ddof)
        out[t] = s * np.sqrt(days_per_year)
    return out

vol_ann = rolling_ann_vol_1d(R, win=252)

# --- Wealth / drawdown ---
W_bh = np.insert(np.cumprod(1.0 + R), 0, 1.0)  # include t=0
P = W_bh.copy()
M = np.maximum.accumulate(P)
DD = 1.0 - P / M

# --- Apply stop-until-recover (relative to running low), with gross & net ---
expo_g = np.ones(N)
expo_n = np.ones(N)
costs_g = np.zeros(N)
costs_n = np.zeros(N)

stopped = False
stop_price = None
sigma_at_stop = None
running_low = None

stop_idx = []
reenter_idx = []

# Lines to plot (aligned with P: length N+1)
static_target_line = np.full(N+1, np.nan)    # stop_price * (1 + K*sigma_at_stop)
running_low_line   = np.full(N+1, np.nan)    # running_low * (1 + K*sigma_at_stop)

for t in range(1, N+1):  # price index; return index t-1
    # If stopped, update running low and the dynamic target
    if stopped:
        running_low = min(running_low, P[t])
        running_low_line[t:] = running_low * (1.0 + K * sigma_at_stop)
        # Re-entry test vs running-low target
        if P[t] >= running_low * (1.0 + K * sigma_at_stop):
            if t < N:
                expo_g[t] = 1.0
                expo_n[t] = 1.0
                costs_n[t] -= tc
            stopped = False
            reenter_idx.append(t)
    # Trigger test (vs running peak DD threshold)
    thr_t = K * vol_ann[t-1]
    if (not stopped) and (DD[t] > thr_t):
        if t < N:
            expo_g[t] = 0.0
            expo_n[t] = 0.0
            costs_n[t] -= tc
        stopped = True
        stop_idx.append(t)
        stop_price = P[t]
        sigma_at_stop = thr_t / K   # equals vol_ann[t-1]
        running_low = P[t]          # reset running low at stop time
        # set static and dynamic targets from this stop onward
        static_target_line[t:]  = stop_price  * (1.0 + K * sigma_at_stop)
        running_low_line[t:]    = running_low * (1.0 + K * sigma_at_stop)
    # Persist 0 exposure if still stopped
    if stopped and t < N:
        expo_g[t] = 0.0
        expo_n[t] = 0.0

# Strategy returns and P&L
R_gross = expo_g * R + costs_g
R_net   = expo_n * R + costs_n
W_gross = np.insert(np.cumprod(1.0 + R_gross), 0, 1.0)
W_net   = np.insert(np.cumprod(1.0 + R_net),   0, 1.0)

PnL_bh = P - 1.0
PnL_g  = W_gross - 1.0
PnL_n  = W_net - 1.0

# --- Plot ---
tgrid = np.arange(N+1)
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: wealth & target lines
axs[0].plot(tgrid, P, label="Wealth (Price)")
axs[0].plot(tgrid, M, linestyle="--", label="Running peak")
axs[0].plot(tgrid, static_target_line, linestyle="-.", label="Static target (stop_price·(1+K·σ_stop))")
axs[0].plot(tgrid, running_low_line, linestyle=":",   label="Dynamic target (running_low·(1+K·σ_stop))")
axs[0].scatter(stop_idx, P[stop_idx], marker="x", label="Stops")
axs[0].scatter(reenter_idx, P[reenter_idx], marker="o", label="Re-entries")
axs[0].set_ylabel("Wealth")
axs[0].set_title("Daily — Re-entry vs Running Low (K=0.5, ρ=0.25, μ=2.5% p.a., σ=5% p.a., tν=4)")
axs[0].legend(loc="best")

# Bottom: cumulative P&L
axs[1].plot(tgrid, PnL_bh, label="Buy & Hold P&L")
axs[1].plot(tgrid, PnL_g,  label="Stop-Loss (gross) P&L")
axs[1].plot(tgrid, PnL_n,  label="Stop-Loss (net, 5 bps/side) P&L")
axs[1].set_xlabel("Day")
axs[1].set_ylabel("Cumulative P&L")
axs[1].legend(loc="best")
axs[1].set_title("Cumulative P&L (Daily)")

plt.tight_layout()
plt.show()
