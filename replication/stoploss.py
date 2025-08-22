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

