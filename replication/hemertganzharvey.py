# Re-run overlay code after reset

import numpy as np
import matplotlib.pyplot as plt

# Recreate Gaussian baseline and fat-tail AR simulations quickly to replot overlay
years = 10
months = years * 12
sigma_annual = 0.10
SR_annual = 0.5
paths = 100_000
rng = np.random.default_rng(123)

nu = 4.0
rho = 0.30
scale_t = np.sqrt(nu/(nu-2.0))
sigma_m = sigma_annual / np.sqrt(12)
mu_m = (SR_annual * sigma_annual) / 12

# Fat-tail AR(1)
Zt = rng.standard_t(df=nu, size=(paths, months)) / scale_t
eps = np.sqrt(1.0 - rho**2) * sigma_m * Zt
R_fat = np.empty_like(Zt)
R_fat[:,0] = mu_m + eps[:,0]
for t in range(1, months):
    R_fat[:,t] = mu_m + rho*(R_fat[:,t-1] - mu_m) + eps[:,t]
Wf = np.cumprod(1.0 + R_fat, axis=1)
Wf = np.c_[np.ones((paths,1)), Wf]
Mf = np.maximum.accumulate(Wf, axis=1)
DDf = Wf/Mf - 1.0
max_dd_fat = DDf.min(axis=1)

# Gaussian baseline
R_g = rng.normal(loc=mu_m, scale=sigma_m, size=(paths, months))
Wg = np.cumprod(1.0 + R_g, axis=1)
Wg = np.c_[np.ones((paths,1)), Wg]
Mg = np.maximum.accumulate(Wg, axis=1)
DDg = Wg/Mg - 1.0
max_dd_g = DDg.min(axis=1)

thresholds = np.array([-0.10, -0.20, -0.30, -0.40])

# Overlay plot
plt.figure(figsize=(8,5))
bins = 160
plt.hist(max_dd_g*100, bins=bins, density=True, alpha=0.5, label="Gaussian baseline")
plt.hist(max_dd_fat*100, bins=bins, density=True, alpha=0.5, label=f"Fat-tail t(ν={int(nu)}), AR(1) ρ={rho}")
for thr in thresholds:
    plt.axvline(thr*100, linestyle="--", color="black", alpha=0.7)
plt.xlabel("Maximum Drawdown (%)")
plt.ylabel("Density")
plt.title("Max Drawdown Distribution (10y)\nBaseline Gaussian vs Fat-tail + AR(1)")
plt.legend()
plt.tight_layout()

fig_overlay_path = "/mnt/data/figure1_overlay.png"
plt.savefig(fig_overlay_path, dpi=200, bbox_inches="tight")
plt.show()

fig_overlay_path




#
#
#

# "Figure 1" with fat tails + autocorrelation
# Replicate Exhibit 1 (distribution of maximum drawdown over 10y), but switch to
# monthly AR(1) returns with Student-t innovations (fat tails) instead of i.i.d. Gaussian.
#
# Baseline (paper): i.i.d. normal monthly, 10y, σ=10% (ann.), SR=0.5.
# Here:            AR(1) monthly with ν=4 Student-t, ρ=+0.30, same σ and SR.
#
# We also compute the baseline Gaussian probabilities again (same settings) for comparison.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Parameters ----------
years = 10
months = years * 12
sigma_annual = 0.10
SR_annual = 0.5
paths = 100_000
rng = np.random.default_rng(87654321)

# Fat-tail + autocorr settings
nu = 4.0           # Student-t degrees of freedom
rho = 0.30         # monthly AR(1) autocorrelation
scale_t = np.sqrt(nu/(nu-2.0))  # std of t_nu; divide by this to standardize to unit variance

# Monthly params
sigma_m = sigma_annual / np.sqrt(12)
mu_m = (SR_annual * sigma_annual) / 12

# ---------- Simulate AR(1)-t monthly returns ----------
Zt = rng.standard_t(df=nu, size=(paths, months)) / scale_t        # unit variance
eps = np.sqrt(1.0 - rho**2) * sigma_m * Zt                         # shock scaled to keep unconditional var = sigma_m^2
R = np.empty_like(Zt)
R[:, 0] = mu_m + eps[:, 0]
for t in range(1, months):
    R[:, t] = mu_m + rho*(R[:, t-1] - mu_m) + eps[:, t]

# Build wealth and MDD
W = np.cumprod(1.0 + R, axis=1)
W = np.c_[np.ones((paths, 1)), W]
M = np.maximum.accumulate(W, axis=1)
DD = W / M - 1.0
max_dd = DD.min(axis=1)

# ---------- Baseline Gaussian (for a quick side-by-side probability comparison) ----------
R_g = rng.normal(loc=mu_m, scale=sigma_m, size=(paths, months))
Wg = np.cumprod(1.0 + R_g, axis=1)
Wg = np.c_[np.ones((paths, 1)), Wg]
Mg = np.maximum.accumulate(Wg, axis=1)
DDg = Wg / Mg - 1.0
max_dd_g = DDg.min(axis=1)

# ---------- Probabilities at kσ, k=1..4 ----------
thresholds = np.array([-0.10, -0.20, -0.30, -0.40])  # -k * sigma_annual
probs_fat = [(max_dd <= thr).mean() for thr in thresholds]
probs_gau = [(max_dd_g <= thr).mean() for thr in thresholds]

# ---------- Plot histogram (fat-tail + AR only) ----------
plt.figure()
bins = 160
plt.hist(max_dd * 100, bins=bins, density=True, alpha=0.9)
for thr in thresholds:
    plt.axvline(thr * 100, linestyle="--")
plt.xlabel("Maximum Drawdown (%)")
plt.ylabel("Density")
plt.title(f"Max Drawdown Distribution (10y) — Fat tails (tν={int(nu)}), AR(1) ρ={rho}\nσ=10% (ann.), SR=0.5 (ann.), monthly returns")
plt.tight_layout()
# Save figure
fig_path = "/mnt/data/figure1_fattail_AR.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.show()

# ---------- Display probability table ----------
prob_table = pd.DataFrame({
    "Threshold (annual σ)": ["−1σ (−10%)", "−2σ (−20%)", "−3σ (−30%)", "−4σ (−40%)"],
    "Fat-tail AR(1) [%]": np.round(100*np.array(probs_fat), 2),
    "Baseline Gaussian [%]": np.round(100*np.array(probs_gau), 2),
})
print(prob_table)

# Save CSV
csv_path = "/mnt/data/figure1_fattail_AR_probs.csv"
prob_table.to_csv(csv_path, index=False)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Figure 1 (fat-tail + AR) — exceedance probabilities", prob_table)

fig_path, csv_path






# Replicate Figure 2 (Exhibit 2) from Van Hemert et al. (2020) "Drawdowns"
# Source: https://people.duke.edu/~charvey/Research/Published_Papers/P147_Drawdowns.pdf
#
# Figure 2 shows, for four separate parameter sweeps, the probability of reaching a
# maximum drawdown worse than k * sigma_annual for k in {1,2,3,4}:
#   A) Vary annualized volatility (Sharpe fixed at 0.5), horizon=10y, rho=0
#   B) Vary horizon in years (vol=10% ann, Sharpe=0.5, rho=0)
#   C) Vary Sharpe ratio (vol=10% ann, horizon=10y, rho=0)
#   D) Vary autocorrelation rho (monthly AR(1), vol=10% ann, Sharpe=0.5, horizon=10y)
#
# Implementation notes:
# - Monthly simple returns; for AR(1): r_t = mu_m + rho*(r_{t-1} - mu_m) + sqrt(1-rho^2)*sigma_m * z_t
#   so unconditional monthly variance = sigma_m^2 regardless of rho.
# - Thresholds are expressed in annual sigma units: we compare MaxDD <= -k * sigma_annual for k=1..4.
# - We simulate many paths at each x-value to estimate the probabilities.
#
# IMPORTANT: To keep execution time reasonable in this environment, we use 10,000 paths per x-point.
# You can increase `paths_per_point` if you want tighter estimates.
# Each panel is plotted in its own figure (no subplots, default colors).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(2025)

# Core MDD calculator for one matrix of wealth paths (paths x T+1)
def max_drawdown_per_path(W):
    M = np.maximum.accumulate(W, axis=1)
    DD = W / M - 1.0
    return DD.min(axis=1)  # most negative

def simulate_paths(months, sigma_annual, SR_annual, rho=0.0, paths=10000):
    """Simulate monthly simple returns and wealth; returns wealth matrix (paths x (months+1))."""
    sigma_m = sigma_annual / np.sqrt(12.0)
    mu_m = (SR_annual * sigma_annual) / 12.0  # mean excess per month

    Z = rng.standard_normal(size=(paths, months))
    if abs(rho) < 1e-12:
        R = mu_m + sigma_m * Z
    else:
        # AR(1) with unconditional var = sigma_m^2
        eps = np.sqrt(1.0 - rho**2) * sigma_m * Z
        R = np.empty_like(Z)
        R[:, 0] = mu_m + eps[:, 0]
        for t in range(1, months):
            R[:, t] = mu_m + rho*(R[:, t-1] - mu_m) + eps[:, t]
    W = np.cumprod(1.0 + R, axis=1)
    W = np.c_[np.ones((paths, 1)), W]
    return W

def estimate_probs(months, sigma_annual, SR_annual, rho, paths_per_point):
    W = simulate_paths(months, sigma_annual, SR_annual, rho=rho, paths=paths_per_point)
    mdd = max_drawdown_per_path(W)
    # thresholds in annual-sigma units (k * sigma_annual)
    thresholds = np.array([1,2,3,4], dtype=float) * sigma_annual
    probs = [(mdd <= -thr).mean() for thr in thresholds]
    return probs  # list of 4

# Baseline
sigma0 = 0.10
SR0 = 0.5
T0_years = 10
rho0 = 0.0

paths_per_point = 10000

# Panel A: vary annualized volatility (keep SR fixed)
vol_grid = np.linspace(0.06, 0.14, 17)  # 6% to 14% in 0.5% steps
probs_A = []
for sig in vol_grid:
    probs_A.append(estimate_probs(months=T0_years*12, sigma_annual=sig, SR_annual=SR0, rho=rho0, paths_per_point=paths_per_point))
probs_A = np.array(probs_A)  # shape (len(vol_grid), 4)

plt.figure()
for k in range(4):
    plt.plot(vol_grid*100, probs_A[:, k]*100, label=f"P(DD ≤ −{k+1}σ)")
plt.axvline(sigma0*100, linestyle="--")
plt.xlabel("Annualized Volatility (%)")
plt.ylabel("Probability of Drawdown (%)")
plt.title("Figure 2A: Probability of Max Drawdown vs Volatility\n(SR fixed at 0.5, Horizon 10y, ρ=0)")
plt.legend()
plt.tight_layout()
plt.show()

# Panel B: vary time horizon (years)
time_grid_years = np.arange(1, 31)  # 1 to 30 years
probs_B = []
for yrs in time_grid_years:
    probs_B.append(estimate_probs(months=yrs*12, sigma_annual=sigma0, SR_annual=SR0, rho=rho0, paths_per_point=paths_per_point))
probs_B = np.array(probs_B)

plt.figure()
for k in range(4):
    plt.plot(time_grid_years, probs_B[:, k]*100, label=f"P(DD ≤ −{k+1}σ)")
plt.axvline(T0_years, linestyle="--")
plt.xlabel("Time (years)")
plt.ylabel("Probability of Drawdown (%)")
plt.title("Figure 2B: Probability of Max Drawdown vs Horizon\n(σ=10% ann, SR=0.5, ρ=0)")
plt.legend()
plt.tight_layout()
plt.show()

# Panel C: vary Sharpe ratio (vol fixed)
SR_grid = np.linspace(0.0, 1.0, 21)
probs_C = []
for SR in SR_grid:
    probs_C.append(estimate_probs(months=T0_years*12, sigma_annual=sigma0, SR_annual=SR, rho=rho0, paths_per_point=paths_per_point))
probs_C = np.array(probs_C)

plt.figure()
for k in range(4):
    plt.plot(SR_grid, probs_C[:, k]*100, label=f"P(DD ≤ −{k+1}σ)")
plt.axvline(SR0, linestyle="--")
plt.xlabel("Sharpe Ratio (annual)")
plt.ylabel("Probability of Drawdown (%)")
plt.title("Figure 2C: Probability of Max Drawdown vs Sharpe\n(σ=10% ann, Horizon 10y, ρ=0)")
plt.legend()
plt.tight_layout()
plt.show()

# Panel D: vary autocorrelation rho (monthly)
rho_grid = np.linspace(-0.10, 0.10, 21)
probs_D = []
for rho in rho_grid:
    probs_D.append(estimate_probs(months=T0_years*12, sigma_annual=sigma0, SR_annual=SR0, rho=rho, paths_per_point=paths_per_point))
probs_D = np.array(probs_D)

plt.figure()
for k in range(4):
    plt.plot(rho_grid, probs_D[:, k]*100, label=f"P(DD ≤ −{k+1}σ)")
plt.axvline(rho0, linestyle="--")
plt.xlabel("ρ (monthly autocorrelation)")
plt.ylabel("Probability of Drawdown (%)")
plt.title("Figure 2D: Probability of Max Drawdown vs Autocorrelation\n(σ=10% ann, SR=0.5, Horizon 10y)")
plt.legend()
plt.tight_layout()
plt.show()

# Package results tables (optional viewing)
dfA = pd.DataFrame(probs_A*100, columns=["≤−1σ","≤−2σ","≤−3σ","≤−4σ"])
dfA.insert(0, "Vol_%", vol_grid*100)

dfB = pd.DataFrame(probs_B*100, columns=["≤−1σ","≤−2σ","≤−3σ","≤−4σ"])
dfB.insert(0, "Years", time_grid_years)

dfC = pd.DataFrame(probs_C*100, columns=["≤−1σ","≤−2σ","≤−3σ","≤−4σ"])
dfC.insert(0, "SR", SR_grid)

dfD = pd.DataFrame(probs_D*100, columns=["≤−1σ","≤−2σ","≤−3σ","≤−4σ"])
dfD.insert(0, "rho", rho_grid)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Figure 2A data (Volatility sweep)", dfA.round(2))
caas_jupyter_tools.display_dataframe_to_user("Figure 2B data (Horizon sweep)", dfB.round(2))
caas_jupyter_tools.display_dataframe_to_user("Figure 2C data (Sharpe sweep)", dfC.round(2))
caas_jupyter_tools.display_dataframe_to_user("Figure 2D data (Autocorr sweep)", dfD.round(2))




# Exhibit 6 replication — Van Hemert et al. (2020) "Drawdowns"
# Panel A: No migration (50% good SR=0.5, 50% bad SR=0.0), normal i.i.d., σ=10% (annual), horizon=10y
# Panel B: Migration (start good SR=0.5; each month 0.5% chance to switch permanently to bad SR=0.0)
# Decision at the end of 10 years using one of three statistics:
#   (1) Total return   (2) Drawdown at T   (3) Maximum drawdown
# For each rule, sweep a threshold and plot Type I vs Type II error trade-off curves.
# A diamond marks the −10% threshold on each curve.

import numpy as np
import matplotlib.pyplot as plt

# ----------------- Parameters -----------------
years = 10
months = years * 12
sigma_annual = 0.10
SR_good = 0.5
SR_bad = 0.0
p_migrate = 0.005        # 0.5% per month (Panel B)
paths = 100_000          # adjust for speed/precision
rng = np.random.default_rng(20250821)

# Monthly parameters (simple returns)
sigma_m = sigma_annual / np.sqrt(12)
mu_m_good = (SR_good * sigma_annual) / 12
mu_m_bad  = (SR_bad  * sigma_annual) / 12  # = 0

# ----------------- Helpers -----------------
def wealth_from_returns(R):
    """Multiplicative wealth paths, starting at 1.0. R: (n_paths, months). Returns: (n_paths, months+1)."""
    W = np.cumprod(1.0 + R, axis=1)
    return np.c_[np.ones((R.shape[0], 1)), W]

def drawdown_series(W):
    """Drawdown time series from wealth. DD_t = W_t / max_{s<=t} W_s - 1."""
    M = np.maximum.accumulate(W, axis=1)
    return W / M - 1.0

def stats_from_returns(R):
    """Return (TotalReturn, Drawdown_at_T, MaxDrawdown) for each path."""
    W = wealth_from_returns(R)
    DD = drawdown_series(W)
    total_return = W[:, -1] - 1.0
    drawdown_T = DD[:, -1]
    max_drawdown = DD.min(axis=1)
    return total_return, drawdown_T, max_drawdown

def typeI_typeII_from_thresholds(values_good, values_bad, th_grid, replace_when_leq=True):
    """
    Compute Type I (replace | good) and Type II (keep | bad) for thresholds in th_grid.
    replace_when_leq=True means we 'replace' when statistic <= threshold (lower is worse).
    """
    tI, tII = [], []
    for thr in th_grid:
        if replace_when_leq:
            ti  = (values_good <= thr).mean()  # replacing a good manager
            tii = (values_bad  >  thr).mean()  # keeping a bad manager
        else:
            ti  = (values_good >= thr).mean()
            tii = (values_bad  <  thr).mean()
        tI.append(ti); tII.append(tii)
    return np.array(tI), np.array(tII)

def nearest(xgrid, x0):
    return xgrid[np.argmin(np.abs(xgrid - x0))]

def mark_pair(thr_grid, typeI, typeII, thr_mark):
    idx = np.argmin(np.abs(thr_grid - thr_mark))
    return typeI[idx], typeII[idx]

# ----------------- Panel A: No migration -----------------
# 50% good, 50% bad
Z_g = rng.standard_normal(size=(paths//2, months))
Z_b = rng.standard_normal(size=(paths//2, months))

R_g = mu_m_good + sigma_m * Z_g
R_b = mu_m_bad  + sigma_m * Z_b

TR_g, DD_T_g, MDD_g = stats_from_returns(R_g)
TR_b, DD_T_b, MDD_b = stats_from_returns(R_b)

# Threshold grids (sweep across reasonable ranges)
thr_total = np.linspace(-0.60, 0.30, 150)   # total return threshold
thr_dd    = np.linspace(-0.60, -0.01, 150)  # drawdown thresholds (negative)

# Type I vs Type II for each rule
ti_tr_A,  tii_tr_A  = typeI_typeII_from_thresholds(TR_g,   TR_b,   thr_total, True)
ti_dd_A,  tii_dd_A  = typeI_typeII_from_thresholds(DD_T_g, DD_T_b, thr_dd,    True)
ti_mdd_A, tii_mdd_A = typeI_typeII_from_thresholds(MDD_g,  MDD_b,  thr_dd,    True)

# Mark the −10% cutoff
cut = -0.10
thr_TR_mark  = nearest(thr_total, cut)
thr_DD_mark  = nearest(thr_dd,    cut)
thr_MDD_mark = nearest(thr_dd,    cut)

mark_TR_A  = mark_pair(thr_total, ti_tr_A,  tii_tr_A,  thr_TR_mark)
mark_DD_A  = mark_pair(thr_dd,    ti_dd_A,  tii_dd_A,  thr_DD_mark)
mark_MDD_A = mark_pair(thr_dd,    ti_mdd_A, tii_mdd_A, thr_MDD_mark)

# Plot Panel A
plt.figure(figsize=(10,6))
plt.plot(ti_tr_A,  tii_tr_A,  '-', label="Total Return")
plt.plot(ti_dd_A,  tii_dd_A,  '-', label="Drawdown (at T)")
plt.plot(ti_mdd_A, tii_mdd_A, '-', label="Maximum Drawdown")
plt.plot(*mark_TR_A,  marker='D', linestyle='None', label="−10% cutoff (Total Return)")
plt.plot(*mark_DD_A,  marker='D', linestyle='None', label="−10% cutoff (Drawdown)")
plt.plot(*mark_MDD_A, marker='D', linestyle='None', label="−10% cutoff (Max DD)")
plt.xlabel("Type I Probability")
plt.ylabel("Type II Probability")
plt.title("Exhibit 6 — Panel A: Type I vs Type II (No Migration)\n(50% good SR=0.5, 50% bad SR=0.0; σ=10% ann.; 10y; normal i.i.d.)")
plt.legend()
plt.tight_layout()
# plt.savefig("exhibit6_panelA.png", dpi=200, bbox_inches="tight")

# ----------------- Panel B: Migration (0.5%/month) -----------------
# Everyone starts good; month of migration K ~ Geometric(p_migrate), 1-based
Z = rng.standard_normal(size=(paths, months))
K = rng.geometric(p_migrate, size=paths)           # 1,2,3,...
K_clip = np.minimum(K, months+1)                   # months+1 => no migration within window
idx = np.arange(months)[None, :]
drift = np.where(idx < (K_clip[:, None]-1), mu_m_good, mu_m_bad)
R = drift + sigma_m * Z

TR, DD_T, MDD = stats_from_returns(R)

# "Truth" at T: good if never migrated within the window
is_good_T = K > months
is_bad_T  = ~is_good_T

TR_gT, TR_bT       = TR[is_good_T],   TR[is_bad_T]
DD_T_gT, DD_T_bT   = DD_T[is_good_T], DD_T[is_bad_T]
MDD_gT, MDD_bT     = MDD[is_good_T],  MDD[is_bad_T]

ti_tr_B,  tii_tr_B  = typeI_typeII_from_thresholds(TR_gT,   TR_bT,   thr_total, True)
ti_dd_B,  tii_dd_B  = typeI_typeII_from_thresholds(DD_T_gT, DD_T_bT, thr_dd,    True)
ti_mdd_B, tii_mdd_B = typeI_typeII_from_thresholds(MDD_gT,  MDD_bT,  thr_dd,    True)

mark_TR_B  = mark_pair(thr_total, ti_tr_B,  tii_tr_B,  thr_TR_mark)
mark_DD_B  = mark_pair(thr_dd,    ti_dd_B,  tii_dd_B,  thr_DD_mark)
mark_MDD_B = mark_pair(thr_dd,    ti_mdd_B, tii_mdd_B, thr_MDD_mark)

# Plot Panel B
plt.figure(figsize=(10,6))
plt.plot(ti_tr_B,  tii_tr_B,  '-', label="Total Return")
plt.plot(ti_dd_B,  tii_dd_B,  '-', label="Drawdown (at T)")
plt.plot(ti_mdd_B, tii_mdd_B, '-', label="Maximum Drawdown")
plt.plot(*mark_TR_B,  marker='D', linestyle='None', label="−10% cutoff (Total Return)")
plt.plot(*mark_DD_B,  marker='D', linestyle='None', label="−10% cutoff (Drawdown)")
plt.plot(*mark_MDD_B, marker='D', linestyle='None', label="−10% cutoff (Max DD)")
plt.xlabel("Type I Probability")
plt.ylabel("Type II Probability")
plt.title("Exhibit 6 — Panel B: Type I vs Type II (Migration 0.5%/mo)\n(Start good SR=0.5; σ=10% ann.; 10y; normal i.i.d.)")
plt.legend()
plt.tight_layout()
# plt.savefig("exhibit6_panelB.png", dpi=200, bbox_inches="tight")

plt.show()

# -------- Optional: print −10% diamond values (Tabular summary) --------
try:
    import pandas as pd
    summary_A = {
        "Rule": ["Total Return","Drawdown (at T)","Maximum Drawdown"],
        "Type I (No migration)": [mark_TR_A[0], mark_DD_A[0], mark_MDD_A[0]],
        "Type II (No migration)": [mark_TR_A[1], mark_DD_A[1], mark_MDD_A[1]],
    }
    summary_B = {
        "Rule": ["Total Return","Drawdown (at T)","Maximum Drawdown"],
        "Type I (Migration)": [mark_TR_B[0], mark_DD_B[0], mark_MDD_B[0]],
        "Type II (Migration)": [mark_TR_B[1], mark_DD_B[1], mark_MDD_B[1]],
    }
    dfA = pd.DataFrame(summary_A).round(4)
    dfB = pd.DataFrame(summary_B).round(4)
    print("\nExhibit 6 — −10% cutoff summary (Panel A):\n", dfA)
    print("\nExhibit 6 — −10% cutoff summary (Panel B):\n", dfB)
except ImportError:
    pass

