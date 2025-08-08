# Two-factor AR(1) sims:
# Set A: vary rho with mu_ann=0
# Set B: vary mu_ann with rho=0
# Annual vol fixed at 5%. Trailing stop = 5%, re-enter above prior peak.
# Charts: separate figures for each metric, per set. Marker 'o' = BuyHold, 'x' = Stop5%.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caas_jupyter_tools import display_dataframe_to_user

np.random.seed(7)

YEARS = 10
DAYS = 252 * YEARS
TRIALS = 60

sigma_annual = 0.05
sigma_daily = sigma_annual / np.sqrt(252.0)

def simulate_ar1_path(mu_d, sigma_d, rho, n_steps):
    sigma_eps = sigma_d * np.sqrt(max(1e-12, 1 - rho**2))
    r = np.zeros(n_steps)
    r[0] = mu_d + np.random.normal(0, sigma_d)
    for t in range(1, n_steps):
        eps = np.random.normal(0, sigma_eps)
        r[t] = mu_d + rho * (r[t-1] - mu_d) + eps
    return r

def trailing_stop_returns(price, stop_pct=0.05):
    rets = price.pct_change().fillna(0.0).values
    prices = price.values
    invested = True
    peak = prices[0]
    strat_rets = np.zeros_like(rets)
    for t in range(len(prices)):
        if invested:
            if prices[t] > peak:
                peak = prices[t]
            dd = (prices[t] - peak) / peak
            if dd <= -stop_pct:
                invested = False
        else:
            if prices[t] >= peak:
                invested = True
                peak = prices[t]
        strat_rets[t] = rets[t] if invested else 0.0
    return pd.Series(strat_rets, index=price.index)

def compute_metrics(returns_series: pd.Series):
    rets = returns_series.dropna()
    if len(rets) < 2:
        return dict(CAGR=np.nan, Vol=np.nan, Sharpe=np.nan, MDD=np.nan)
    wealth = (1 + rets).cumprod()
    running_max = wealth.cummax()
    dd = (wealth - running_max) / running_max
    mdd = dd.min()
    cagr = (wealth.iloc[-1] / wealth.iloc[0]) ** (252 / len(wealth)) - 1
    vol_ann = rets.std() * np.sqrt(252)
    sharpe = cagr / vol_ann if vol_ann and not np.isnan(vol_ann) and vol_ann != 0 else np.nan
    return dict(CAGR=cagr, Vol=vol_ann, Sharpe=sharpe, MDD=mdd)

def run_grid(param_values, vary="rho", fixed_mu_ann=0.0, fixed_rho=0.0):
    rows = []
    for val in param_values:
        if vary == "rho":
            rho = float(val); mu_ann = fixed_mu_ann
        else:
            rho = fixed_rho; mu_ann = float(val)
        mu_d = mu_ann / 252.0
        bh_list, sl_list = [], []
        for _ in range(TRIALS):
            r = simulate_ar1_path(mu_d, sigma_daily, rho, DAYS)
            price = pd.Series(100.0 * np.cumprod(1 + r), index=pd.RangeIndex(DAYS))
            bh_rets = price.pct_change().dropna()
            sl_rets = trailing_stop_returns(price, stop_pct=0.05).iloc[1:]
            bh_list.append(compute_metrics(bh_rets))
            sl_list.append(compute_metrics(sl_rets))
        bh_ = pd.DataFrame(bh_list).mean().to_dict()
        sl_ = pd.DataFrame(sl_list).mean().to_dict()
        rows.append({"Strategy":"BuyHold", vary: val, "mu_ann": mu_ann, "rho": rho, **bh_})
        rows.append({"Strategy":"Stop5pct", vary: val, "mu_ann": mu_ann, "rho": rho, **sl_})
    return pd.DataFrame(rows)

# Set A: vary rho
rho_values = np.linspace(-0.5, 0.5, 11)
df_rho = run_grid(rho_values, vary="rho", fixed_mu_ann=0.0)

# Set B: vary mu_ann
mu_values = np.linspace(-0.10, 0.10, 9)
df_mu = run_grid(mu_values, vary="mu_ann", fixed_rho=0.0)

# Save/display tables
display_dataframe_to_user("Set A (vary rho, mu_ann=0): metrics", df_rho)
display_dataframe_to_user("Set B (vary mu_ann, rho=0): metrics", df_mu)

# Plot helpers
def plot_set(df, xcol, title_prefix):
    # Each metric in its own plot
    for metric, ylbl in [("CAGR","CAGR"),("Vol","Annualized Volatility"),("MDD","Max Drawdown"),("Sharpe","Sharpe Ratio")]:
        plt.figure()
        for strat, marker in [("BuyHold",'o'),("Stop5pct",'x')]:
            sub = df[df["Strategy"]==strat]
            plt.scatter(sub[xcol], sub[metric], marker=marker, label=strat, alpha=0.8)
        plt.xlabel(xcol)
        plt.ylabel(ylbl)
        plt.title(f"{title_prefix}: {ylbl} vs {xcol}")
        plt.legend()
        plt.tight_layout()
        plt.show()

plot_set(df_rho, "rho", "Set A (mu_ann=0)")
plot_set(df_mu, "mu_ann", "Set B (rho=0)")

# Export CSVs so you can grab them if needed
df_rho.to_csv("/mnt/data/sim_setA_vary_rho.csv", index=False)
df_mu.to_csv("/mnt/data/sim_setB_vary_mu.csv", index=False)
print("CSV exports ready:",
      "/mnt/data/sim_setA_vary_rho.csv",
      "/mnt/data/sim_setB_vary_mu.csv")
