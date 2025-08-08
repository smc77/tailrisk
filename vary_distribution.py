import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(11)

# Simulation parameters
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
    return dict(CAGR=float(cagr), Vol=float(vol_ann), Sharpe=float(sharpe), MDD=float(mdd))

def run_grid_with_ci(param_values, vary="rho", fixed_mu_ann=0.0, fixed_rho=0.0):
    rows = []
    for val in param_values:
        if vary == "rho":
            rho = float(val); mu_ann = fixed_mu_ann
        else:
            rho = fixed_rho; mu_ann = float(val)
        mu_d = mu_ann / 252.0
        for strat_name in ["BuyHold", "Stop5pct"]:
            metrics_list = []
            for _ in range(TRIALS):
                r = simulate_ar1_path(mu_d, sigma_daily, rho, DAYS)
                price = pd.Series(100.0 * np.cumprod(1 + r), index=pd.RangeIndex(DAYS))
                if strat_name == "BuyHold":
                    rets = price.pct_change().dropna()
                else:
                    rets = trailing_stop_returns(price, stop_pct=0.05).iloc[1:]
                metrics_list.append(compute_metrics(rets))
            dfm = pd.DataFrame(metrics_list).apply(pd.to_numeric, errors="coerce")
            dfm = dfm.dropna()
            for metric in ["CAGR","Vol","MDD","Sharpe"]:
                if metric not in dfm or dfm[metric].empty:
                    mean = np.nan; ci_low = np.nan; ci_high = np.nan
                else:
                    mean = dfm[metric].mean()
                    se = dfm[metric].std(ddof=1) / np.sqrt(len(dfm))
                    ci_low = mean - 1.96 * se
                    ci_high = mean + 1.96 * se
                rows.append({
                    "Strategy": strat_name,
                    vary: float(val),
                    "mu_ann": float(mu_ann),
                    "rho": float(rho),
                    "Metric": metric,
                    "Mean": float(mean) if pd.notna(mean) else np.nan,
                    "CI_low": float(ci_low) if pd.notna(ci_low) else np.nan,
                    "CI_high": float(ci_high) if pd.notna(ci_high) else np.nan
                })
    return pd.DataFrame(rows)

# Run Set A (vary rho, mu=0) and Set B (vary mu, rho=0)
rho_values = np.linspace(-0.5, 0.5, 11)
mu_values = np.linspace(-0.10, 0.10, 9)

df_rho_ci = run_grid_with_ci(rho_values, vary="rho", fixed_mu_ann=0.0)
df_mu_ci  = run_grid_with_ci(mu_values, vary="mu_ann", fixed_rho=0.0)

print(df_rho_ci)
print(df_mu_ci)

def sort_and_clean(sub, xcol):
    sub = sub[[xcol,"Mean","CI_low","CI_high"]].dropna()
    sub = sub.sort_values(xcol)
    # cast to float numpy arrays for plotting
    x = sub[xcol].astype(float).values
    mean = sub["Mean"].astype(float).values
    lo = sub["CI_low"].astype(float).values
    hi = sub["CI_high"].astype(float).values
    return x, mean, lo, hi

def plot_with_ci(df, xcol, title_prefix):
    metrics = ["CAGR","Vol","MDD","Sharpe"]
    for metric in metrics:
        plt.figure()
        # One line per strategy
        for strat in ["BuyHold","Stop5pct"]:
            sub = df[(df["Strategy"]==strat) & (df["Metric"]==metric)]
            x, mean, lo, hi = sort_and_clean(sub, xcol)
            plt.plot(x, mean, label=strat)
            plt.fill_between(x, lo, hi, alpha=0.2)
        plt.xlabel(xcol)
        plt.ylabel(metric)
        plt.title(f"{title_prefix}: {metric} vs {xcol} (95% CI shaded)")
        plt.legend()
        plt.tight_layout()
        plt.show()

plot_with_ci(df_rho_ci, "rho", "Set A (mu_ann=0)")
plot_with_ci(df_mu_ci, "mu_ann", "Set B (rho=0)")
