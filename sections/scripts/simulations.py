
import numpy as np
import pandas as pd

def simulate_ar1(mu_ann=0.0, rho=0.0, sigma_ann=0.05, years=10, seed=None):
    rng = np.random.default_rng(seed)
    T = years * 252
    mu_d = mu_ann / 252.0
    sigma_d = sigma_ann / np.sqrt(252.0)
    sigma_eps = sigma_d * np.sqrt(max(1e-12, 1 - rho**2))
    r = np.zeros(T)
    r[0] = mu_d + rng.normal(0, sigma_d)
    for t in range(1, T):
        eps = rng.normal(0, sigma_eps)
        r[t] = mu_d + rho * (r[t-1] - mu_d) + eps
    price = 100 * np.cumprod(1 + r)
    return pd.Series(price)

def trailing_stop_returns(price_series, stop_pct=0.05):
    rets = price_series.pct_change().fillna(0.0).values
    prices = price_series.values
    invested = True
    peak = prices[0]
    strat = np.zeros_like(rets)
    for i, p in enumerate(prices):
        if invested:
            peak = max(peak, p)
            dd = (p - peak) / peak
            if dd <= -stop_pct:
                invested = False
        else:
            if p >= peak:
                invested = True
                peak = p
        strat[i] = rets[i] if invested else 0.0
    return pd.Series(strat, index=price_series.index)

def metrics(rets):
    rets = pd.Series(rets).dropna()
    if len(rets) < 2:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MDD": np.nan}
    wealth = (1 + rets).cumprod()
    dd = (wealth - wealth.cummax()) / wealth.cummax()
    mdd = float(dd.min())
    cagr = float((wealth.iloc[-1] / wealth.iloc[0]) ** (252 / len(wealth)) - 1)
    vol = float(rets.std() * np.sqrt(252))
    sharpe = float(cagr / vol) if vol else np.nan
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MDD": mdd}
