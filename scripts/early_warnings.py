
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def _lppl_func(t, A, B, C, m, omega, phi, tc):
    return A + B * (tc - t)**m + C * (tc - t)**m * np.cos(omega * np.log(tc - t) + phi)

def lppl_fit_last_window(prices: pd.Series, horizon: int = 60):
    if len(prices) < horizon + 5:
        return np.nan, np.nan
    t = np.arange(len(prices))
    y = np.log(prices.values + 1e-12)
    t_window = t[-horizon:]
    y_window = y[-horizon:]
    A0 = y_window[-1]
    p0 = [A0, -1.0, 0.1, 0.5, 8.0, 0.0, len(prices) + horizon/2]
    bounds = ([A0-5, -5, -1, 0.1, 6, 0, len(prices)+1],
              [A0+5, 0, 1, 0.99, 13, 2*np.pi, len(prices)+horizon])
    try:
        popt, _ = curve_fit(_lppl_func, t_window, y_window, p0=p0, bounds=bounds, maxfev=5000)
        A, B, C, m, omega, phi, tc = popt
        proximity = max(0, (horizon - (tc - len(prices)))) / horizon
        score = proximity * abs(C)
        return popt, float(score)
    except Exception:
        return np.nan, np.nan

def rolling_lppl_score(prices: pd.Series, horizon: int = 60):
    scores = []
    for i in range(len(prices)):
        if i < horizon:
            scores.append(np.nan)
        else:
            _, s = lppl_fit_last_window(prices.iloc[:i], horizon=horizon)
            scores.append(s)
    return pd.Series(scores, index=prices.index, name="lppl_risk")

def build_ews_features_from_prices(prices: pd.Series):
    rets = prices.pct_change()
    vol20 = rets.rolling(20).std()
    vol60 = rets.rolling(60).std()
    vol_ratio = vol20 / (vol60.replace(0.0, np.nan))

    wealth = (1 + rets.fillna(0)).cumprod()
    dd = wealth / wealth.cummax() - 1
    dd_slope = dd.diff(5)

    sma60 = prices.rolling(60).mean()
    sma_dev = prices / sma60 - 1

    lppl_risk = rolling_lppl_score(prices, horizon=60)

    feats = pd.DataFrame({
        "vol_ratio": vol_ratio,
        "dd_slope": dd_slope,
        "sma_dev": sma_dev,
        "lppl_risk": lppl_risk
    }).dropna()
    return feats

def label_future_drawdown_from_prices(prices: pd.Series, horizon: int = 60, threshold: float = -0.15):
    labels = []
    p = prices.dropna()
    for i in range(len(p)):
        if i + horizon >= len(p):
            labels.append(np.nan)
        else:
            seg = p.iloc[i:i+horizon+1]
            dd = (seg / seg.cummax() - 1).min()
            labels.append(1 if dd <= threshold else 0)
    return pd.Series(labels, index=p.index, name="future_dd")
