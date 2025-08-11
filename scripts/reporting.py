
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def wealth_series(returns: pd.Series) -> pd.Series:
    return (1 + returns.dropna()).cumprod()

def compute_summary(returns: pd.Series):
    r = returns.dropna()
    if len(r) < 2:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MDD": np.nan, "AvgDD": np.nan, "TUW_days": np.nan}
    W = wealth_series(r)
    dd = (W - W.cummax()) / W.cummax()
    mdd = float(dd.min())
    avgdd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    tuw = int((dd < 0).sum())
    cagr = float((W.iloc[-1] / W.iloc[0]) ** (252 / len(W)) - 1)
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(cagr / vol) if vol else np.nan
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MDD": mdd, "AvgDD": avgdd, "TUW_days": tuw}

def compare_strategies(returns_dict: dict) -> pd.DataFrame:
    rows = []
    for name, rets in returns_dict.items():
        m = compute_summary(rets)
        m["Strategy"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("Strategy")
    return df

def save_table(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)
    return path

def save_figure(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    return path
