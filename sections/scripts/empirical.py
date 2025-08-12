
import pandas as pd

def compute_drawdown_series(prices: pd.Series) -> pd.Series:
    rets = prices.pct_change().dropna()
    wealth = (1 + rets).cumprod()
    return (wealth - wealth.cummax()) / wealth.cummax()
