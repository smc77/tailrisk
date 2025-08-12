
import pandas as pd
import numpy as np

def load_prices_yf(ticker: str, start="1990-01-01"):
    import yfinance as yf
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Download failed or empty for {ticker}")
    s = df["Close"]#.rename(ticker)
    print(s.head())
    s.index = pd.to_datetime(s.index)
    s = s.asfreq("B").ffill()
    return s.squeeze()

def pct_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()

def wealth_series(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()

def drawdown_series(returns: pd.Series) -> pd.Series:
    W = wealth_series(returns)
    return (W - W.cummax()) / W.cummax()

def rolling_stats(returns: pd.Series, w_short=20, w_long=60):
    mu = returns.rolling(w_long).mean() * 252
    vol = returns.rolling(w_long).std() * np.sqrt(252)
    rho = returns.rolling(w_long).apply(lambda x: pd.Series(x).autocorr(lag=1))
    return pd.DataFrame({"mu_ann": mu, "vol_ann": vol, "rho1": rho})

def summary_metrics(returns: pd.Series):
    rets = returns.dropna()
    if len(rets) < 2:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MDD": np.nan, "AvgDD": np.nan, "TUW_days": np.nan}
    W = wealth_series(rets)
    dd = (W - W.cummax()) / W.cummax()
    mdd = float(dd.min())
    avgdd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    tuw = int((dd < 0).sum())
    cagr = float((W.iloc[-1] / W.iloc[0]) ** (252 / len(W)) - 1)
    vol = float(rets.std() * np.sqrt(252))
    sharpe = float(cagr / vol) if vol else np.nan
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MDD": mdd, "AvgDD": avgdd, "TUW_days": tuw}
