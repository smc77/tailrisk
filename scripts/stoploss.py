
import numpy as np
import pandas as pd
from .costs import apply_tc_roundtrip, spread_cost_on_turnover

def apply_trailing_stop(prices: pd.Series, stop_pct: float = 0.05, reenter_above_peak: bool = True):
    rets = prices.pct_change().fillna(0.0)
    pos = np.zeros(len(prices), dtype=int)
    trade = np.zeros(len(prices), dtype=bool)
    turnover = np.zeros(len(prices))
    invested = True
    peak = prices.iloc[0]
    pos[0] = 1
    for i, p in enumerate(prices.index):
        price = prices.loc[p]
        if invested:
            if price > peak:
                peak = price
            dd = (price - peak) / peak
            if dd <= -stop_pct:
                invested = False
                trade[i] = True
        else:
            if (reenter_above_peak and price >= peak) or (not reenter_above_peak and rets.loc[p] > 0):
                invested = True
                peak = price
                trade[i] = True
        pos[i] = 1 if invested else 0
        if i > 0 and pos[i] != pos[i-1]:
            turnover[i] = 1.0
    strat_rets = rets * pos
    return strat_rets, pd.Series(pos, index=prices.index), pd.Series(trade, index=prices.index), pd.Series(turnover, index=prices.index)

def apply_costs(strat_rets: pd.Series, trade_flags: pd.Series, turnover: pd.Series,
                fixed_bps: float = 0.0, slippage_bps: float = 0.0, spread_bps: float = 0.0):
    after_tc = apply_tc_roundtrip(strat_rets, trade_flags, fixed_bps=fixed_bps, slippage_bps=slippage_bps)
    after_spread = spread_cost_on_turnover(after_tc, turnover, spread_bps=spread_bps)
    return after_spread
