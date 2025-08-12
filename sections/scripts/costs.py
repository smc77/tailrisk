
import numpy as np
import pandas as pd

def apply_tc_roundtrip(returns: pd.Series, trade_flags: pd.Series, 
                       fixed_bps: float = 0.0, slippage_bps: float = 0.0):
    returns = returns.copy().fillna(0.0)
    trade_flags = trade_flags.reindex_like(returns).fillna(False)
    per_side = (fixed_bps + slippage_bps) / 1e4
    daily_cost = np.where(trade_flags, -2 * per_side, 0.0)
    return returns + daily_cost

def spread_cost_on_turnover(returns: pd.Series, turnover: pd.Series, spread_bps: float = 0.0):
    returns = returns.copy().fillna(0.0)
    turnover = turnover.reindex_like(returns).fillna(0.0)
    half_spread = (spread_bps / 1e4) / 2.0
    return returns - half_spread * turnover
