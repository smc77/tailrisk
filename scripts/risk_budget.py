
import numpy as np
import pandas as pd

from .stoploss import apply_trailing_stop, apply_costs
from .hedging import vix_overlay

def _metrics(returns: pd.Series):
    rets = returns.dropna()
    if len(rets) < 2:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MDD": np.nan}
    wealth = (1 + rets).cumprod()
    running_max = wealth.cummax()
    dd = (wealth - running_max) / running_max
    mdd = float(dd.min())
    cagr = float((wealth.iloc[-1] / wealth.iloc[0]) ** (252 / len(wealth)) - 1)
    vol = float(rets.std() * np.sqrt(252))
    sharpe = float(cagr / vol) if vol else np.nan
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MDD": mdd}

def build_stop_series(prices: pd.Series, stop_pct: float, costs=None):
    sl_rets, pos, trade, tovr = apply_trailing_stop(prices, stop_pct=stop_pct, reenter_above_peak=True)
    if costs:
        sl_rets = apply_costs(sl_rets, trade, tovr,
                              fixed_bps=costs.get("fixed_bps",0.0),
                              slippage_bps=costs.get("slippage_bps",0.0),
                              spread_bps=costs.get("spread_bps",0.0))
    return sl_rets

def apply_hedge(base_rets: pd.Series, vix_proxy_rets: pd.Series, trigger: pd.Series, hedge_weight: float):
    return vix_overlay(base_rets, vix_proxy_rets, trigger, hedge_weight=float(hedge_weight))

def risk_budget_grid(prices: pd.Series,
                     s_grid = np.linspace(0.02, 0.10, 9),
                     h_grid = np.linspace(0.00, 0.30, 7),
                     vix_proxy_rets: pd.Series = None,
                     trigger: pd.Series = None,
                     costs_stop: dict = None,
                     cagr_floor_ratio: float = 0.80):
    rets = prices.pct_change().dropna()
    bh_m = _metrics(rets)
    cagr_floor = (bh_m["CAGR"] if not np.isnan(bh_m["CAGR"]) else 0.0) * cagr_floor_ratio

    if vix_proxy_rets is None:
        vix_proxy_rets = (-3.0 * rets).reindex_like(rets)
    if trigger is None:
        vol_s = rets.rolling(20).std().shift(1)
        vol_l = rets.rolling(60).std().shift(1)
        trigger = (vol_s > vol_l).reindex_like(rets).fillna(False)

    rows = []
    best = None
    best_key = None
    for s in s_grid:
        sl_rets = build_stop_series(prices, stop_pct=float(s), costs=costs_stop)
        base = sl_rets.reindex_like(rets).fillna(0.0)
        for h in h_grid:
            strat = apply_hedge(base, vix_proxy_rets, trigger, hedge_weight=float(h)).dropna()
            m = _metrics(strat)
            feasible = (not np.isnan(m["CAGR"])) and (m["CAGR"] >= cagr_floor)
            rows.append({"stop_pct": float(s), "hedge_w": float(h), "CAGR": m["CAGR"], "Vol": m["Vol"], "Sharpe": m["Sharpe"], "MDD": m["MDD"], "feasible": feasible})
            if feasible:
                key = (m["MDD"], -m["Vol"])
                if (best is None) or (key < best_key):
                    best = {"stop_pct": float(s), "hedge_w": float(h), **m}
                    best_key = key

    df = pd.DataFrame(rows)
    return best, df, bh_m
