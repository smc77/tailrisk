
import numpy as np
import pandas as pd
from math import sqrt, exp
from scipy.stats import norm

def bs_put_price(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return K*exp(-r*T)*norm.cdf(-d2) - S*exp(-q*T)*norm.cdf(-d1)

def rolling_put_hedge(prices: pd.Series, moneyness=0.95, tenor_days=21, r=0.0, q=0.0, iv_annual=0.20):
    rets = prices.pct_change().fillna(0.0)
    dates = prices.index
    last_roll = None
    sigma = iv_annual
    hedged_rets = []
    for i, d in enumerate(dates):
        S = prices.iloc[i]
        if (last_roll is None) or ((d - last_roll).days >= tenor_days):
            K = moneyness * S
            last_roll = d
        if i == 0:
            hedged_rets.append(0.0); continue
        days_left = max(1, tenor_days - (d - last_roll).days)
        T_remaining = days_left/252.0
        if sigma > 0 and T_remaining > 0:
            d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T_remaining)/(sigma*sqrt(T_remaining))
            delta_put = -np.exp(-q*T_remaining)*norm.cdf(-d1)
        else:
            delta_put = -1.0 if S < K else 0.0
        dS = prices.iloc[i] - prices.iloc[i-1]
        d_opt = delta_put * dS
        hr = rets.iloc[i] + (d_opt / prices.iloc[i-1])
        hedged_rets.append(hr)
    return pd.Series(hedged_rets, index=dates)

def vix_overlay(asset_rets: pd.Series, vix_rets: pd.Series, trigger: pd.Series, hedge_weight=0.1):
    asset_rets = asset_rets.fillna(0.0)
    vix_rets = vix_rets.reindex_like(asset_rets).fillna(0.0)
    trig = trigger.reindex_like(asset_rets).fillna(False)
    combined = np.where(trig, (1-hedge_weight)*asset_rets + hedge_weight*vix_rets, asset_rets)
    return pd.Series(combined, index=asset_rets.index)

def apply_option_costs(hedged_rets: pd.Series, roll_flags: pd.Series, bid_ask_bps: float = 0.0):
    hedged_rets = hedged_rets.copy().fillna(0.0)
    roll_flags = roll_flags.reindex_like(hedged_rets).fillna(False)
    per_roll = bid_ask_bps / 1e4
    return hedged_rets + np.where(roll_flags, -per_roll, 0.0)
