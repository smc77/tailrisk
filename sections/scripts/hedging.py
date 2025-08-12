import numpy as np
import pandas as pd
from math import sqrt, exp
from scipy.stats import norm

def _bs_put_delta(S, K, T, r, q, sigma):
    # Black–Scholes put delta (asset price sensitivity)
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    # Delta of a European put under continuous dividend yield q
    return -np.exp(-q*T) * norm.cdf(-d1)

def rolling_put_hedge(prices: pd.Series, moneyness=0.95, tenor_days=21, r=0.0, q=0.0, iv_annual=0.20):
    """
    Index-agnostic rolling put hedge:
    - Rolls every `tenor_days` steps (not calendar days).
    - Sets strike K = moneyness * spot at roll.
    - Returns the hedged returns series and a boolean roll flag series.
    """
    prices = prices.astype(float)
    rets = prices.pct_change().fillna(0.0)
    n = len(prices)
    if n < 2:
        return rets*0.0, pd.Series(False, index=prices.index)

    hedged_rets = np.zeros(n)
    roll_flags = np.zeros(n, dtype=bool)

    # initialize
    K = moneyness * prices.iloc[0]
    steps_since_roll = 0
    sigma = float(iv_annual)

    for i in range(n):
        if i == 0:
            hedged_rets[i] = 0.0
            roll_flags[i] = True
            continue

        # roll if needed (by step count, not calendar time)
        if steps_since_roll >= tenor_days:
            K = moneyness * prices.iloc[i-1]
            steps_since_roll = 0
            roll_flags[i] = True
        else:
            roll_flags[i] = False

        steps_left = max(1, tenor_days - steps_since_roll)  # at least 1 step left
        T_remaining = steps_left / 252.0

        # hedge via put delta
        delta_put = _bs_put_delta(prices.iloc[i-1], K, T_remaining, r, q, sigma)

        # asset return + option delta * price change (scaled)
        dS = prices.iloc[i] - prices.iloc[i-1]
        opt_pnl_over_S_prev = delta_put * (dS / prices.iloc[i-1])

        hedged_rets[i] = rets.iloc[i] + opt_pnl_over_S_prev
        steps_since_roll += 1

    return pd.Series(hedged_rets, index=prices.index), pd.Series(roll_flags, index=prices.index)

def bs_put_price(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return K*exp(-r*T)*norm.cdf(-d2) - S*exp(-q*T)*norm.cdf(-d1)

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
