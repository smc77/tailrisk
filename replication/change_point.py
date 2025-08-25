# Three online break/drawdown detectors on a fat‑tailed AR(1) daily path:
# 1) Bayesian Online Changepoint Detection (BOCPD, Adams & MacKay 2007) with NIG conjugacy
# 2) Particle filter for 2‑state (good/bad) Gaussian regime
# 3) 2‑state Gaussian HMM via EM (Baum‑Welch) and forward‑backward smoothing
#
# We simulate a Student‑t AR(1) daily series (ν=4) with ρ=0.25, μ=10% p.a., σ=10% p.a.,
# compute wealth and drawdown, then for each model produce a plot of wealth with
# detection markers indicating "entering drawdown" (probability crosses a threshold while DD increases).
#
# Notes:
# - Charts use matplotlib only, one plot per figure (no subplots), no explicit colors.
# - Thresholds below can be tuned; defaults chosen for illustration.
#
# -------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import lgamma, pi

rng = np.random.default_rng(20250822)

# ---------- Simulation: AR(1) with Student‑t noise ----------
def simulate_ar1_t_daily(N=252*10, mu_ann=0.10, sigma_ann=0.10, rho=0.25, nu=4, dpy=252, seed=20250822):
    rng_loc = np.random.default_rng(seed)
    mu_d = mu_ann / dpy
    sigma_d = sigma_ann / np.sqrt(dpy)
    scale_t = np.sqrt(nu/(nu-2.0))
    Z = rng_loc.standard_t(df=nu, size=N) / scale_t
    eps = np.sqrt(1.0 - rho**2) * sigma_d * Z
    r = np.empty(N)
    r[0] = mu_d + eps[0]
    for t in range(1, N):
        r[t] = mu_d + rho*(r[t-1] - mu_d) + eps[t]
    idx = pd.bdate_range("2010-01-01", periods=N)
    return pd.Series(r, index=idx, name="r")

# ---------- Utilities ----------
def wealth_and_drawdown(returns: pd.Series):
    W = (1.0 + returns).cumprod()
    P = np.r_[1.0, W.values]
    M = np.maximum.accumulate(P)
    DD = 1.0 - P/M
    # align to returns index (drop t=0 element for convenience when plotting with returns index)
    return pd.Series(P[1:], index=returns.index, name="Wealth"), pd.Series(DD[1:], index=returns.index, name="DD")

# Student‑t log pdf utility (for predictive likelihoods)
def student_t_logpdf(x, nu, mu, sigma):
    # density of StudentT(nu, loc=mu, scale=sigma)
    z = (x - mu) / sigma
    return (lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*np.log(pi*nu) - np.log(sigma)
            - 0.5*(nu+1)*np.log1p((z*z)/nu))

# ---------- 1) BOCPD (Gaussian unknown μ,σ² with NIG prior) ----------
def bocpd_nig(x, hazard_lambda=200.0, R_max=300, mu0=0.0, kappa0=1e-4, alpha0=1e-4, beta0=1e-4):
    """
    x: numpy array of observations
    Returns:
        cp_prob: probability of a changepoint at each time t (run-length = 0)
        runlen_mode: modal run-length per time
    """
    T = len(x)
    # run-length posterior (time t -> vector length R_max+1); start with r=0 at t=0
    R = np.zeros((T+1, R_max+1))
    R[0,0] = 1.0

    # Sufficient stats for each possible run-length r at current time
    # We'll store μ, κ, α, β for r=0..R_max
    mu = np.full(R_max+1, mu0, dtype=float)
    kappa = np.full(R_max+1, kappa0, dtype=float)
    alpha = np.full(R_max+1, alpha0, dtype=float)
    beta = np.full(R_max+1, beta0, dtype=float)

    # base prior (for r=0 updates)
    base = (mu0, kappa0, alpha0, beta0)

    h = 1.0/hazard_lambda  # constant hazard
    cp_prob = np.zeros(T)
    runlen_mode = np.zeros(T, dtype=int)

    for t in range(1, T+1):
        xt = x[t-1]

        # Predictive probabilities for growth from run-length r at t-1
        pred_log = np.empty(R_max+1)
        for r in range(min(t, R_max)+1):
            nu = 2*alpha[r]
            mu_r = mu[r]
            sig = np.sqrt(beta[r]*(kappa[r]+1)/(alpha[r]*kappa[r]))
            pred_log[r] = student_t_logpdf(xt, nu, mu_r, sig)

        # Growth probabilities r -> r+1
        R[t,1:min(t, R_max)+1] = R[t-1,0:min(t, R_max)] * (1-h) * np.exp(pred_log[0:min(t, R_max)])

        # Changepoint probabilities r -> 0 using base prior predictive
        mu_b, k_b, a_b, b_b = base
        nu0 = 2*a_b
        sig0 = np.sqrt(b_b*(k_b+1)/(a_b*k_b))
        pred0 = np.exp(student_t_logpdf(xt, nu0, mu_b, sig0))
        R[t,0] = np.sum(R[t-1,0:min(t, R_max)+1] * h) * pred0

        # Normalize
        Z = np.sum(R[t,:min(t, R_max)+1])
        if Z == 0 or not np.isfinite(Z):
            R[t,:min(t, R_max)+1] = 0
            R[t,0] = 1.0  # reset if numeric issues
            Z = 1.0
        R[t,:] /= Z

        cp_prob[t-1] = R[t,0]
        runlen_mode[t-1] = int(np.argmax(R[t,:min(t, R_max)+1]))

        # Update sufficient stats for next time step
        # Prepare new arrays new_mu for r+1 and base updated for r=0
        new_mu  = np.empty_like(mu)
        new_k   = np.empty_like(kappa)
        new_a   = np.empty_like(alpha)
        new_b   = np.empty_like(beta)

        # r=0 case (new run) from base prior
        # posterior after observing xt
        k1 = k_b + 1
        m1 = (k_b*mu_b + xt)/k1
        a1 = a_b + 0.5
        b1 = b_b + 0.5*(k_b*(xt - mu_b)**2)/k1
        new_mu[0], new_k[0], new_a[0], new_b[0] = m1, k1, a1, b1

        # growth: r -> r+1 for r up to R_max-1
        upto = min(t, R_max)
        for r in range(upto):
            k1 = kappa[r] + 1
            m1 = (kappa[r]*mu[r] + xt)/k1
            a1 = alpha[r] + 0.5
            b1 = beta[r] + 0.5*(kappa[r]*(xt - mu[r])**2)/k1
            new_mu[r+1], new_k[r+1], new_a[r+1], new_b[r+1] = m1, k1, a1, b1

        mu, kappa, alpha, beta = new_mu, new_k, new_a, new_b

    return cp_prob, runlen_mode

# ---------- 2) Particle filter for 2‑state regime (good/bad) ----------
def particle_filter_2state(x, Np=300, p_stay=0.995, mu_good=None, mu_bad=None, sigma=None, seed=20250822):
    """
    Simple bootstrap PF with binary latent state s_t in {0,1}; Gaussian emissions N(mu_s, sigma^2).
    Returns posterior P(s_t=1 | x_{1:t}) for each t.
    """
    rng_loc = np.random.default_rng(seed)
    T = len(x)
    if sigma is None: sigma = np.std(x, ddof=1)
    if mu_good is None: mu_good = np.mean(x)
    if mu_bad  is None: mu_bad  = -abs(mu_good)

    # particles: states and weights
    s = rng_loc.integers(0, 2, size=Np)  # start equally likely
    w = np.full(Np, 1.0/Np)

    prob_bad = np.zeros(T)
    for t in range(T):
        # propagate
        flip = rng_loc.random(Np) > p_stay
        s = np.where(flip, 1 - s, s)
        # weights
        mu_emit = np.where(s==0, mu_good, mu_bad)
        ll = -0.5*np.log(2*np.pi*sigma**2) - 0.5*((x[t]-mu_emit)**2)/(sigma**2)
        w *= np.exp(ll - np.max(ll))  # stabilize
        w_sum = np.sum(w)
        if w_sum == 0 or not np.isfinite(w_sum):
            w = np.full(Np, 1.0/Np)
        else:
            w /= w_sum
        prob_bad[t] = np.sum(w*(s==1))
        # resample (systematic)
        u0 = rng_loc.random()/Np
        c = np.cumsum(w)
        idx = np.searchsorted(c, u0 + np.arange(Np)/Np)
        s = s[idx]
        w = np.full(Np, 1.0/Np)
    return prob_bad

# ---------- 3) 2‑state Gaussian HMM (EM + smoothing) ----------
def hmm_2state_fit_smooth(x, iters=12, tol=1e-5):
    """
    Fit 2‑state Gaussian HMM via EM; return smoothed P(state=1 | x_{1:T}) with state 1 designated
    as the 'bad' (lower‑mean) state.
    """
    T = len(x)
    # init
    mu1, mu2 = np.percentile(x, [30, 70])
    if mu1 > mu2: mu1, mu2 = mu2, mu1
    var1 = var2 = np.var(x, ddof=1)
    A = np.array([[0.99, 0.01],
                  [0.01, 0.99]], dtype=float)
    pi0 = np.array([0.5, 0.5])

    def gauss_ll(x, m, v):
        return -0.5*np.log(2*np.pi*v) - 0.5*((x-m)**2)/v

    for _ in range(iters):
        # E-step: forward
        logB1 = gauss_ll(x, mu1, var1)
        logB2 = gauss_ll(x, mu2, var2)
        logA = np.log(A)
        logpi = np.log(pi0 + 1e-16)
        alpha_log = np.zeros((T,2))
        alpha_log[0,0] = logpi[0] + logB1[0]
        alpha_log[0,1] = logpi[1] + logB2[0]
        # scale to avoid underflow
        c0 = np.logaddexp.reduce(alpha_log[0])
        alpha_log[0] -= c0
        cs = [c0]
        for t in range(1, T):
            for j in range(2):
                alpha_log[t,j] = np.logaddexp(alpha_log[t-1,0] + logA[0,j],
                                              alpha_log[t-1,1] + logA[1,j]) + (logB1[t] if j==0 else logB2[t])
            ct = np.logaddexp.reduce(alpha_log[t])
            alpha_log[t] -= ct
            cs.append(ct)
        # Backward
        beta_log = np.zeros((T,2))
        for t in range(T-2, -1, -1):
            for i in range(2):
                beta_log[t,i] = np.logaddexp(
                    logA[i,0] + (logB1[t+1] + beta_log[t+1,0]),
                    logA[i,1] + (logB2[t+1] + beta_log[t+1,1])
                )
            # normalize with same scaling
            beta_log[t] -= cs[t+1]
        # Gammas and Xis
        gamma = np.exp(alpha_log + beta_log)
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi_num = np.zeros((2,2))
        for t in range(T-1):
            denom = np.logaddexp.reduce([alpha_log[t,0]+logA[0,0]+logB1[t+1]+beta_log[t+1,0],
                                         alpha_log[t,0]+logA[0,1]+logB2[t+1]+beta_log[t+1,1],
                                         alpha_log[t,1]+logA[1,0]+logB1[t+1]+beta_log[t+1,0],
                                         alpha_log[t,1]+logA[1,1]+logB2[t+1]+beta_log[t+1,1]])
            xi_num[0,0] += np.exp(alpha_log[t,0]+logA[0,0]+logB1[t+1]+beta_log[t+1,0]-denom)
            xi_num[0,1] += np.exp(alpha_log[t,0]+logA[0,1]+logB2[t+1]+beta_log[t+1,1]-denom)
            xi_num[1,0] += np.exp(alpha_log[t,1]+logA[1,0]+logB1[t+1]+beta_log[t+1,0]-denom)
            xi_num[1,1] += np.exp(alpha_log[t,1]+logA[1,1]+logB2[t+1]+beta_log[t+1,1]-denom)
        # M-step
        pi0 = gamma[0] / gamma[0].sum()
        A = xi_num / xi_num.sum(axis=1, keepdims=True)
        mu1 = np.sum(gamma[:,0]*x)/np.sum(gamma[:,0])
        mu2 = np.sum(gamma[:,1]*x)/np.sum(gamma[:,1])
        var1 = np.sum(gamma[:,0]*(x-mu1)**2)/np.sum(gamma[:,0])
        var2 = np.sum(gamma[:,1]*(x-mu2)**2)/np.sum(gamma[:,1])
        if abs(mu1 - mu2) < tol:
            break

    # Ensure state 1 is 'bad' (lower mean)
    if mu1 < mu2:
        bad_ix = 0
    else:
        bad_ix = 1
    return gamma[:, bad_ix]  # smoothed prob of bad state

# ---------- Detection wrapper ----------
def detect_entering_drawdown(returns, method="bocpd",
                             dd_increase=True,
                             threshold=0.5,
                             **kwargs):
    """
    returns: pd.Series
    method: "bocpd", "pf", "hmm"
    dd_increase: require that drawdown increases from t-1 to t for an alarm
    threshold: probability threshold to raise alarm
    kwargs: method-specific params
      - bocpd: hazard_lambda, R_max, mu0, kappa0, alpha0, beta0
      - pf: Np, p_stay, mu_good, mu_bad, sigma
      - hmm: iters, tol
    Returns: (prob_series, alarm_boolean_series)
    """
    W, DD = wealth_and_drawdown(returns)
    x = returns.values

    if method == "bocpd":
        cp_prob, _ = bocpd_nig(x, **kwargs)
        prob = pd.Series(cp_prob, index=returns.index, name="prob")
        if dd_increase:
            dd_inc = DD.diff().fillna(0.0) > 0
            alarm = (prob > threshold) & dd_inc
        else:
            alarm = (prob > threshold)
        return prob, alarm
    elif method == "pf":
        prob_bad = particle_filter_2state(x, **kwargs)
        prob = pd.Series(prob_bad, index=returns.index, name="prob_bad")
        if dd_increase:
            dd_inc = DD.diff().fillna(0.0) > 0
            alarm = (prob > threshold) & dd_inc
        else:
            alarm = (prob > threshold)
        return prob, alarm
    elif method == "hmm":
        prob_bad = hmm_2state_fit_smooth(x, **kwargs)
        prob = pd.Series(prob_bad, index=returns.index, name="prob_bad")
        if dd_increase:
            dd_inc = DD.diff().fillna(0.0) > 0
            alarm = (prob > threshold) & dd_inc
        else:
            alarm = (prob > threshold)
        return prob, alarm
    else:
        raise ValueError("Unknown method")

# ---------- Run demo ----------
rets = simulate_ar1_t_daily(N=252*10, mu_ann=0.10, sigma_ann=0.10, rho=0.25, nu=4, dpy=252, seed=20250822)
W, DD = wealth_and_drawdown(rets)

# 1) BOCPD
prob_bocpd, alarm_bocpd = detect_entering_drawdown(
    rets, method="bocpd",
    threshold=0.25,     # changepoint probability threshold
    hazard_lambda=200.0,
    R_max=300,
    mu0=0.0, kappa0=1e-4, alpha0=1e-4, beta0=1e-4
)

# 2) Particle filter
prob_pf, alarm_pf = detect_entering_drawdown(
    rets, method="pf",
    threshold=0.6,      # posterior "bad regime" probability
    Np=300, p_stay=0.995
)

# 3) HMM
prob_hmm, alarm_hmm = detect_entering_drawdown(
    rets, method="hmm",
    threshold=0.6,      # smoothed "bad regime" probability
    iters=12, tol=1e-5
)

# ---------- Plot 1: Wealth + BOCPD alarms ----------
fig = plt.figure(figsize=(13,4))
plt.plot(W.index, W.values, label="Wealth")
plt.scatter(W.index[alarm_bocpd.values], W.values[alarm_bocpd.values], marker="x", label="BOCPD alarm")
plt.title("BOCPD detection on Wealth (arrows mark alarms)")
plt.xlabel("Date"); plt.ylabel("Wealth"); plt.legend()
plt.tight_layout(); plt.show()

# ---------- Plot 2: Wealth + PF alarms ----------
fig = plt.figure(figsize=(13,4))
plt.plot(W.index, W.values, label="Wealth")
plt.scatter(W.index[alarm_pf.values], W.values[alarm_pf.values], marker="x", label="PF alarm")
plt.title("Particle Filter detection on Wealth (arrows mark alarms)")
plt.xlabel("Date"); plt.ylabel("Wealth"); plt.legend()
plt.tight_layout(); plt.show()

# ---------- Plot 3: Wealth + HMM alarms ----------
fig = plt.figure(figsize=(13,4))
plt.plot(W.index, W.values, label="Wealth")
plt.scatter(W.index[alarm_hmm.values], W.values[alarm_hmm.values], marker="x", label="HMM alarm")
plt.title("HMM detection on Wealth (arrows mark alarms)")
plt.xlabel("Date"); plt.ylabel("Wealth"); plt.legend()
plt.tight_layout(); plt.show()

# ---------- Optional: show the probability series as separate plots ----------
fig = plt.figure(figsize=(13,3))
plt.plot(prob_bocpd.index, prob_bocpd.values, label="BOCPD CP prob")
plt.title("BOCPD: P(changepoint at t)")
plt.xlabel("Date"); plt.ylabel("Probability"); plt.legend()
plt.tight_layout(); plt.show()

fig = plt.figure(figsize=(13,3))
plt.plot(prob_pf.index, prob_pf.values, label="PF: P(bad regime)")
plt.title("Particle Filter: P(bad regime)")
plt.xlabel("Date"); plt.ylabel("Probability"); plt.legend()
plt.tight_layout(); plt.show()

fig = plt.figure(figsize=(13,3))
plt.plot(prob_hmm.index, prob_hmm.values, label="HMM: P(bad regime, smoothed)")
plt.title("HMM: P(bad regime)")
plt.xlabel("Date"); plt.ylabel("Probability"); plt.legend()
plt.tight_layout(); plt.show()



#
#
#

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Simulate daily AR(1) Student-t ---
def simulate_ar1_t_daily(N, mu_ann, sigma_ann, rho, nu=4, dpy=252, seed=123):
    rng = np.random.default_rng(seed)
    mu_d = mu_ann / dpy
    sigma_d = sigma_ann / np.sqrt(dpy)
    scale_t = np.sqrt(nu/(nu-2.0))
    Z = rng.standard_t(df=nu, size=N) / scale_t
    eps = np.sqrt(1.0 - rho**2) * sigma_d * Z
    r = np.empty(N)
    r[0] = mu_d + eps[0]
    for t in range(1, N):
        r[t] = mu_d + rho*(r[t-1] - mu_d) + eps[t]
    idx = pd.bdate_range("2000-01-03", periods=N)
    return pd.Series(r, index=idx, name="r")

# --- Setup simulation ---
years, dpy = 20, 252
N = years * dpy
rets = simulate_ar1_t_daily(N, mu_ann=0.10, sigma_ann=0.05, rho=0.25, nu=4, dpy=dpy, seed=20250901)

# Wealth & drawdown
W = (1 + rets).cumprod()
P = np.r_[1.0, W.values]
M = np.maximum.accumulate(P)
DD = pd.Series(1.0 - (P[1:]/M[1:]), index=rets.index)
target = (DD > 0.025).astype(int)  # Event: DD > 2.5%

# --- Rule 1: Simple stop-loss (K=0.5σ ≈ 2.5%) ---
rule1_signal = (DD > 0.025).astype(int)  # exit when threshold breached
# For event classification, treat signal=1 as "predicting drawdown"
rule1_pred = rule1_signal.values

# --- Rule 2: Logistic regression probability model ---
# Features: lag1, lag2, rolling mean & vol, dd_prev
lag1 = rets.shift(1)
lag2 = rets.shift(2)
roll_mean5 = rets.rolling(5).mean().shift(1)
roll_vol20 = rets.rolling(20).std().shift(1)
dd_prev = DD.shift(1)

df = pd.DataFrame({
    "lag1": lag1, "lag2": lag2,
    "roll_mean5": roll_mean5, "roll_vol20": roll_vol20,
    "dd_prev": dd_prev
}).dropna()
X = df.values
y = target.loc[df.index].values

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
clf = LogisticRegression(max_iter=500, class_weight="balanced")
clf.fit(X_scaled, y)
probs = clf.predict_proba(X_scaled)[:,1]
rule2_pred = (probs > 0.5).astype(int)

# Align rule1 to df.index
rule1_pred_aligned = rule1_signal.loc[df.index].values

# --- Confusion matrices ---
def confusion_metrics(y_true, y_pred):
    TP = np.sum((y_true==1) & (y_pred==1))
    TN = np.sum((y_true==0) & (y_pred==0))
    FP = np.sum((y_true==0) & (y_pred==1))
    FN = np.sum((y_true==1) & (y_pred==0))
    fpr = FP / (FP + TN) if (FP+TN)>0 else np.nan
    fnr = FN / (FN + TP) if (FN+TP)>0 else np.nan
    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN,"FPR":fpr,"FNR":fnr}

metrics_rule1 = confusion_metrics(y, rule1_pred_aligned)
metrics_rule2 = confusion_metrics(y, rule2_pred)

summary = pd.DataFrame([metrics_rule1, metrics_rule2], index=["StopLoss_K=0.5σ","Model_Prob>0.5"])

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Type I vs Type II error comparison", summary)



# Write a full, self-contained script with expanding-window CV, plots, and exportable outputs.
full_script = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drawdown classification with rich features + heavy expanding-window CV.
- Simulates a daily AR(1)-Student-t series (ν=4)
- Builds leak-free features including BOCPD-knownσ and 2-state particle-filter probabilities
- Defines target as cumulative drawdown > 2.5%
- Runs expanding-window CV, producing OOS probabilities & metrics
- Trains on full data to compute standardized logistic coefficients and tree importances
- Saves figures and CSV outputs to ./outputs/

Usage:
    python drawdown_classifier_full_cv.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.base import clone

# ---------------- Simulation ----------------
def simulate_ar1_t_daily(N, mu_ann, sigma_ann, rho, nu=4, dpy=252, seed=20250822):
    rng_loc = np.random.default_rng(seed)
    mu_d = mu_ann / dpy
    sigma_d = sigma_ann / np.sqrt(dpy)
    scale_t = np.sqrt(nu/(nu-2.0))
    Z = rng_loc.standard_t(df=nu, size=N) / scale_t
    eps = np.sqrt(1.0 - rho**2) * sigma_d * Z
    r = np.empty(N)
    r[0] = mu_d + eps[0]
    for t in range(1, N):
        r[t] = mu_d + rho*(r[t-1] - mu_d) + eps[t]
    idx = pd.bdate_range("2000-01-03", periods=N)
    return pd.Series(r, index=idx, name="r")

# ---------------- Particle Filter (2-state) ----------------
def particle_filter_2state(x, Np=300, p_stay=0.995, mu_good=None, mu_bad=None, sigma=None, seed=20250822):
    rng_loc = np.random.default_rng(seed)
    T = len(x)
    if sigma is None: sigma = np.std(x, ddof=1)
    if mu_good is None: mu_good = np.mean(x)
    if mu_bad  is None: mu_bad  = -abs(mu_good)
    s = rng_loc.integers(0, 2, size=Np)
    w = np.full(Np, 1.0/Np)
    prob_bad = np.zeros(T)
    for t in range(T):
        flip = rng_loc.random(Np) > p_stay
        s = np.where(flip, 1 - s, s)
        mu_emit = np.where(s==0, mu_good, mu_bad)
        ll = -0.5*np.log(2*np.pi*sigma**2) - 0.5*((x[t]-mu_emit)**2)/(sigma**2)
        w *= np.exp(ll - np.max(ll))
        w_sum = np.sum(w)
        if w_sum == 0 or not np.isfinite(w_sum):
            w = np.full(Np, 1.0/Np)
        else:
            w /= w_sum
        prob_bad[t] = np.sum(w*(s==1))
        # systematic resample
        u0 = rng_loc.random()/Np
        c = np.cumsum(w)
        idx = np.searchsorted(c, u0 + np.arange(Np)/Np)
        s = s[idx]
        w = np.full(Np, 1.0/Np)
    return prob_bad

# ---------------- BOCPD with known variance (stable) ----------------
def bocpd_knownvar(x, sigma2, hazard_lambda=200.0, R_max=300, mu0=0.0, kappa0=10.0):
    T = len(x)
    R = np.zeros((T+1, R_max+1)); R[0,0] = 1.0
    mu = np.full(R_max+1, mu0, dtype=float)
    kappa = np.full(R_max+1, kappa0, dtype=float)
    h = 1.0 / hazard_lambda
    cp_prob = np.zeros(T)
    for t in range(1, T+1):
        xt = x[t-1]
        upto = min(t, R_max)
        # Predictive log-likelihood
        pred_log = np.empty(upto+1)
        for r in range(upto+1):
            var = sigma2*(1.0 + 1.0/kappa[r])
            pred_log[r] = -0.5*np.log(2*np.pi*var) - 0.5*((xt - mu[r])**2)/var
        # Growth r->r+1
        R[t,1:upto+1] = R[t-1,0:upto] * (1-h) * np.exp(pred_log[0:upto])
        # CP r->0
        var0 = sigma2*(1.0 + 1.0/kappa0)
        pred0 = np.exp(-0.5*np.log(2*np.pi*var0) - 0.5*((xt - mu0)**2)/var0)
        R[t,0] = np.sum(R[t-1,0:upto+1] * h) * pred0
        # Normalize
        Z = np.sum(R[t,0:upto+1])
        if not np.isfinite(Z) or Z == 0:
            R[t,0] = 1.0; Z = 1.0
        R[t,:] /= Z
        cp_prob[t-1] = R[t,0]
        # Update sufficient stats
        new_mu = np.empty_like(mu); new_k = np.empty_like(kappa)
        k1 = kappa0 + 1.0; m1 = (kappa0*mu0 + xt)/k1
        new_mu[0] = m1; new_k[0] = k1
        for r in range(upto):
            k1 = kappa[r] + 1.0; m1 = (kappa[r]*mu[r] + xt)/k1
            new_mu[r+1] = m1; new_k[r+1] = k1
        mu, kappa = new_mu, new_k
    return cp_prob

# ---------------- Expanding-window CV ----------------
def expanding_splits(n, n_splits=6, min_train_frac=0.35):
    start = int(n * min_train_frac)
    block = (n - start) // n_splits
    splits = []
    t0 = start
    for i in range(n_splits):
        t1 = t0 + block if i < n_splits - 1 else n
        train_idx = np.arange(0, t0)
        test_idx = np.arange(t0, t1)
        splits.append((train_idx, test_idx))
        t0 = t1
    return splits

def eval_model(clf, X, y, n_splits=6):
    splits = expanding_splits(len(y), n_splits=n_splits, min_train_frac=0.35)
    y_prob = np.zeros(len(y)); y_pred = np.zeros(len(y))
    for train_idx, test_idx in splits:
        # ensure both classes in train; if not, push boundary forward
        unique = np.unique(y[train_idx])
        t_start = test_idx[0]; t_end = test_idx[-1] + 1
        while len(unique) < 2 and t_start < t_end - 10:
            t_start += 10
            train_idx = np.arange(0, t_start)
            unique = np.unique(y[train_idx])
            test_idx = np.arange(t_start, t_end)
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx]); y_train = y[train_idx]
        X_test  = scaler.transform(X[test_idx])
        model = clone(clf); model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:,1]
        y_prob[test_idx] = prob
        y_pred[test_idx] = (prob > 0.5).astype(int)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    return y_prob, y_pred, report, cm, auc

# ---------------- Main ----------------
def main():
    # ---- Config ----
    dpy = 252; years = 20; N = years * dpy
    mu_ann = 0.10; sigma_ann = 0.05; rho = 0.25
    outdir = "./outputs"
    os.makedirs(outdir, exist_ok=True)

    # ---- Simulate ----
    rets = simulate_ar1_t_daily(N, mu_ann, sigma_ann, rho, nu=4, dpy=dpy, seed=20250822)

    # ---- Wealth & drawdown; target ----
    W = (1.0 + rets).cumprod()
    P = np.r_[1.0, W.values]; M = np.maximum.accumulate(P)
    DD = pd.Series(1.0 - (P[1:]/M[1:]), index=rets.index, name="DD")
    y = (DD > 0.025).astype(int)

    # ---- Features (lag to avoid look-ahead) ----
    lag1 = rets.shift(1); lag2 = rets.shift(2); lag5 = rets.shift(5)
    roll_mean5 = rets.rolling(5).mean().shift(1)
    roll_mean21 = rets.rolling(21).mean().shift(1)
    roll_vol20 = rets.rolling(20).std().shift(1)
    # EWMA vol annualized
    lam = 0.94; ewvar = np.zeros(len(rets)); prev = 0.0
    for i, ri in enumerate(rets.fillna(0.0).values):
        prev = lam*prev + (1-lam)*(ri**2)
        ewvar[i] = prev
    ewma_vol = pd.Series(np.sqrt(ewvar)*np.sqrt(dpy), index=rets.index).shift(1)
    # From-peak & lagged DD
    peak = pd.Series(np.maximum.accumulate(np.r_[1.0, W.values])[1:], index=rets.index)
    from_peak = W/peak - 1.0
    from_peak_lag = from_peak.shift(1)
    dd_prev = DD.shift(1)
    # BOCPD-knownσ & PF (lagged 1d)
    sigma2_known = rets.var(ddof=1)
    bocpd_cp = pd.Series(bocpd_knownvar(rets.values, sigma2=sigma2_known,
                                        hazard_lambda=200.0, R_max=250, mu0=0.0, kappa0=10.0),
                         index=rets.index).shift(1)
    pf_bad = pd.Series(particle_filter_2state(rets.values, Np=300, p_stay=0.995),
                       index=rets.index).shift(1)

    feat_df = pd.DataFrame({
        "lag1": lag1, "lag2": lag2, "lag5": lag5,
        "roll_mean5": roll_mean5, "roll_mean21": roll_mean21,
        "roll_vol20": roll_vol20, "ewma_vol": ewma_vol,
        "dd_prev": dd_prev, "from_peak_lag": from_peak_lag,
        "bocpd_cp": bocpd_cp, "pf_bad": pf_bad,
    }).dropna()

    X_all = feat_df.values
    y_all = y.loc[feat_df.index].values

    # ---- Models ----
    models = {
        "Logistic": LogisticRegression(max_iter=500, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42),
        "GBM": GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
    }

    # ---- Heavy CV evaluation (expanding-window) ----
    results = {}
    for name, clf in models.items():
        y_prob, y_pred, report, cm, auc = eval_model(clf, X_all, y_all, n_splits=8)
        results[name] = {"prob": y_prob, "pred": y_pred, "report": report, "cm": cm, "auc": auc}

    # ---- Save metrics ----
    rows = []
    for name, res in results.items():
        rep = res["report"]
        rows.append({
            "Model": name,
            "Accuracy": rep["accuracy"],
            "AUC": res["auc"],
            "DD Precision": rep["1"]["precision"],
            "DD Recall": rep["1"]["recall"],
            "DD F1": rep["1"]["f1-score"],
            "NoDD Precision": rep["0"]["precision"],
            "NoDD Recall": rep["0"]["recall"],
            "NoDD F1": rep["0"]["f1-score"],
            "TP": int(res["cm"][1,1]), "FP": int(res["cm"][0,1]),
            "FN": int(res["cm"][1,0]), "TN": int(res["cm"][0,0]),
        })
    summary_df = pd.DataFrame(rows).set_index("Model")
    summary_path = os.path.join(outdir, "cv_summary.csv")
    summary_df.to_csv(summary_path, float_format="%.6f")

    # ---- ROC curves ----
    fig = plt.figure(figsize=(7,7))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_all, res["prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
    plt.plot([0,1],[0,1],'--')
    plt.title("ROC — Drawdown (>2.5%) with Rich Features")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    fig.tight_layout()
    roc_path = os.path.join(outdir, "roc.png")
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    # ---- Probability timeline ----
    fig = plt.figure(figsize=(14,4))
    plt.plot(feat_df.index, y_all, label="Actual DD>2.5%", alpha=0.7)
    for name, res in results.items():
        plt.plot(feat_df.index, res["prob"], label=f"{name} prob", alpha=0.8)
    plt.title("Predicted Probabilities vs Actual State — Rich Features")
    plt.ylabel("Probability / State"); plt.xlabel("Date"); plt.legend()
    fig.tight_layout()
    prob_path = os.path.join(outdir, "prob_timeline.png")
    fig.savefig(prob_path, dpi=150)
    plt.close(fig)

    # ---- Fit on full data for importances/coefficients ----
    scaler_full = StandardScaler().fit(X_all)
    X_scaled = scaler_full.transform(X_all)
    logit = LogisticRegression(max_iter=500, class_weight="balanced")
    logit.fit(X_scaled, y_all)
    coef_series = pd.Series(logit.coef_[0], index=feat_df.columns).sort_values(key=np.abs, ascending=False)

    rf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42)
    rf.fit(X_all, y_all)
    rf_imp = pd.Series(rf.feature_importances_, index=feat_df.columns).sort_values(ascending=False)

    gbm = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
    gbm.fit(X_all, y_all)
    gbm_imp = pd.Series(gbm.feature_importances_, index=feat_df.columns).sort_values(ascending=False)

    # ---- Save importances to CSV ----
    coef_series.to_csv(os.path.join(outdir, "logit_coefficients.csv"), header=["coef"])
    rf_imp.to_csv(os.path.join(outdir, "rf_importances.csv"), header=["importance"])
    gbm_imp.to_csv(os.path.join(outdir, "gbm_importances.csv"), header=["importance"])

    # ---- Save feature matrix and targets ----
    feat_df.to_csv(os.path.join(outdir, "features.csv"))
    pd.Series(y_all, index=feat_df.index, name="target_dd_gt_2p5").to_csv(os.path.join(outdir, "target.csv"))

    # ---- Also save the OOS probabilities for each model ----
    for name, res in results.items():
        pd.Series(res["prob"], index=feat_df.index, name=f"prob_{name}").to_csv(
            os.path.join(outdir, f"oos_prob_{name}.csv")
        )

    print("Saved outputs to:", os.path.abspath(outdir))

if __name__ == "__main__":
    main()
'''
path = "/mnt/data/drawdown_classifier_full_cv.py"
with open(path, "w") as f:
    f.write(full_script)

print(f"Saved full script to {path}")


# Fast path: skip CV evaluation; fit models on full dataset for importances/coeffs, plot, and save script.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Ensure feat_df, X_all, y_all exist (if not, rebuild quickly)
try:
    feat_df, X_all, y_all
except NameError:
    # Minimal rebuild using variables likely still present (rets, W, DD, y)
    lag1 = rets.shift(1); lag2 = rets.shift(2); lag5 = rets.shift(5)
    roll_mean5 = rets.rolling(5).mean().shift(1)
    roll_mean21 = rets.rolling(21).mean().shift(1)
    roll_vol20 = rets.rolling(20).std().shift(1)
    lam = 0.94
    ewvar = np.zeros(len(rets)); prev = 0.0
    for i, ri in enumerate(rets.fillna(0.0).values):
        prev = lam*prev + (1-lam)*(ri**2)
        ewvar[i] = prev
    ewma_vol = pd.Series(np.sqrt(ewvar)*np.sqrt(252), index=rets.index).shift(1)
    peak = pd.Series(np.maximum.accumulate(np.r_[1.0, W.values])[1:], index=rets.index)
    from_peak = W / peak - 1.0
    from_peak_lag = from_peak.shift(1)
    dd_prev = DD.shift(1)
    # BOCPD-knownσ and PF (recompute quickly with compact R_max / Np)
    def bocpd_knownvar(x, sigma2, hazard_lambda=200.0, R_max=200, mu0=0.0, kappa0=10.0):
        T = len(x)
        R = np.zeros((T+1, R_max+1)); R[0,0] = 1.0
        mu = np.full(R_max+1, mu0, dtype=float)
        kappa = np.full(R_max+1, kappa0, dtype=float)
        h = 1.0 / hazard_lambda
        cp_prob = np.zeros(T)
        for t in range(1, T+1):
            xt = x[t-1]
            upto = min(t, R_max)
            pred_log = np.empty(upto+1)
            for r in range(upto+1):
                var = sigma2*(1.0 + 1.0/kappa[r])
                pred_log[r] = -0.5*np.log(2*np.pi*var) - 0.5*((xt - mu[r])**2)/var
            R[t,1:upto+1] = R[t-1,0:upto] * (1-h) * np.exp(pred_log[0:upto])
            var0 = sigma2*(1.0 + 1.0/kappa0)
            pred0 = np.exp(-0.5*np.log(2*np.pi*var0) - 0.5*((xt - mu0)**2)/var0)
            R[t,0] = np.sum(R[t-1,0:upto+1] * h) * pred0
            Z = np.sum(R[t,0:upto+1])
            if not np.isfinite(Z) or Z == 0:
                R[t,0] = 1.0; Z = 1.0
            R[t,:] /= Z
            cp_prob[t-1] = R[t,0]
            new_mu = np.empty_like(mu); new_k = np.empty_like(kappa)
            k1 = kappa0 + 1.0; m1 = (kappa0*mu0 + xt)/k1
            new_mu[0] = m1; new_k[0] = k1
            for r in range(upto):
                k1 = kappa[r] + 1.0; m1 = (kappa[r]*mu[r] + xt)/k1
                new_mu[r+1] = m1; new_k[r+1] = k1
            mu, kappa = new_mu, new_k
        return cp_prob
    def particle_filter_2state(x, Np=200, p_stay=0.995, mu_good=None, mu_bad=None, sigma=None, seed=20250822):
        rng_loc = np.random.default_rng(seed)
        T = len(x)
        if sigma is None: sigma = np.std(x, ddof=1)
        if mu_good is None: mu_good = np.mean(x)
        if mu_bad  is None: mu_bad  = -abs(mu_good)
        s = rng_loc.integers(0, 2, size=Np); w = np.full(Np, 1.0/Np)
        prob_bad = np.zeros(T)
        for t in range(T):
            flip = rng_loc.random(Np) > p_stay
            s = np.where(flip, 1 - s, s)
            mu_emit = np.where(s==0, mu_good, mu_bad)
            ll = -0.5*np.log(2*np.pi*sigma**2) - 0.5*((x[t]-mu_emit)**2)/(sigma**2)
            w *= np.exp(ll - np.max(ll))
            w_sum = np.sum(w); w = np.full(Np, 1.0/Np) if (w_sum==0 or not np.isfinite(w_sum)) else (w / w_sum)
            prob_bad[t] = np.sum(w*(s==1))
            u0 = rng_loc.random()/Np; c = np.cumsum(w)
            idx = np.searchsorted(c, u0 + np.arange(Np)/Np)
            s = s[idx]; w = np.full(Np, 1.0/Np)
        return prob_bad
    sigma2_known = rets.var(ddof=1)
    bocpd_cp = pd.Series(bocpd_knownvar(rets.values, sigma2=sigma2_known, hazard_lambda=200.0, R_max=150, mu0=0.0, kappa0=10.0),
                         index=rets.index).shift(1)
    pf_bad = pd.Series(particle_filter_2state(rets.values, Np=200, p_stay=0.995), index=rets.index).shift(1)
    feat_df = pd.DataFrame({
        "lag1": lag1, "lag2": lag2, "lag5": lag5,
        "roll_mean5": roll_mean5, "roll_mean21": roll_mean21,
        "roll_vol20": roll_vol20, "ewma_vol": ewma_vol,
        "dd_prev": dd_prev, "from_peak_lag": from_peak_lag,
        "bocpd_cp": bocpd_cp, "pf_bad": pf_bad,
    }).dropna()
    X_all = feat_df.values
    y_all = y.loc[feat_df.index].values

# Fit models on full data
scaler_full = StandardScaler().fit(X_all)
X_scaled = scaler_full.transform(X_all)

logit = LogisticRegression(max_iter=500, class_weight="balanced")
logit.fit(X_scaled, y_all)
coef_series = pd.Series(logit.coef_[0], index=feat_df.columns).sort_values(key=np.abs, ascending=False)

rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_all, y_all)
rf_imp = pd.Series(rf.feature_importances_, index=feat_df.columns).sort_values(ascending=False)

gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
gbm.fit(X_all, y_all)
gbm_imp = pd.Series(gbm.feature_importances_, index=feat_df.columns).sort_values(ascending=False)

# Plot: one chart per figure
fig = plt.figure(figsize=(7,6))
coef_series.plot(kind="barh")
plt.title("Logistic Regression Coefficients (standardized)")
plt.gca().invert_yaxis()
plt.tight_layout(); plt.show()

fig = plt.figure(figsize=(7,6))
rf_imp.plot(kind="barh")
plt.title("Random Forest Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout(); plt.show()

fig = plt.figure(figsize=(7,6))
gbm_imp.plot(kind="barh")
plt.title("GBM Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout(); plt.show()

# Display tables
coef_tbl = coef_series.round(4).to_frame("Logit_Coef")
rf_tbl = rf_imp.round(4).to_frame("RF_Importance")
gbm_tbl = gbm_imp.round(4).to_frame("GBM_Importance")

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Standardized Logistic Coefficients", coef_tbl)
caas_jupyter_tools.display_dataframe_to_user("Random Forest Importances", rf_tbl)
caas_jupyter_tools.display_dataframe_to_user("GBM Importances", gbm_tbl)

# Save a lightweight script that reproduces the pipeline and importance extraction without heavy CV
script_path = "/mnt/data/drawdown_classifier_rich_features_light.py"
with open(script_path, "w") as f:
    f.write('# Lightweight drawdown classifier with rich features (no heavy CV)\\n')
    f.write('import numpy as np, pandas as pd\\n')
    f.write('from sklearn.linear_model import LogisticRegression\\n')
    f.write('from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\\n')
    f.write('from sklearn.preprocessing import StandardScaler\\n')
    f.write('\\n# (Insert the same functions as above or import from your module as needed)\\n')
    f.write('# This placeholder keeps the notebook snappy; use the other saved script for full version.\\n')
print(f"Saved lightweight script to {script_path}")




importances = {}

# Logistic regression coefficients (standardized)
scaler = StandardScaler().fit(X_all)
X_scaled = scaler.transform(X_all)
logit = LogisticRegression(max_iter=500, class_weight="balanced")
logit.fit(X_scaled, y_all)
coefs = pd.Series(logit.coef_[0], index=feat_df.columns)
coefs = coefs.sort_values(key=abs, ascending=False)
importances["Logistic"] = coefs

# Random Forest feature importances
rf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42)
rf.fit(X_all, y_all)
rf_imp = pd.Series(rf.feature_importances_, index=feat_df.columns).sort_values(ascending=False)
importances["RandomForest"] = rf_imp

# Gradient Boosting feature importances
gbm = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42)
gbm.fit(X_all, y_all)
gbm_imp = pd.Series(gbm.feature_importances_, index=feat_df.columns).sort_values(ascending=False)
importances["GBM"] = gbm_imp

# Plotting feature importances
fig, axes = plt.subplots(1, 3, figsize=(16,5), sharey=True)
for ax, (model, imp) in zip(axes, importances.items()):
    imp.plot(kind="barh", ax=ax)
    ax.set_title(f"{model} — Feature Importance/Coef")
    ax.invert_yaxis()
plt.tight_layout()
plt.show()

importances["Logistic"].round(3), importances["RandomForest"].round(3), importances["GBM"].round(3)






#
#
#

# Build full feature set fresh (with stable BOCPD-knownσ), then evaluate.

# Basic features
lag1 = rets.shift(1); lag2 = rets.shift(2); lag5 = rets.shift(5)
roll_mean5 = rets.rolling(5).mean().shift(1)
roll_mean21 = rets.rolling(21).mean().shift(1)
roll_vol20 = rets.rolling(20).std().shift(1)

lam = 0.94
ewvar = np.zeros(len(rets)); prev = 0.0
for i, ri in enumerate(rets.fillna(0.0).values):
    prev = lam*prev + (1-lam)*(ri**2)
    ewvar[i] = prev
ewma_vol = pd.Series(np.sqrt(ewvar)*np.sqrt(dpy), index=rets.index).shift(1)

peak = pd.Series(np.maximum.accumulate(np.r_[1.0, W.values])[1:], index=rets.index)
from_peak = W/peak - 1.0
from_peak_lag = from_peak.shift(1)
dd_prev = DD.shift(1)

# Model-based features (lagged)
sigma2_known = rets.var(ddof=1)
cp_prob_stable = pd.Series(bocpd_knownvar(rets.values, sigma2=sigma2_known,
                                          hazard_lambda=200.0, R_max=250, mu0=0.0, kappa0=10.0),
                           index=rets.index).shift(1)
pf_prob = pd.Series(particle_filter_2state(rets.values, Np=300, p_stay=0.995),
                    index=rets.index).shift(1)

feat_df = pd.DataFrame({
    "lag1": lag1, "lag2": lag2, "lag5": lag5,
    "roll_mean5": roll_mean5, "roll_mean21": roll_mean21,
    "roll_vol20": roll_vol20, "ewma_vol": ewma_vol,
    "dd_prev": dd_prev, "from_peak_lag": from_peak_lag,
    "bocpd_cp": cp_prob_stable, "pf_bad": pf_prob,
}).dropna()

X_all = feat_df.values
y_all = y.loc[feat_df.index].values

# Evaluate
results = {}
for name, clf in models.items():
    y_prob, y_pred, report, cm, auc = eval_model(clf, X_all, y_all, n_splits=6)
    results[name] = {"prob": y_prob, "pred": y_pred, "report": report, "cm": cm, "auc": auc}

# ROC
fig, ax = plt.subplots(figsize=(6,6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_all, res["prob"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],'--')
ax.set_title("ROC — Drawdown (>2.5%) with Rich Features (BOCPD-knownσ/PF/EWMA)")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
plt.tight_layout(); plt.show()

# Prob timeline
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(feat_df.index, y_all, label="Actual DD>2.5%", alpha=0.7)
for name, res in results.items():
    ax.plot(feat_df.index, res["prob"], label=f"{name} prob", alpha=0.8)
ax.set_title("Predicted Probabilities vs Actual State — Rich Features (BOCPD-knownσ/PF/EWMA)")
ax.set_ylabel("Probability / State"); ax.set_xlabel("Date"); ax.legend()
plt.tight_layout(); plt.show()

# Summary table
summary = []
for name, res in results.items():
    rep = res["report"]
    row = {
        "Model": name,
        "Accuracy": rep["accuracy"],
        "AUC": res["auc"],
        "DD Precision": rep["1"]["precision"],
        "DD Recall": rep["1"]["recall"],
        "DD F1": rep["1"]["f1-score"],
        "NoDD Precision": rep["0"]["precision"],
        "NoDD Recall": rep["0"]["recall"],
        "NoDD F1": rep["0"]["f1-score"],
        "TP": int(res["cm"][1,1]), "FP": int(res["cm"][0,1]),
        "FN": int(res["cm"][1,0]), "TN": int(res["cm"][0,0]),
    }
    summary.append(row)
summary_df = pd.DataFrame(summary).set_index("Model").round(4)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Classifier comparison — Rich features (BOCPD-knownσ/PF/EWMA)", summary_df)



#
#
#

# Redefine eval_model with n_splits parameter (expanding-window CV)

from sklearn.base import clone

def expanding_splits(n, n_splits=6, min_train_frac=0.35):
    start = int(n * min_train_frac)
    block = (n - start) // n_splits
    splits = []
    t0 = start
    for i in range(n_splits):
        t1 = t0 + block if i < n_splits - 1 else n
        train_idx = np.arange(0, t0)
        test_idx = np.arange(t0, t1)
        splits.append((train_idx, test_idx))
        t0 = t1
    return splits

def eval_model(clf, X, y, n_splits=6):
    splits = expanding_splits(len(y), n_splits=n_splits, min_train_frac=0.35)
    y_prob = np.zeros(len(y)); y_pred = np.zeros(len(y))
    for train_idx, test_idx in splits:
        # ensure both classes in train; if not, push boundary forward
        unique = np.unique(y[train_idx])
        t_start = test_idx[0]; t_end = test_idx[-1] + 1
        while len(unique) < 2 and t_start < t_end - 10:
            t_start += 10
            train_idx = np.arange(0, t_start)
            unique = np.unique(y[train_idx])
            test_idx = np.arange(t_start, t_end)
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test  = scaler.transform(X[test_idx])
        model = clone(clf)
        model.fit(X_train, y[train_idx])
        prob = model.predict_proba(X_test)[:,1]
        y_prob[test_idx] = prob
        y_pred[test_idx] = (prob > 0.5).astype(int)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    return y_prob, y_pred, report, cm, auc

# Re-run models
results = {}
for name, clf in models.items():
    y_prob, y_pred, report, cm, auc = eval_model(clf, X, y, n_splits=6)
    results[name] = {"prob": y_prob, "pred": y_pred, "report": report, "cm": cm, "auc": auc}

# ROC curves
fig, ax = plt.subplots(figsize=(6,6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y, res["prob"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],'--')
ax.set_title("ROC Curves — Drawdown (>2.5%) Detection")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
plt.tight_layout(); plt.show()

# Probability timeline
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df.index, y, label="Actual DD>2.5%", alpha=0.7)
for name, res in results.items():
    ax.plot(df.index, res["prob"], label=f"{name} prob", alpha=0.8)
ax.set_title("Predicted Probabilities vs Actual Drawdown State")
ax.set_ylabel("Probability / State"); ax.set_xlabel("Date"); ax.legend()
plt.tight_layout(); plt.show()

# Summary table
summary = []
for name, res in results.items():
    rep = res["report"]
    row = {
        "Model": name,
        "Accuracy": rep["accuracy"],
        "AUC": res["auc"],
        "DD Precision": rep["1"]["precision"],
        "DD Recall": rep["1"]["recall"],
        "DD F1": rep["1"]["f1-score"],
        "NoDD Precision": rep["0"]["precision"],
        "NoDD Recall": rep["0"]["recall"],
        "NoDD F1": rep["0"]["f1-score"],
        "TP": int(res["cm"][1,1]), "FP": int(res["cm"][0,1]),
        "FN": int(res["cm"][1,0]), "TN": int(res["cm"][0,0]),
    }
    summary.append(row)
summary_df = pd.DataFrame(summary).set_index("Model").round(4)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Classifier comparison — Drawdown > 2.5% (20y, σ=5%)", summary_df)


#
#
#

# Let's re-run the model vs stop-loss vs buy&hold comparison.
# We'll simulate a 20-year AR(1)-Student-t return series, then compare:
# - Buy & Hold
# - Simple Stop-loss (DD > 0.5σ_ann = 2.5%)
# - Logistic Regression model (prob>0.5 exit)
# - Random Forest model (prob>0.5 exit)

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def simulate_ar1_t_daily(N, mu_ann=0.10, sigma_ann=0.05, rho=0.25, nu=4, dpy=252, seed=1):
    rng = np.random.default_rng(seed)
    mu_d = mu_ann / dpy
    sigma_d = sigma_ann / np.sqrt(dpy)
    Z = rng.standard_t(df=nu, size=N) / np.sqrt(nu/(nu-2.0))
    eps = np.sqrt(1.0 - rho**2) * sigma_d * Z
    r = np.empty(N); r[0] = mu_d + eps[0]
    for t in range(1, N):
        r[t] = mu_d + rho*(r[t-1]-mu_d) + eps[t]
    return pd.Series(r, index=pd.bdate_range("2000-01-03", periods=N), name="r")

def wealth_drawdown(returns: pd.Series):
    W = (1.0 + returns).cumprod()
    P = np.r_[1.0, W.values]; M = np.maximum.accumulate(P)
    DD = pd.Series(1.0 - (P[1:]/M[1:]), index=returns.index, name="DD")
    return W, DD

def strategy_perf(returns: pd.Series, signal: pd.Series, tc=0.0005):
    sig = signal.astype(int).reindex(returns.index).fillna(0).values
    r = returns.values
    strat_r = r * sig
    switches = np.abs(np.diff(sig, prepend=sig[0]))
    strat_r = strat_r - tc * switches
    W = np.cumprod(1.0 + strat_r)
    DD = 1.0 - W/np.maximum.accumulate(W)
    sharpe = (np.mean(strat_r)/np.std(strat_r))*np.sqrt(252) if np.std(strat_r)>0 else np.nan
    mdd = float(np.max(DD))
    return sharpe, mdd

# --- Simulate ---
years, dpy = 20, 252
rets = simulate_ar1_t_daily(N=years*dpy, mu_ann=0.10, sigma_ann=0.05, rho=0.25, nu=4, dpy=dpy, seed=123)
W, DD = wealth_drawdown(rets)

# --- Stop-loss (K=0.5σ_ann = 2.5%) ---
thr = 0.025
sig_stop = (DD <= thr).astype(int)

# --- Features ---
lag1 = rets.shift(1); lag2 = rets.shift(2)
roll_mean5 = rets.rolling(5).mean().shift(1)
roll_vol20 = rets.rolling(20).std().shift(1)
dd_prev = DD.shift(1)
W_ = (1.0 + rets).cumprod()
peak = pd.Series(np.maximum.accumulate(np.r_[1.0, W_.values])[1:], index=rets.index)
from_peak = W_/peak - 1.0
from_peak_lag = from_peak.shift(1)

feat = pd.DataFrame({
    "lag1": lag1, "lag2": lag2,
    "roll_mean5": roll_mean5, "roll_vol20": roll_vol20,
    "dd_prev": dd_prev, "from_peak_lag": from_peak_lag
}).dropna()

y_all = (DD > thr).astype(int).loc[feat.index].values
X = feat.values

# --- Train logistic and RF in-sample (for speed) ---
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

logit = LogisticRegression(max_iter=500, class_weight="balanced").fit(X_scaled, y_all)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42).fit(X, y_all)

prob_logit = logit.predict_proba(X_scaled)[:,1]
prob_rf = rf.predict_proba(X)[:,1]

sig_logit = pd.Series(1 - (prob_logit>0.5).astype(int), index=feat.index).reindex(rets.index).fillna(1)
sig_rf = pd.Series(1 - (prob_rf>0.5).astype(int), index=feat.index).reindex(rets.index).fillna(1)

# --- Evaluate ---
tc = 0.0005
sr_bh, dd_bh = strategy_perf(rets, pd.Series(1, index=rets.index), tc=tc)
sr_stop, dd_stop = strategy_perf(rets, sig_stop, tc=tc)
sr_log, dd_log = strategy_perf(rets, sig_logit, tc=tc)
sr_rf, dd_rf = strategy_perf(rets, sig_rf, tc=tc)

summary = pd.DataFrame({
    "Sharpe":[sr_bh, sr_stop, sr_log, sr_rf],
    "MaxDrawdown":[dd_bh, dd_stop, dd_log, dd_rf]
}, index=["Buy&Hold","StopLoss_K=0.5σ","Logistic (Prob>0.5)","RandomForest (Prob>0.5)"]).round(4)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Performance Comparison: Buy&Hold vs StopLoss vs Logistic vs RF", summary)



#
#
#

# Use a single train/test split to keep things fast and purely out-of-sample for the test period.
# Train on first 60% of the 20y span; evaluate on the last 40%.

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def simulate_ar1_t_daily(N, mu_ann=0.10, sigma_ann=0.05, rho=0.25, nu=4, dpy=252, seed=12345):
    rng = np.random.default_rng(seed)
    mu_d = mu_ann / dpy
    sigma_d = sigma_ann / np.sqrt(dpy)
    Z = rng.standard_t(df=nu, size=N) / np.sqrt(nu/(nu-2.0))
    eps = np.sqrt(1.0 - rho**2) * sigma_d * Z
    r = np.empty(N); r[0] = mu_d + eps[0]
    for t in range(1, N):
        r[t] = mu_d + rho*(r[t-1]-mu_d) + eps[t]
    return pd.Series(r, index=pd.bdate_range("2000-01-03", periods=N), name="r")

def wealth_drawdown(returns: pd.Series):
    W = (1.0 + returns).cumprod()
    P = np.r_[1.0, W.values]; M = np.maximum.accumulate(P)
    DD = pd.Series(1.0 - (P[1:]/M[1:]), index=returns.index, name="DD")
    return W, DD

def make_features(rets: pd.Series, DD: pd.Series):
    lag1 = rets.shift(1); lag2 = rets.shift(2)
    roll_mean5 = rets.rolling(5).mean().shift(1)
    roll_vol20 = rets.rolling(20).std().shift(1)
    dd_prev = DD.shift(1)
    W_ = (1.0 + rets).cumprod()
    peak = pd.Series(np.maximum.accumulate(np.r_[1.0, W_.values])[1:], index=rets.index)
    from_peak = W_/peak - 1.0
    from_peak_lag = from_peak.shift(1)
    feat = pd.DataFrame({
        "lag1": lag1, "lag2": lag2,
        "roll_mean5": roll_mean5, "roll_vol20": roll_vol20,
        "dd_prev": dd_prev, "from_peak_lag": from_peak_lag
    }).dropna()
    return feat

def strategy_perf(returns: pd.Series, signal: pd.Series, tc=0.0005):
    sig = signal.astype(int).reindex(returns.index).fillna(0).values
    r = returns.values
    strat_r = r * sig
    switches = np.abs(np.diff(sig, prepend=sig[0]))
    strat_r = strat_r - tc * switches
    W = np.cumprod(1.0 + strat_r)
    DD = 1.0 - W/np.maximum.accumulate(W)
    sharpe = (np.mean(strat_r)/np.std(strat_r))*np.sqrt(252) if np.std(strat_r)>0 else np.nan
    mdd = float(np.max(DD))
    n_trades = int(switches.sum())
    invest_frac = float(sig.mean())
    return sharpe, mdd, n_trades, invest_frac

# Simulate 20y
years, dpy = 20, 252
rets = simulate_ar1_t_daily(N=years*dpy, mu_ann=0.10, sigma_ann=0.05, rho=0.25, nu=4, dpy=dpy, seed=20250824)
W, DD = wealth_drawdown(rets)
feat = make_features(rets, DD)
thr = 0.025

# Train/test split (60%/40%)
n = len(feat)
split = int(0.6*n)
train_idx = feat.index[:split]
test_idx  = feat.index[split:]

X_train, X_test = feat.loc[train_idx].values, feat.loc[test_idx].values
y_train, y_test = (DD>thr).astype(int).loc[train_idx].values, (DD>thr).astype(int).loc[test_idx].values

# Logistic
scaler = StandardScaler().fit(X_train)
logit = LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear")
logit.fit(scaler.transform(X_train), y_train)
prob_logit_test = logit.predict_proba(scaler.transform(X_test))[:,1]
sig_logit = pd.Series(1 - (prob_logit_test > 0.5).astype(int), index=test_idx)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
prob_rf_test = rf.predict_proba(X_test)[:,1]
sig_rf = pd.Series(1 - (prob_rf_test > 0.5).astype(int), index=test_idx)

# Stop-loss and Buy&Hold signals on test period
sig_bh   = pd.Series(1, index=test_idx)
sig_stop = (DD <= thr).astype(int).loc[test_idx]

# Evaluate on test period only
tc = 0.0005
rows = []
for name, sig in [
    ("Buy&Hold (test)", sig_bh),
    ("StopLoss_K=0.5σ (test)", sig_stop),
    ("Logistic Prob>0.5 (test)", sig_logit),
    ("RandomForest Prob>0.5 (test)", sig_rf),
]:
    sr, mdd, ntr, invest_frac = strategy_perf(rets.loc[test_idx], sig, tc=tc)
    rows.append({"Strategy": name, "Sharpe": sr, "MaxDrawdown": mdd,
                 "#Trades": ntr, "%Invested": invest_frac})

summary = pd.DataFrame(rows).set_index("Strategy").round(4)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Model vs Stop-loss — OOS on final 40% (Sharpe/MaxDD)", summary)


#
#
#

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- Simulation ---
def simulate_ar1_t_daily(N, mu_ann=0.10, sigma_ann=0.05, rho=0.25, nu=4, dpy=252, seed=12345):
    rng = np.random.default_rng(seed)
    mu_d = mu_ann / dpy
    sigma_d = sigma_ann / np.sqrt(dpy)
    Z = rng.standard_t(df=nu, size=N) / np.sqrt(nu/(nu-2.0))
    eps = np.sqrt(1.0 - rho**2) * sigma_d * Z
    r = np.empty(N); r[0] = mu_d + eps[0]
    for t in range(1, N):
        r[t] = mu_d + rho*(r[t-1]-mu_d) + eps[t]
    return pd.Series(r, index=pd.bdate_range("2000-01-03", periods=N), name="r")

def wealth_drawdown(returns: pd.Series):
    W = (1.0 + returns).cumprod()
    P = np.r_[1.0, W.values]; M = np.maximum.accumulate(P)
    DD = pd.Series(1.0 - (P[1:]/M[1:]), index=returns.index, name="DD")
    return W, DD

def make_features(rets: pd.Series, DD: pd.Series):
    lag1 = rets.shift(1); lag2 = rets.shift(2)
    roll_mean5 = rets.rolling(5).mean().shift(1)
    roll_vol20 = rets.rolling(20).std().shift(1)
    dd_prev = DD.shift(1)
    W_ = (1.0 + rets).cumprod()
    peak = pd.Series(np.maximum.accumulate(np.r_[1.0, W_.values])[1:], index=rets.index)
    from_peak = W_/peak - 1.0
    from_peak_lag = from_peak.shift(1)
    feat = pd.DataFrame({
        "lag1": lag1, "lag2": lag2,
        "roll_mean5": roll_mean5, "roll_vol20": roll_vol20,
        "dd_prev": dd_prev, "from_peak_lag": from_peak_lag
    }).dropna()
    return feat

def strategy_perf(returns: pd.Series, signal: pd.Series, tc=0.0005):
    sig = signal.astype(int).reindex(returns.index).fillna(0).values
    r = returns.values[:len(sig)]
    strat_r = r * sig
    switches = np.abs(np.diff(sig, prepend=sig[0]))
    strat_r = strat_r - tc * switches
    W = np.cumprod(1.0 + strat_r)
    DD = 1.0 - W/np.maximum.accumulate(W)
    sharpe = (np.mean(strat_r)/np.std(strat_r))*np.sqrt(252) if np.std(strat_r)>0 else np.nan
    mdd = float(np.max(DD))
    return sharpe, mdd

# --- Simulation (20y) ---
years, dpy = 20, 252
rets = simulate_ar1_t_daily(N=years*dpy, mu_ann=0.10, sigma_ann=0.05, rho=0.25, nu=4, dpy=dpy, seed=20250824)
W, DD = wealth_drawdown(rets)
feat = make_features(rets, DD)
thr = 0.025

# Train/test split (60% train, 40% test)
n = len(feat)
split = int(0.6*n)
train_idx, test_idx = feat.index[:split], feat.index[split:]
X_train, X_test = feat.loc[train_idx].values, feat.loc[test_idx].values
y_train, y_test = (DD>thr).astype(int).loc[train_idx].values, (DD>thr).astype(int).loc[test_idx].values

# Train models
scaler = StandardScaler().fit(X_train)
logit = LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear")
logit.fit(scaler.transform(X_train), y_train)
rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

prob_logit = logit.predict_proba(scaler.transform(X_test))[:,1]
prob_rf    = rf.predict_proba(X_test)[:,1]

# Sweep thresholds
thresholds = np.linspace(0.1,0.9,17)
rows = []
for th in thresholds:
    sig_logit = pd.Series(1 - (prob_logit > th).astype(int), index=test_idx)
    sig_rf    = pd.Series(1 - (prob_rf    > th).astype(int), index=test_idx)
    sr_log, dd_log = strategy_perf(rets.loc[test_idx], sig_logit)
    sr_rf,  dd_rf  = strategy_perf(rets.loc[test_idx], sig_rf)
    rows.append({"Threshold": th, "Sharpe_Logit": sr_log, "MaxDD_Logit": dd_log,
                 "Sharpe_RF": sr_rf, "MaxDD_RF": dd_rf})

frontier = pd.DataFrame(rows)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,2, figsize=(12,5))
axes[0].plot(frontier["Threshold"], frontier["Sharpe_Logit"], label="Logit")
axes[0].plot(frontier["Threshold"], frontier["Sharpe_RF"], label="RF")
axes[0].set_title("Sharpe vs Threshold"); axes[0].set_xlabel("Prob threshold"); axes[0].set_ylabel("Sharpe"); axes[0].legend()

axes[1].plot(frontier["Threshold"], frontier["MaxDD_Logit"], label="Logit")
axes[1].plot(frontier["Threshold"], frontier["MaxDD_RF"], label="RF")
axes[1].set_title("Max Drawdown vs Threshold"); axes[1].set_xlabel("Prob threshold"); axes[1].set_ylabel("Max DD"); axes[1].legend()

plt.tight_layout()
plt.show()

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Sharpe & MaxDD across probability thresholds", frontier.round(4))



#
#
#



#
# Retry with much smaller training load (5 years of data, 5 epochs, batch_episodes=1)

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim

def simulate_ar1_t_daily(N, mu_ann=0.10, sigma_ann=0.05, rho=0.25, nu=4, dpy=252, seed=21):
    rng = np.random.default_rng(seed)
    mu_d = mu_ann / dpy
    sigma_d = sigma_ann / np.sqrt(dpy)
    Z = rng.standard_t(df=nu, size=N) / np.sqrt(nu/(nu-2.0))
    eps = np.sqrt(1.0 - rho**2) * sigma_d * Z
    r = np.empty(N); r[0] = mu_d + eps[0]
    for t in range(1, N):
        r[t] = mu_d + rho*(r[t-1]-mu_d) + eps[t]
    return pd.Series(r, index=pd.bdate_range("2000-01-03", periods=N), name="r")

def make_features(rets, ewma_lambda=0.94, dpy=252):
    W = (1+rets).cumprod()
    P = np.r_[1.0, W.values]; M = np.maximum.accumulate(P)
    DD = pd.Series(1.0 - (P[1:]/M[1:]), index=rets.index, name="DD")
    lag1, lag2 = rets.shift(1), rets.shift(2)
    rm5, rm21 = rets.rolling(5).mean().shift(1), rets.rolling(21).mean().shift(1)
    rv20 = rets.rolling(20).std().shift(1)
    ewvar, prev = np.zeros(len(rets)), 0.0
    for i, ri in enumerate(rets.fillna(0.0).values):
        prev = ewma_lambda*prev + (1-ewma_lambda)*ri*ri
        ewvar[i] = prev
    ewma_vol = pd.Series(np.sqrt(ewvar)*np.sqrt(dpy), index=rets.index).shift(1)
    peak = pd.Series(np.maximum.accumulate(np.r_[1.0, W.values])[1:], index=rets.index)
    from_peak = (W/peak - 1.0).shift(1)
    feat = pd.concat([lag1, lag2, rm5, rm21, rv20, ewma_vol, DD.shift(1), from_peak], axis=1)
    feat.columns = ["lag1","lag2","rm5","rm21","rv20","ewma_vol","dd_prev","from_peak_lag"]
    feat = feat.dropna()
    return feat, DD.loc[feat.index]

class StopReentryEnv:
    def __init__(self, rets, feats, dd, tc=5e-4, reward="drawdown", lam=10.0, eta=50.0, tau=0.025):
        X = (feats - feats.mean())/feats.std(ddof=0)
        self.X = X.values.astype(np.float32)
        self.rets = rets.loc[feats.index].values.astype(np.float32)
        self.dd = dd.loc[feats.index].values.astype(np.float32)
        self.tc, self.reward_type = float(tc), reward
        self.lam, self.eta, self.tau = float(lam), float(eta), float(tau)
        self.T = len(self.rets)
        self.reset()
    def reset(self):
        self.t = 0; self.pos = 1; self.wealth = 1.0
        return np.concatenate([self.X[self.t], [self.pos]]).astype(np.float32)
    def step(self, a):
        a = int(a); prev = self.pos; self.pos = a
        r_port = self.pos * self.rets[self.t]
        switch_cost = self.tc * (self.pos != prev)
        reward = r_port - switch_cost
        if self.reward_type == "meanvar":
            reward -= self.lam * (r_port**2)
        elif self.reward_type == "drawdown":
            reward -= self.eta * max(0.0, float(self.dd[self.t]) - self.tau)
        self.wealth *= (1 + r_port - switch_cost)
        self.t += 1
        done = (self.t >= self.T-1)
        state = np.concatenate([self.X[self.t], [self.pos]]).astype(np.float32)
        return state, float(reward), done, {}
    @property
    def state_dim(self): return self.X.shape[1] + 1

class Policy(nn.Module):
    def __init__(self, state_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
    def forward(self, s): return self.net(s).squeeze(-1)

def train(env, epochs=5, gamma=0.99, lr=5e-4, batch_episodes=1, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    pol = Policy(env.state_dim); opt = optim.Adam(pol.parameters(), lr=lr)
    for ep in range(epochs):
        all_logps, all_returns = [], []
        s = torch.tensor(env.reset())
        done = False; logps = []; rewards = []
        while not done:
            p = pol(s)
            m = torch.distributions.Bernoulli(p.clamp(1e-5, 1-1e-5))
            a = m.sample()
            logps.append(m.log_prob(a))
            ns, r, done, _ = env.step(int(a.item()))
            s = torch.tensor(ns)
            rewards.append(r)
        G, ret = [], 0.0
        for r in reversed(rewards):
            ret = r + gamma*ret
            G.append(ret)
        G.reverse()
        R = torch.tensor(G, dtype=torch.float32)
        R = (R - R.mean())/(R.std()+1e-8)
        loss = -(torch.stack(logps)*R).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return pol

def run_policy(env, pol):
    s = torch.tensor(env.reset())
    pos, strat_r = [], []
    done=False
    while not done:
        with torch.no_grad():
            p = pol(s).item()
        a = 1 if p>0.5 else 0
        ns, r, done, _ = env.step(a)
        switch_cost = env.tc * (a != int(s[-1].item()))
        strat_r.append(env.rets[env.t-1]*a - switch_cost)
        pos.append(a)
        s = torch.tensor(ns)
    strat_r = np.array(strat_r)
    wealth = np.cumprod(1+strat_r)
    dd = 1 - wealth/np.maximum.accumulate(wealth)
    sharpe = (np.mean(strat_r)/np.std(strat_r))*np.sqrt(252) if np.std(strat_r)>0 else np.nan
    return sharpe, float(np.max(dd))

# --- 5-year run ---
N = 252*5
rets = simulate_ar1_t_daily(N, mu_ann=0.10, sigma_ann=0.05, rho=0.25, nu=4, seed=42)
feats, DD = make_features(rets)
idx = feats.index
def perf(sig):
    r = rets.loc[idx].values
    sig = sig.astype(int).reindex(idx).fillna(0).values
    switches = np.abs(np.diff(sig, prepend=sig[0]))
    tc = 5e-4
    strat = r*sig - tc*switches
    wealth = np.cumprod(1+strat)
    dd = 1 - wealth/np.maximum.accumulate(wealth)
    sharpe = (np.mean(strat)/np.std(strat))*np.sqrt(252) if np.std(strat)>0 else np.nan
    return sharpe, float(np.max(dd))

sr_bh, mdd_bh = perf(pd.Series(1, index=idx))
sr_stop, mdd_stop = perf((DD <= 0.025).astype(int))

env = StopReentryEnv(rets, feats, DD, tc=5e-4, reward="drawdown", lam=10.0, eta=50.0, tau=0.025)
pol = train(env, epochs=5, gamma=0.99, lr=5e-4, batch_episodes=1, seed=0)
sr_rl, mdd_rl = run_policy(env, pol)

res = pd.DataFrame({
    "Sharpe":[sr_bh, sr_stop, sr_rl],
    "MaxDrawdown":[mdd_bh, mdd_stop, mdd_rl]
}, index=["Buy&Hold","StopLoss_K=0.5σ","RL (drawdown-aware)"]).round(4)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("RL vs Stop-loss — Sharpe & MaxDD (5y, quick run)", res)



