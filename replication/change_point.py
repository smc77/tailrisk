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
