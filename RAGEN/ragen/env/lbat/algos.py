from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm
from scipy.stats import gamma as gamma_dist


def brier_score(pred_prob: float, y: int) -> float:
    p = float(np.clip(pred_prob, 0.0, 1.0))
    return - (p - float(y)) ** 2


def hedge_brier(expert_probs: np.ndarray, y_true: np.ndarray, eta: float | None = None) -> float:
    """
    Hedge over probabilistic experts under Brier loss (negative squared error).
    expert_probs: shape (T, N), rows are expert probability predictions P(y=1)
    y_true: shape (T,), binary labels in {0,1}
    Returns total reward (sum of negative Brier losses) over T.
    """
    T, N = expert_probs.shape
    # loss in [0,1] if p,y in [0,1], so eta ~ sqrt(8 ln N / T) typical; we use sqrt(2 ln N / T)
    if eta is None:
        eta = np.sqrt(2.0 * np.log(N) / max(T, 1))

    w = np.ones(N, dtype=float) / N
    total_reward = 0.0
    for t in range(T):
        probs_t = expert_probs[t]
        y = int(y_true[t])
        p_hat = float(np.dot(w, probs_t))
        total_reward += brier_score(p_hat, y)
        # expert losses and weight update
        losses = (probs_t - y) ** 2  # in [0,1]
        w = w * np.exp(-eta * losses)
        s = w.sum()
        if s <= 0:
            w = np.ones(N, dtype=float) / N
        else:
            w /= s
    return float(total_reward)


def ons_log_loss(returns: np.ndarray, horizon: int, init_w: np.ndarray | None = None, beta: float = 1.0) -> float:
    """
    ONS for online portfolio selection with log wealth objective.
    returns: shape (T, d), per-round gross returns x_t in R_+^d (e.g., price relatives)
    horizon: T
    init_w: initial weights simplex
    beta: regularization parameter for Hessian init
    Returns total log wealth sum_t log(w_t^T x_t).
    """
    T, d = returns.shape
    if init_w is None:
        w = np.ones(d) / d
    else:
        w = init_w.astype(float)
        w = np.maximum(w, 0.0)
        s = w.sum()
        w = (w / s) if s > 0 else np.ones(d) / d

    A = beta * np.eye(d)
    total_log_wealth = 0.0

    for t in range(T):
        x = returns[t]
        denom = max(1e-12, float(np.dot(w, x)))
        total_log_wealth += np.log(denom)

        # gradient of -log(w^T x) wrt w is -x / (w^T x)
        g = - x / denom
        # ONS update: w' = Proj_simplex( w - A^{-1} g ) with A += g g^T
        A += np.outer(g, g)
        try:
            step = np.linalg.solve(A, g)
        except Exception as e:
            print(f"ONS solve failed: {e}")
            step = g
        w = w - step
        # project onto simplex
        w = _project_to_simplex(w)
    return float(total_log_wealth)


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto probability simplex."""
    v = np.asarray(v, dtype=float)
    if v.sum() == 1.0 and np.all(v >= 0):
        return v
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    s = w.sum()
    return w if s == 0 else (w / s)


# ============ MAB UCB baselines ============

def ucb_beta_bernoulli(
    arm_prior_alpha: List[float] | np.ndarray,
    arm_prior_beta: List[float] | np.ndarray,
    arm_true_p: List[float] | np.ndarray,
    horizon: int,
    rng: Optional[np.random.Generator] = None,
    pre_rewards: Optional[np.ndarray] = None,
) -> float:
    rng = rng or np.random.default_rng()
    K = len(arm_true_p)
    alpha = np.asarray(arm_prior_alpha, dtype=float).copy()
    beta = np.asarray(arm_prior_beta, dtype=float).copy()
    p_true = np.asarray(arm_true_p, dtype=float)
    rewards = 0.0
    for t in range(horizon):
        q = 1.0 - 1.0 / (t + 2)
        ucbs = beta_dist.ppf(q, alpha, beta)
        arm = int(rng.choice(np.flatnonzero(ucbs == ucbs.max())))
        if pre_rewards is not None:
            r = float(pre_rewards[t, arm])
        else:
            r = float(rng.binomial(1, p_true[arm]))
        alpha[arm] += r
        beta[arm] += 1.0 - r
        rewards += r
    return float(rewards)


def ucb_gaussian_gaussian(
    arm_prior_mu: List[float] | np.ndarray,
    arm_prior_sigma: List[float] | np.ndarray,
    arm_true_mu: List[float] | np.ndarray,
    arm_true_sigma: List[float] | np.ndarray,
    horizon: int,
    rng: Optional[np.random.Generator] = None,
    pre_rewards: Optional[np.ndarray] = None,
) -> float:
    rng = rng or np.random.default_rng()
    K = len(arm_true_mu)
    mu = np.asarray(arm_prior_mu, dtype=float).copy()
    sigmasq = (np.asarray(arm_prior_sigma, dtype=float) ** 2).copy()
    tmu = np.asarray(arm_true_mu, dtype=float)
    rsigmasq = (np.asarray(arm_true_sigma, dtype=float) ** 2).copy()
    rewards = 0.0
    for t in range(horizon):
        q = 1.0 - 1.0 / (t + 2)
        z = norm.ppf(q)
        ucbs = mu + z * np.sqrt(sigmasq)
        arm = int(rng.choice(np.flatnonzero(ucbs == ucbs.max())))
        if pre_rewards is not None:
            r = float(pre_rewards[t, arm])
        else:
            r = float(rng.normal(tmu[arm], np.sqrt(rsigmasq[arm])))
        # Bayesian update with known observation variance
        prec_post = 1.0 / sigmasq[arm] + 1.0 / rsigmasq[arm]
        sigmasq_post = 1.0 / prec_post
        mu_post = sigmasq_post * (mu[arm] / sigmasq[arm] + r / rsigmasq[arm])
        mu[arm], sigmasq[arm] = mu_post, sigmasq_post
        rewards += r
    return float(rewards)


def ucb_poisson_gamma(
    arm_prior_a: List[float] | np.ndarray,
    arm_prior_b: List[float] | np.ndarray,
    arm_true_lambda: List[float] | np.ndarray,
    horizon: int,
    rng: Optional[np.random.Generator] = None,
    pre_rewards: Optional[np.ndarray] = None,
) -> float:
    rng = rng or np.random.default_rng()
    K = len(arm_true_lambda)
    a = np.asarray(arm_prior_a, dtype=float).copy()
    b = np.asarray(arm_prior_b, dtype=float).copy()
    lam = np.asarray(arm_true_lambda, dtype=float)
    rewards = 0.0
    for t in range(horizon):
        q = 1.0 - 1.0 / (t + 2)
        ucbs = gamma_dist.ppf(q, a, scale=1.0 / b)
        arm = int(rng.choice(np.flatnonzero(ucbs == ucbs.max())))
        if pre_rewards is not None:
            r = float(pre_rewards[t, arm])
        else:
            r = float(rng.poisson(lam[arm]))
        a[arm] += r
        b[arm] += 1.0
        rewards += r
    return float(rewards)


