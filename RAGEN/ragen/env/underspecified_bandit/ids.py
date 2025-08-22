from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from scipy.special import digamma, gammaln
from .utils import sample

def _entropy_discrete(p: np.ndarray) -> float:
    """Shannon entropy (nats) of a discrete distribution vector p (sum ~ 1)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _prob_optimal_from_draws(draws: np.ndarray) -> np.ndarray:
    """
    draws: (K, n) samples of per-arm means (one column = one world)
    returns P(A* = k) vector of length K via argmax counts.
    """
    best = np.argmax(draws, axis=0)
    K = draws.shape[0]
    return np.bincount(best, minlength=K) / draws.shape[1]


def _delta_mc_from_draws(draws: np.ndarray) -> np.ndarray:
    """
    draws: (K, n) samples of per-arm means.
    returns Delta_i = E[max_k mu_k - mu_i] for each arm i.
    """
    max_per_world = np.max(draws, axis=0)  # (n,)
    return np.mean(max_per_world[None, :] - draws, axis=1)  # (K,)


def _choose_ids_action(Delta: np.ndarray, g: np.ndarray, rng: np.random.Generator) -> int:
    """
    Solve the IDS one-step optimization over distributions on arms.
    Uses the fact that an optimal solution has support on ≤ 2 arms.
    Returns an arm index sampled from the optimal distribution.

    We consider all single-arm solutions and all pairs (i,j) with
    the optimal mixture weight w* (clipped to [0,1]); then sample.
    """
    K = len(Delta)
    Delta = np.asarray(Delta, dtype=float)
    g = np.asarray(g, dtype=float)

    # Helper: cost for mixture weight w for pair (i,j)
    def _cost_pair(w, i, j):
        w = float(np.clip(w, 0.0, 1.0))
        denom = w * g[i] + (1.0 - w) * g[j]
        if denom <= 0.0:
            # If zero info but also zero regret, the ratio is 0; otherwise inf
            num = (w * Delta[i] + (1.0 - w) * Delta[j]) ** 2
            return 0.0 if num == 0.0 else np.inf, w
        num = (w * Delta[i] + (1.0 - w) * Delta[j]) ** 2
        return num / denom, w

    best_cost = np.inf
    best_pair: Tuple[int, Optional[int]] = (0, None)
    best_w = 1.0

    # Single-arm candidates
    for i in range(K):
        if g[i] > 0.0:
            cost = (Delta[i] ** 2) / g[i]
        else:
            cost = 0.0 if Delta[i] == 0.0 else np.inf
        if cost < best_cost:
            best_cost, best_pair, best_w = cost, (i, None), 1.0

    # Two-arm mixtures
    for i in range(K):
        for j in range(i + 1, K):
            Di, Dj = Delta[i], Delta[j]
            gi, gj = g[i], g[j]
            a = Di - Dj
            b = Dj
            c = gi - gj
            d = gj

            candidates = []
            # Endpoints
            candidates.append(_cost_pair(0.0, i, j))
            candidates.append(_cost_pair(1.0, i, j))

            # Stationary point (if defined)
            # Derived from derivative of f(w) = (b + a w)^2 / (d + c w)
            # w* = (c*b - 2*a*d) / (a*c)  when a*c != 0
            if a != 0.0 and c != 0.0:
                w0 = (c * b - 2.0 * a * d) / (a * c)
                candidates.append(_cost_pair(w0, i, j))
            elif c == 0.0 and a != 0.0:
                # Equal infos: minimize (b + a w)^2
                w0 = -b / a
                candidates.append(_cost_pair(w0, i, j))
            elif a == 0.0 and c != 0.0:
                # Equal regrets: maximize denominator (pick arm with larger g)
                w0 = 1.0 if gi > gj else 0.0
                candidates.append(_cost_pair(w0, i, j))

            # Pick best candidate for this pair
            for cost, w in candidates:
                if cost < best_cost:
                    best_cost, best_pair, best_w = cost, (i, j), w

    # Sample an arm from the optimal distribution
    if best_pair[1] is None:
        return best_pair[0]
    i, j = best_pair
    return i if rng.random() < best_w else j


# ------------------------- Beta–Bernoulli IDS -------------------------

def _p_opt_beta(alpha: np.ndarray, beta: np.ndarray, n_mc: int, rng: np.random.Generator) -> np.ndarray:
    """P(A* = k) under Beta posteriors via MC."""
    K = len(alpha)
    draws = rng.beta(alpha[:, None], beta[:, None], size=(K, n_mc))
    return _prob_optimal_from_draws(draws)


def _delta_beta(alpha: np.ndarray, beta: np.ndarray, n_mc: int, rng: np.random.Generator) -> np.ndarray:
    """MC-exact Delta_i for Beta posteriors."""
    K = len(alpha)
    draws = rng.beta(alpha[:, None], beta[:, None], size=(K, n_mc))
    return _delta_mc_from_draws(draws)


def _g_beta_Astar(alpha: np.ndarray, beta: np.ndarray, n_mc: int, rng: np.random.Generator) -> np.ndarray:
    """
    g_i = I(A*; Y_i) = H(P(A*)) - E_{Y in {0,1}}[ H(P(A* | posterior after pulling i and observing Y)) ].
    Uses MC for P(A*).
    """
    K = len(alpha)
    H0 = _entropy_discrete(_p_opt_beta(alpha, beta, n_mc, rng))
    means = alpha / (alpha + beta)
    g = np.zeros(K, dtype=float)

    # Precompute eye updates
    for i in range(K):
        p1 = means[i]         # P(Y=1 | history)
        p0 = 1.0 - p1

        # Success update
        H1 = _entropy_discrete(_p_opt_beta(alpha + np.eye(K)[i], beta, n_mc, rng))
        # Failure update
        H0y = _entropy_discrete(_p_opt_beta(alpha, beta + np.eye(K)[i], n_mc, rng))

        g[i] = H0 - (p1 * H1 + p0 * H0y)
        if g[i] < 0.0:  # clamp tiny negative due to MC noise
            g[i] = 0.0
    return g


def ids_beta_bernoulli(
    arm_prior_alpha,
    arm_prior_beta,
    arm_true_p,
    horizon: int,
    *,
    n_delta_mc: int = 2048,
    n_info_mc: int = 512,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Information-Directed Sampling for Beta–Bernoulli.
    Returns total reward over the horizon (like your original).
    """
    rng = rng or np.random.default_rng()
    alpha = np.asarray(arm_prior_alpha, dtype=float).copy()
    beta = np.asarray(arm_prior_beta, dtype=float).copy()
    arm_true_p = np.asarray(arm_true_p, dtype=float)
    K = len(arm_true_p)

    # Validation
    if not (alpha.shape == beta.shape == (K,)):
        raise ValueError("Shapes must match: alpha, beta, and arm_true_p lengths must be equal.")
    if np.any(alpha <= 0.0) or np.any(beta <= 0.0):
        raise ValueError("Beta prior parameters must be > 0.")
    if np.any((arm_true_p < 0.0) | (arm_true_p > 1.0)):
        raise ValueError("True Bernoulli probabilities must be in [0,1].")

    rewards = np.empty(horizon, dtype=float)

    for _t in range(horizon):
        Delta = _delta_beta(alpha, beta, n_delta_mc, rng)
        g = _g_beta_Astar(alpha, beta, n_info_mc, rng)

        arm = _choose_ids_action(Delta, g, rng)

        r = sample({"type": "bernoulli", "p": arm_true_p[arm]}, rng)
        r = float(r)

        alpha[arm] += r
        beta[arm] += 1.0 - r

        rewards[_t] = r

    return float(rewards.sum())


# ------------------------- Gaussian–Gaussian IDS -------------------------

def _delta_gaussian(mu: np.ndarray, var: np.ndarray, n_mc: int, rng: np.random.Generator) -> np.ndarray:
    draws = rng.normal(loc=mu[:, None], scale=np.sqrt(var)[:, None], size=(mu.size, n_mc))
    return _delta_mc_from_draws(draws)


def _g_gaussian_theta(var: np.ndarray, rvar: np.ndarray) -> np.ndarray:
    """θ-IDS information term: I(theta; Y) = 0.5 * log(1 + var / noise_var)."""
    # Guard against zero/negative values
    var = np.maximum(var, 1e-18)
    rvar = np.maximum(rvar, 1e-18)
    return 0.5 * np.log1p(var / rvar)


def ids_gaussian_gaussian(
    arm_prior_mu,
    arm_prior_sigma,
    arm_true_mu,
    arm_true_sigma,
    horizon: int,
    *,
    n_delta_mc: int = 2048,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    IDS with Normal-Normal model, known observation std dev (arm_true_sigma).
    Uses θ-IDS information term (closed-form). Returns total reward.
    """
    rng = rng or np.random.default_rng()

    mu = np.asarray(arm_prior_mu, dtype=float).copy()
    var = (np.asarray(arm_prior_sigma, dtype=float) ** 2).copy()
    true_mu = np.asarray(arm_true_mu, dtype=float)
    rvar = (np.asarray(arm_true_sigma, dtype=float) ** 2).copy()

    K = len(true_mu)
    if not (mu.shape == var.shape == rvar.shape == (K,)):
        raise ValueError("All parameter arrays must have the same length K.")
    if np.any(var < 0.0) or np.any(rvar < 0.0):
        raise ValueError("Variances must be >= 0.")

    rewards = np.empty(horizon, dtype=float)

    for _t in range(horizon):
        Delta = _delta_gaussian(mu, var, n_delta_mc, rng)
        g = _g_gaussian_theta(var, rvar)

        arm = _choose_ids_action(Delta, g, rng)

        r = sample({"type": "gaussian", "mu": true_mu[arm], "sigma": np.sqrt(rvar[arm])}, rng)

        # Posterior update (known noise variance)
        if rvar[arm] == 0.0:
            mu[arm], var[arm] = float(r), 0.0
        else:
            prec_post = 1.0 / var[arm] + 1.0 / rvar[arm]
            var_post = 1.0 / prec_post
            mu_post = var_post * (mu[arm] / var[arm] + r / rvar[arm])
            mu[arm], var[arm] = mu_post, var_post

        rewards[_t] = float(r)

    return float(rewards.sum())


# ------------------------- Poisson–Gamma IDS -------------------------

def _H_gamma_rate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Entropy of Gamma(a, rate=b) in nats:
    H = a - log b + log Γ(a) + (1 - a) ψ(a)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a - np.log(b) + gammaln(a) + (1.0 - a) * digamma(a)


def _delta_gamma(a: np.ndarray, b: np.ndarray, n_mc: int, rng: np.random.Generator) -> np.ndarray:
    """MC-exact Delta_i for Gamma posteriors over Poisson means (λ)."""
    draws = rng.gamma(shape=a[:, None], scale=1.0 / b[:, None], size=(a.size, n_mc))
    return _delta_mc_from_draws(draws)


def _g_poisson_gamma_theta(a: np.ndarray, b: np.ndarray, n_mc: int, rng: np.random.Generator) -> np.ndarray:
    """
    θ-IDS information term for Poisson–Gamma:
    g_i = H(Gamma(a_i, b_i)) - E_{Y ~ Predictive}[ H(Gamma(a_i + Y, b_i + 1)) ].
    Expectation over predictive is approximated via MC:
      λ ~ Gamma(a, 1/b), Y ~ Poisson(λ).
    """
    K = len(a)
    H_prior = _H_gamma_rate(a, b)  # (K,)
    # Sample per-arm independently
    lam = rng.gamma(shape=a[:, None], scale=1.0 / b[:, None], size=(K, n_mc))
    y = rng.poisson(lam)
    H_post = _H_gamma_rate(a[:, None] + y, (b[:, None] + 1.0))
    # Average over MC for each arm
    E_H_post = np.mean(H_post, axis=1)
    g = H_prior - E_H_post
    # Clamp tiny negatives from MC noise
    g[g < 0.0] = 0.0
    return g


def ids_poisson_gamma(
    arm_prior_a,
    arm_prior_b,
    arm_true_lambda,
    horizon: int,
    *,
    n_delta_mc: int = 2048,
    n_info_mc: int = 1024,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    IDS with Poisson rewards and Gamma(a,b) prior on λ (rate parameterization).
    Uses θ-IDS information term computed via entropy-drop MC. Returns total reward.
    Assumes unit exposure per pull (b += 1).
    """
    rng = rng or np.random.default_rng()

    a = np.asarray(arm_prior_a, dtype=float).copy()
    b = np.asarray(arm_prior_b, dtype=float).copy()
    true_lam = np.asarray(arm_true_lambda, dtype=float)
    K = len(true_lam)

    if not (a.shape == b.shape == (K,)):
        raise ValueError("Shapes must match: a, b, and arm_true_lambda lengths must be equal.")
    if np.any(a <= 0.0) or np.any(b <= 0.0):
        raise ValueError("Gamma prior parameters must be > 0.")
    if np.any(true_lam < 0.0):
        raise ValueError("True Poisson rates must be >= 0.")

    rewards = np.empty(horizon, dtype=float)

    for _t in range(horizon):
        Delta = _delta_gamma(a, b, n_delta_mc, rng)
        g = _g_poisson_gamma_theta(a, b, n_info_mc, rng)

        arm = _choose_ids_action(Delta, g, rng)

        r = sample({"type": "poisson", "lambda": true_lam[arm]}, rng)
        a[arm] += r
        b[arm] += 1.0

        rewards[_t] = float(r)

    return float(rewards.sum())