import numpy as np
from .utils import sample
from scipy.stats import beta as beta_dist
from scipy.stats import norm
from scipy.stats import gamma as gamma_dist

def ucb_beta_bernoulli(arm_prior_alpha, arm_prior_beta, arm_true_p, horizon):
    rng = np.random.default_rng()
    K = len(arm_true_p)

    alpha = np.asarray(arm_prior_alpha, dtype=float).copy()
    beta = np.asarray(arm_prior_beta, dtype=float).copy()

    actions = np.empty(horizon, dtype=int)
    rewards = np.empty(horizon)

    for t in range(horizon):
        ucbs = beta_dist.ppf(1 - 1.0/(t+1), alpha, beta)
        arm = int(rng.choice(np.flatnonzero(ucbs == ucbs.max())))

        r = sample({"type": "bernoulli", "p": arm_true_p[arm]}, rng)
        r = int(r)

        alpha[arm] += r
        beta[arm] += 1 - r

        actions[t], rewards[t] = arm, r

    return rewards.sum()

def ucb_gaussian_gaussian(arm_prior_mu, arm_prior_sigma, arm_true_mu, arm_true_sigma, horizon):
    rng = np.random.default_rng()
    K = len(arm_true_mu)

    mu = np.asarray(arm_prior_mu, dtype=float).copy()
    sigmasq = np.asarray(arm_prior_sigma, dtype=float) ** 2
    rsigmasq = np.asarray(arm_true_sigma, dtype=float) ** 2

    actions = np.empty(horizon, dtype=int)
    rewards = np.empty(horizon)

    pulls = np.zeros(K)

    for t in range(horizon):
        z = norm.ppf(1 - 1.0/(t+1))
        ucbs = mu + z * np.sqrt(sigmasq)
        arm = int(rng.choice(np.flatnonzero(ucbs == ucbs.max())))

        r = sample(
            {"type": "gaussian", "mu": arm_true_mu[arm], "sigma": arm_true_sigma[arm]},
            rng,
        )

        pulls[arm] += 1

        prec_post = 1.0 / sigmasq[arm] + 1.0 / rsigmasq[arm]
        sigmasq_post = 1.0 / prec_post
        mu_post = sigmasq_post * (mu[arm] / sigmasq[arm] + r / rsigmasq[arm])

        mu[arm], sigmasq[arm] = mu_post, sigmasq_post

        actions[t], rewards[t] = arm, r

    return rewards.sum()

def ucb_poisson_gamma(arm_prior_a, arm_prior_b, arm_true_lambda, horizon):
    rng = np.random.default_rng()
    K = len(arm_true_lambda)

    a = np.asarray(arm_prior_a, dtype=float).copy()
    b = np.asarray(arm_prior_b, dtype=float).copy()

    actions = np.empty(horizon, dtype=int)
    rewards = np.empty(horizon)

    for t in range(horizon):
        ucbs = gamma_dist.ppf(1 - 1.0/(t+1), a, scale=1.0/b)
        arm = int(rng.choice(np.flatnonzero(ucbs == ucbs.max())))

        r = sample({"type": "poisson", "lambda": arm_true_lambda[arm]}, rng)

        a[arm] += r
        b[arm] += 1.0

        actions[t], rewards[t] = arm, r

    return rewards.sum()