from dataclasses import dataclass
from typing import Tuple


@dataclass
class LBATEnvConfig:
    # Shared
    render_mode: str = "text"
    min_horizon: int = 10
    max_horizon: int = 40
    # OOD controls
    # If enabled, with probability ood_probability per reset() we will draw parameters
    # from the OOD ranges below instead of the in-distribution ranges.
    ood_enabled: bool = False
    ood_probability: float = 1.0
    ood_min_horizon: int = 30
    ood_max_horizon: int = 40
    # Probability to rewrite environment prompts via GPT (0 disables rewriting)
    rewrite_probability: float = 0.01

    # MAB (unchanged semantics)
    min_arms: int = 2
    max_arms: int = 6
    possible_bandit_types: Tuple[str, ...] = ("beta-bernoulli", "gaussian-gaussian", "poisson-gamma")
    # beta-bernoulli
    max_alpha: float = 2.0
    max_beta: float = 2.0
    min_alpha: float = 0.25
    min_beta: float = 0.25
    max_p: float = 0.9
    min_p: float = 0.1
    # OOD MAB ranges
    ood_min_arms: int = 8
    ood_max_arms: int = 12
    ood_max_alpha: float = 3.0
    ood_max_beta: float = 3.0
    # skewed towards extremes
    ood_min_alpha: float = 0.1
    ood_min_beta: float = 0.1
    ood_max_p: float = 0.98
    ood_min_p: float = 0.02
    # gaussian-gaussian
    max_mu_prior: float = 1.2
    min_mu_prior: float = 0.2
    max_sigma_prior: float = 2.0
    min_sigma_prior: float = 1.0
    max_mu_true: float = 1.0
    min_mu_true: float = 0.3
    max_sigma_true: float = 0.12
    min_sigma_true: float = 0.03
    # OOD gaussian-gaussian
    # skewed: priors can be slightly negative; true sigma higher
    ood_max_mu_prior: float = 1.4
    ood_min_mu_prior: float = -0.2
    ood_max_sigma_prior: float = 2.8
    ood_min_sigma_prior: float = 0.7
    ood_max_mu_true: float = 1.3
    ood_min_mu_true: float = 0.15
    ood_max_sigma_true: float = 0.35
    ood_min_sigma_true: float = 0.18
    # poisson-gamma
    max_a: float = 3.0
    max_b: float = 3.0
    min_a: float = 0.3
    min_b: float = 0.3
    max_lambda: float = 2.5
    min_lambda: float = 0.4
    # OOD poisson-gamma
    ood_max_a: float = 4.0
    ood_max_b: float = 4.0
    # skewed: allow smaller rates and larger rates
    ood_min_a: float = 0.15
    ood_min_b: float = 0.15
    ood_max_lambda: float = 3.8
    ood_min_lambda: float = 0.15

    # PEA
    pea_min_experts: int = 2
    pea_max_experts: int = 6
    # Latent truth Bernoulli parameter range
    pea_min_truth_p: float = 0.1
    pea_max_truth_p: float = 0.9
    # Expert sharpness parameters kappa; experts predict Beta(kappa*truth_p, kappa*(1-truth_p))
    pea_min_expert_kappa: float = 2.0
    pea_max_expert_kappa: float = 10.0
    # Expert biases in logit space; p' = sigmoid(logit(p) + bias)
    pea_min_expert_bias: float = -1.0
    pea_max_expert_bias: float = 1.0
    # OOD PEA
    pea_ood_min_experts: int = 8
    pea_ood_max_experts: int = 12
    # skewed: push truth closer to extremes
    pea_ood_min_truth_p: float = 0.03
    pea_ood_max_truth_p: float = 0.97
    pea_ood_min_expert_kappa: float = 12.0
    pea_ood_max_expert_kappa: float = 25.0
    pea_ood_min_expert_bias: float = -2.0
    pea_ood_max_expert_bias: float = 2.0

    # OPS
    ops_min_assets: int = 2
    ops_max_assets: int = 6
    # Returns modeled as log-normal via Normal log-returns
    ops_min_mu: float = -0.02
    ops_max_mu: float = 0.02
    ops_min_sigma: float = 0.01
    ops_max_sigma: float = 0.10
    # Correlation strength for covariance generation
    ops_min_corr: float = -0.3
    ops_max_corr: float = 0.6
    # OOD OPS
    ops_ood_min_assets: int = 8
    ops_ood_max_assets: int = 12
    # skewed: allow more negative drifts and stronger correlations
    ops_ood_min_mu: float = -0.04
    ops_ood_max_mu: float = 0.02
    ops_ood_min_sigma: float = 0.14
    ops_ood_max_sigma: float = 0.28
    ops_ood_min_corr: float = -0.92
    ops_ood_max_corr: float = 0.92
    # Reward shaping/scaling for OPS: reward = scale * log(w^T x) + shift; make reward larger so might be easier for the LLM to parse / understand
    ops_reward_scale: float = 100.0
    ops_reward_shift: float = 0.0


