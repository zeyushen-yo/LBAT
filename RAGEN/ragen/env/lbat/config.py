from dataclasses import dataclass
from typing import Tuple


@dataclass
class LBATEnvConfig:
    # Shared
    render_mode: str = "text"
    min_horizon: int = 20
    max_horizon: int = 20
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
    # gaussian-gaussian
    max_mu_prior: float = 1.2
    min_mu_prior: float = 0.2
    max_sigma_prior: float = 2.0
    min_sigma_prior: float = 1.0
    max_mu_true: float = 1.0
    min_mu_true: float = 0.3
    max_sigma_true: float = 0.12
    min_sigma_true: float = 0.03
    # poisson-gamma
    max_a: float = 3.0
    max_b: float = 3.0
    min_a: float = 0.3
    min_b: float = 0.3
    max_lambda: float = 2.5
    min_lambda: float = 0.4

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
    # Reward shaping/scaling for OPS: reward = scale * log(w^T x) + shift; make reward larger so might be easier for the LLM to parse / understand
    ops_reward_scale: float = 100.0
    ops_reward_shift: float = 0.0


