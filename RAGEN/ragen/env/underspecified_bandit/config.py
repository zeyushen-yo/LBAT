from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class UnderspecifiedBanditEnvConfig:
    min_arms: int = 2
    max_arms: int = 6
    max_horizon: int = 20
    min_horizon: int = 20
    # max_horizon: int = 5 # for testing
    # min_horizon: int = 5 # for testing
    possible_bandit_types: Tuple[str, ...] = ("beta-bernoulli", "gaussian-gaussian", "poisson-gamma")
    
    # for beta-bernoulli
    max_alpha: float = 2.0
    max_beta: float = 2.0
    min_alpha: float = 0.25
    min_beta: float = 0.25
    max_p: float = 0.9
    min_p: float = 0.1

    # for gaussian-gaussian
    max_mu_prior: float = 1.2
    min_mu_prior: float = 0.2
    max_sigma_prior: float = 2.0
    min_sigma_prior: float = 1.0
    max_mu_true: float = 1.0
    min_mu_true: float = 0.3
    max_sigma_true: float = 0.12
    min_sigma_true: float = 0.03

    # for poisson-gamma
    max_a: float = 3.0
    max_b: float = 3.0
    min_a: float = 0.3
    min_b: float = 0.3
    max_lambda: float = 2.5
    min_lambda: float = 0.4

    render_mode: str = "text"