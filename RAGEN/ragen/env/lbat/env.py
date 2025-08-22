from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Optional
import logging
from ragen.sandbox.prompt_rewriter import rewrite_text_via_sandbox

from ragen.env.base import BaseDiscreteActionEnv
from .config import LBATEnvConfig
from .utils import expectation, variance, sample
from .accel import BanditParams, PEAParams, OPSParams
from .algos import hedge_brier, ons_log_loss, ucb_beta_bernoulli, ucb_gaussian_gaussian, ucb_poisson_gamma

logger = logging.getLogger(__name__)


class LBATEnv(BaseDiscreteActionEnv, gym.Env):
    """
    Unified environment supporting three families:
    - MAB (beta-bernoulli, gaussian-gaussian, poisson-gamma)
    - PEA (experts with probabilistic predictions, Brier reward)
    - OPS (portfolio weights over log-normal returns, log-wealth reward)
    """
    def __init__(self, config: Optional[LBATEnvConfig] = None, *, buffer: Optional[List[Dict[str, Any]]] = None):
        BaseDiscreteActionEnv.__init__(self)
        self.config = config if config is not None else LBATEnvConfig()
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        self.buffer: List[Dict[str, Any]] = buffer if buffer is not None else []

        # Common
        self.horizon: int = 20
        self.current_step: int = 0
        self.cumulative_reward: float = 0.0
        self.action_names: List[str] = []
        self.ACTION_SPACE: Optional[gym.spaces.discrete.Discrete] = None
        self.render_cache: Optional[str] = None

        # Mode selection: 'mab' | 'pea' | 'ops'
        self.family: str = 'mab'

        # MAB specifics
        self.type: str = 'beta-bernoulli'
        self.num_arms: int = 2
        self.arm_prior_alpha: List[float] = []
        self.arm_prior_beta: List[float] = []
        self.arm_true_p: List[float] = []
        self.arm_prior_mu: List[float] = []
        self.arm_prior_sigma: List[float] = []
        self.arm_true_mu: List[float] = []
        self.arm_true_sigma: List[float] = []
        self.arm_prior_a: List[float] = []
        self.arm_prior_b: List[float] = []
        self.arm_true_lambda: List[float] = []
        self._mu_star: float = 0.0
        self._mu_bar: float = 0.0

        # PEA specifics
        self.num_experts: int = 2
        self.truth_p: float = 0.5
        self.expert_kappa: List[float] = []
        self.expert_bias: List[float] = []

        # OPS specifics
        self.num_assets: int = 2
        self.ops_mu: List[float] = []
        self.ops_sigma: List[float] = []
        self.ops_corr_upper: List[float] = []
        # Per-instance reward shift to keep OPS rewards non-negative without affecting regret
        self._ops_reward_shift_runtime: float = 0.0

        # next-override hooks per family
        self._next_params_override: Optional[Dict[str, Any]] = None

        # Pre-sampled per-round data to ensure protagonist and antagonist share the same realizations
        self._mab_rewards: Optional[np.ndarray] = None           # shape (T, K)
        self._pea_expert_probs: Optional[np.ndarray] = None      # shape (T, N)
        self._pea_labels: Optional[np.ndarray] = None            # shape (T,)
        self._ops_returns: Optional[np.ndarray] = None           # shape (T, d)

    # ---- Params I/O ----
    def _as_params(self) -> Dict[str, Any]:
        if self.family == 'mab':
            if self.type == "beta-bernoulli":
                p = BanditParams(
                    type=self.type, num_arms=self.num_arms, horizon=self.horizon,
                    arm_prior_alpha=self.arm_prior_alpha, arm_prior_beta=self.arm_prior_beta,
                    arm_true_p=self.arm_true_p
                )
            elif self.type == "gaussian-gaussian":
                p = BanditParams(
                    type=self.type, num_arms=self.num_arms, horizon=self.horizon,
                    arm_prior_mu=self.arm_prior_mu, arm_prior_sigma=self.arm_prior_sigma,
                    arm_true_mu=self.arm_true_mu, arm_true_sigma=self.arm_true_sigma
                )
            elif self.type == "poisson-gamma":
                p = BanditParams(
                    type=self.type, num_arms=self.num_arms, horizon=self.horizon,
                    arm_prior_a=self.arm_prior_a, arm_prior_b=self.arm_prior_b,
                    arm_true_lambda=self.arm_true_lambda
                )
            else:
                raise ValueError(f"Unknown bandit type: {self.type}")
            return {"family": "mab", "params": p.to_dict()}

        if self.family == 'pea':
            p = PEAParams(
                num_experts=self.num_experts,
                horizon=self.horizon,
                truth_p=self.truth_p,
                expert_kappa=self.expert_kappa,
                expert_bias=self.expert_bias,
            )
            return {"family": "pea", "params": p.to_dict()}

        if self.family == 'ops':
            p = OPSParams(
                num_assets=self.num_assets,
                horizon=self.horizon,
                mu=self.ops_mu,
                sigma=self.ops_sigma,
                corr_upper=self.ops_corr_upper,
            )
            return {"family": "ops", "params": p.to_dict()}

        raise ValueError(f"Unknown family: {self.family}")

    def _apply_params(self, family: str, p_dict: Dict[str, Any]) -> None:
        if family == 'mab':
            p = BanditParams(**p_dict)
            self.family = 'mab'
            self.type = p.type
            self.num_arms = int(p.num_arms)
            self.horizon = int(p.horizon)
            if p.type == "beta-bernoulli":
                self.arm_prior_alpha = list(p.arm_prior_alpha)
                self.arm_prior_beta = list(p.arm_prior_beta)
                self.arm_true_p = list(p.arm_true_p)
            elif p.type == "gaussian-gaussian":
                self.arm_prior_mu = list(p.arm_prior_mu)
                self.arm_prior_sigma = list(p.arm_prior_sigma)
                self.arm_true_mu = list(p.arm_true_mu)
                self.arm_true_sigma = list(p.arm_true_sigma)
            elif p.type == "poisson-gamma":
                self.arm_prior_a = list(p.arm_prior_a)
                self.arm_prior_b = list(p.arm_prior_b)
                self.arm_true_lambda = list(p.arm_true_lambda)
            return

        if family == 'pea':
            p = PEAParams(**p_dict)
            self.family = 'pea'
            self.horizon = int(p.horizon)
            self.num_experts = int(p.num_experts)
            self.truth_p = float(p.truth_p)
            self.expert_kappa = list(p.expert_kappa)
            self.expert_bias = list(p.expert_bias)
            return

        if family == 'ops':
            p = OPSParams(**p_dict)
            self.family = 'ops'
            self.horizon = int(p.horizon)
            self.num_assets = int(p.num_assets)
            self.ops_mu = list(p.mu)
            self.ops_sigma = list(p.sigma)
            self.ops_corr_upper = list(p.corr_upper)
            return

        raise ValueError(f"Unknown family to apply: {family}")

    # ---- Family helpers ----
    def _true_means_mab(self) -> np.ndarray:
        if self.type == "beta-bernoulli":
            return np.asarray(self.arm_true_p, dtype=float)
        if self.type == "gaussian-gaussian":
            return np.asarray(self.arm_true_mu, dtype=float)
        if self.type == "poisson-gamma":
            return np.asarray(self.arm_true_lambda, dtype=float)
        raise ValueError(f"Unknown bandit type: {self.type}")

    def _build_ops_cov(self) -> np.ndarray:
        d = self.num_assets
        corr = np.eye(d)
        iu = np.triu_indices(d, k=1)
        vals = np.asarray(self.ops_corr_upper, dtype=float)
        corr[iu] = vals
        corr[(iu[1], iu[0])] = vals

        # PSD fix via eigen clipping
        w, V = np.linalg.eigh(corr)
        w = np.clip(w, 1e-6, None)
        corr_psd = (V * w) @ V.T

        # Re-normalize to correlation (diag exactly 1)
        dstd = np.sqrt(np.clip(np.diag(corr_psd), 1e-12, None))
        corr_psd = corr_psd / (dstd[:, None] * dstd[None, :])

        s = np.asarray(self.ops_sigma, dtype=float)
        S = np.diag(s)  # s are standard deviations of log-returns
        cov = S @ corr_psd @ S
        return cov


    # ---- Gym API ----
    def reset(self, *, seed: Optional[int] = None, **kwargs):
        gym.Env.reset(self, seed=seed)
        rng: np.random.Generator = self.np_random

        # Family selection policy and overrides
        families = ['mab', 'pea', 'ops']
        if self._next_params_override is not None:
            fam = self._next_params_override.get('family')
            self._apply_params(fam, self._next_params_override.get('params'))
            self._next_params_override = None
        else:
            self.family = str(kwargs.get('family', rng.choice(families)))
            self.horizon = int(rng.integers(self.config.min_horizon, self.config.max_horizon + 1))

            if self.family == 'mab':
                self.num_arms = int(rng.integers(self.config.min_arms, self.config.max_arms + 1))
                self.type = rng.choice(self.config.possible_bandit_types)
                if self.type == "beta-bernoulli":
                    self.arm_prior_alpha = rng.uniform(self.config.min_alpha, self.config.max_alpha, size=self.num_arms).tolist()
                    self.arm_prior_beta = rng.uniform(self.config.min_beta, self.config.max_beta, size=self.num_arms).tolist()
                    self.arm_true_p = rng.uniform(self.config.min_p, self.config.max_p, size=self.num_arms).tolist()
                elif self.type == "gaussian-gaussian":
                    self.arm_prior_mu = rng.uniform(self.config.min_mu_prior, self.config.max_mu_prior, size=self.num_arms).tolist()
                    self.arm_prior_sigma = rng.uniform(self.config.min_sigma_prior, self.config.max_sigma_prior, size=self.num_arms).tolist()
                    self.arm_true_mu = rng.uniform(self.config.min_mu_true, self.config.max_mu_true, size=self.num_arms).tolist()
                    self.arm_true_sigma = rng.uniform(self.config.min_sigma_true, self.config.max_sigma_true, size=self.num_arms).tolist()
                elif self.type == "poisson-gamma":
                    self.arm_prior_a = rng.uniform(self.config.min_a, self.config.max_a, size=self.num_arms).tolist()
                    self.arm_prior_b = rng.uniform(self.config.min_b, self.config.max_b, size=self.num_arms).tolist()
                    self.arm_true_lambda = rng.uniform(self.config.min_lambda, self.config.max_lambda, size=self.num_arms).tolist()
                else:
                    raise ValueError(f"Unknown bandit type: {self.type}")

            elif self.family == 'pea':
                self.num_experts = int(rng.integers(self.config.pea_min_experts, self.config.pea_max_experts + 1))
                self.truth_p = float(rng.uniform(self.config.pea_min_truth_p, self.config.pea_max_truth_p))
                self.expert_kappa = rng.uniform(self.config.pea_min_expert_kappa, self.config.pea_max_expert_kappa, size=self.num_experts).tolist()
                self.expert_bias = rng.uniform(self.config.pea_min_expert_bias, self.config.pea_max_expert_bias, size=self.num_experts).tolist()

            elif self.family == 'ops':
                self.num_assets = int(rng.integers(self.config.ops_min_assets, self.config.ops_max_assets + 1))
                self.ops_mu = rng.uniform(self.config.ops_min_mu, self.config.ops_max_mu, size=self.num_assets).tolist()
                self.ops_sigma = rng.uniform(self.config.ops_min_sigma, self.config.ops_max_sigma, size=self.num_assets).tolist()
                # random correlation upper entries, then we will PSD-correct
                num_upper = self.num_assets * (self.num_assets - 1) // 2
                self.ops_corr_upper = rng.uniform(self.config.ops_min_corr, self.config.ops_max_corr, size=num_upper).tolist()
            else:
                raise ValueError(f"Unknown family: {self.family}")

        self.cumulative_reward = 0.0
        self.current_step = 0
        # Clear pre-sampled sequences
        self._mab_rewards = None
        self._pea_expert_probs = None
        self._pea_labels = None
        self._ops_returns = None

        lines: List[str] = []
        if self.family == 'mab':
            lines.append(
                "You need to solve a decision-making problem. In each iteration you must choose exactly one of the available arms. "
                "After pulling an arm you observe a stochastic reward drawn from that arm’s unknown distribution. Your objective is to maximise the total reward over the horizon. "
                "Wrap your choice in <answer></answer> tags, e.g. <answer>1</answer> if you choose the first arm. "
                "Be succinct in your reasoning, and do not try to apply any specific algorithms."
            )
            lines.append(f"This instance has {self.num_arms} arms, and in each round, you need to pull one of the arms.")
            if self.type == "beta-bernoulli":
                for i, (alpha, beta) in enumerate(zip(self.arm_prior_alpha, self.arm_prior_beta)):
                    prior = {"type": "beta", "alpha": alpha, "beta": beta}
                    exp = expectation(prior); var = variance(prior)
                    lines.append(
                        f"Your prior belief on the reward of pulling arm {i} is Beta(alpha={alpha:.2f}, beta={beta:.2f}). Expectation {exp:.2f}, std dev {var:.2f}."
                    )
            elif self.type == "gaussian-gaussian":
                for i, (mu, sigma) in enumerate(zip(self.arm_prior_mu, self.arm_prior_sigma)):
                    prior = {"type": "gaussian", "mu": mu, "sigma": sigma}
                    exp = expectation(prior); var = variance(prior)
                    lines.append(
                        f"Your prior belief on the reward of pulling arm {i} is Gaussian(mu={mu:.2f}, sigma={sigma:.2f}). Expectation {exp:.2f}, std dev {var:.2f}."
                    )
            elif self.type == "poisson-gamma":
                for i, (a, b) in enumerate(zip(self.arm_prior_a, self.arm_prior_b)):
                    prior = {"type": "gamma", "a": a, "b": b}
                    exp = expectation(prior); var = variance(prior)
                    lines.append(
                        f"Your prior belief on the reward of pulling arm {i} is Gamma(a={a:.2f}, b={b:.2f}). Expectation {exp:.2f}, std dev {var:.2f}."
                    )
            start = 0
            self.action_names = [f"arm_{i}" for i in range(self.num_arms)]
            self.ACTION_SPACE = gym.spaces.discrete.Discrete(self.num_arms, start=start)
            true_mu = self._true_means_mab()
            self._mu_star = float(np.max(true_mu))
            self._mu_bar = float(np.mean(true_mu))
            # Pre-sample per-round per-arm rewards so protagonist and antagonist share the same realizations
            T = int(self.horizon)
            K = int(self.num_arms)
            if self.type == "beta-bernoulli":
                p = np.asarray(self.arm_true_p, dtype=float)
                self._mab_rewards = np.vstack([
                    rng.binomial(1, p, size=K).astype(float) for _ in range(T)
                ])
            elif self.type == "gaussian-gaussian":
                mu = np.asarray(self.arm_true_mu, dtype=float)
                sigma = np.asarray(self.arm_true_sigma, dtype=float)
                self._mab_rewards = np.vstack([
                    rng.normal(mu, sigma, size=K).astype(float) for _ in range(T)
                ])
            elif self.type == "poisson-gamma":
                lam = np.asarray(self.arm_true_lambda, dtype=float)
                self._mab_rewards = np.vstack([
                    rng.poisson(lam, size=K).astype(float) for _ in range(T)
                ])

        elif self.family == 'pea':
            lines.append(
                "You face a prediction-with-expert-advice problem. In each round, experts output probabilities for a binary label, and you must output a probability for y=1. "
                "In each round, the expert forecasts will be listed in a fixed order, e.g. [0.1, 0.2] means the first expert predicts 0.1 and the second expert predicts 0.2. Some experts may be more accurate than others. "
                "Output an integer index of the probability bucket from 0..10, where k means probability k/10. Wrap your answer in <answer></answer>, e.g. <answer>6</answer> if your estimated probability is 0.6. "
                "Be succinct in your reasoning, and do not try to apply any specific algorithms."
            )
            lines.append(f"This instance has {self.num_experts} experts and horizon {self.horizon}.")
            for i, (kappa, bias) in enumerate(zip(self.expert_kappa, self.expert_bias)):
                a = float(kappa) * float(self.truth_p)
                b = float(kappa) * float(1.0 - self.truth_p)
                # I think we shouldn't tell the LLM any information about the experts?
                # lines.append(f"Expert {i} predictions are drawn from Beta(alpha={a:.2f}, beta={b:.2f}) with kappa={kappa:.2f} tied to truth_p={self.truth_p:.2f}, then logit-bias {bias:.2f} is applied.")
            self.action_names = [f"p_{k}" for k in range(11)]
            self.ACTION_SPACE = gym.spaces.discrete.Discrete(11, start=0)
            # Pre-sample expert probabilities and labels for all rounds
            T = int(self.horizon)
            N = int(self.num_experts)
            self._pea_expert_probs = np.empty((T, N), dtype=float)
            alpha_vec = np.asarray(self.expert_kappa, dtype=float) * float(self.truth_p)
            beta_vec = np.asarray(self.expert_kappa, dtype=float) * float(1.0 - self.truth_p)
            bias_vec = np.asarray(self.expert_bias, dtype=float)
            for t in range(T):
                p_raw = rng.beta(alpha_vec, beta_vec)
                # apply logit-bias and squash back with sigmoid
                p_raw = np.clip(p_raw, 1e-6, 1 - 1e-6)
                z = np.log(p_raw / (1.0 - p_raw)) + bias_vec
                p_biased = 1.0 / (1.0 + np.exp(-z))
                self._pea_expert_probs[t] = p_biased
            self._pea_labels = rng.binomial(1, float(self.truth_p), size=T).astype(int)
            # Reveal current round (t=0) expert predictions at the start
            probs0 = ", ".join(f"{p:.2f}" for p in self._pea_expert_probs[0])
            lines.append(f"Round 1/{self.horizon}: expert probability forecasts for y=1 are: [{probs0}].")

        elif self.family == 'ops':
            lines.append(
                "You face an online portfolio selection problem. In each round, allocate a unit budget across d assets using a weight vector w that sums to 1 (w_i ≥ 0). "
                "In each round, the gross returns for each asset will be listed in a fixed order, e.g. [0.01, 0.02] means the first asset returns 0.01 and the second asset returns 0.02. "
                "Format your answer as a comma-separated vector inside <answer></answer>, e.g., <answer>0.2, 0.3, 0.5</answer> for d=3. "
                "Be succinct in your reasoning, and do not try to apply any specific algorithms."
            )
            lines.append(f"This instance has {self.num_assets} assets and horizon {self.horizon}.")
            self.action_names = []
            # Optional: continuous action space specification (not used by parser, but informative)
            self.ACTION_SPACE = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)
            # Pre-sample per-round gross returns for all assets
            T = int(self.horizon)
            d = int(self.num_assets)
            mu = np.asarray(self.ops_mu, dtype=float)
            cov = self._build_ops_cov()
            log_r = rng.multivariate_normal(mean=mu, cov=cov, size=T)
            self._ops_returns = np.exp(log_r)
            # Choose a per-step shift so that for any valid portfolio w (simplex),
            # scale * log(w^T x_t) + shift >= 0 for all t. Since w^T x_t >= min_i x_{t,i},
            # a sufficient shift is -scale * min_{t,i} log x_{t,i}.
            min_log_return = float(np.min(np.log(self._ops_returns)))
            required_shift = - self.config.ops_reward_scale * min_log_return
            # Do not reduce an explicitly configured positive shift
            self._ops_reward_shift_runtime = max(float(self.config.ops_reward_shift), float(required_shift))
        else:
            raise ValueError(f"Unknown family: {self.family}")

        # Compose initial prompt
        initial_text = "\n".join(lines)
        # Optional rephrasing via Princeton AI Sandbox (GPT-4o) with probability gate
        do_rewrite = False
        try:
            p = float(self.config.rewrite_probability)
            if p > 0.0:
                # Use env RNG for reproducibility
                do_rewrite = bool(self.np_random.random() < p)
        except Exception as e:
            print(f"[LBATEnv.reset] rewrite_probability error: {e}")
            do_rewrite = False

        if do_rewrite:
            try:
                rewritten = rewrite_text_via_sandbox(initial_text, mode="reset")
                self.render_cache = rewritten if isinstance(rewritten, str) and len(rewritten) > 0 else initial_text
                print("rewrited") # for testing
            except Exception as e:
                print(f"[LBATEnv.reset] prompt rewrite failed: {e}")
                self.render_cache = initial_text
        else:
            self.render_cache = initial_text
        return self.render_cache

    def get_all_actions(self):
        # Discrete families (MAB, PEA)
        if isinstance(self.ACTION_SPACE, gym.spaces.Discrete):
            start = int(self.ACTION_SPACE.start)
            return list(range(start, start + int(self.ACTION_SPACE.n)))

        # Continuous family (OPS): return a schema + examples (for UI/help text)
        if isinstance(self.ACTION_SPACE, gym.spaces.Box):
            d = int(self.ACTION_SPACE.shape[0])
            examples = []
            # one-hots
            for i in range(d):
                e = [0.0] * d
                e[i] = 1.0
                examples.append(e)
            # uniform
            examples.append([1.0 / d] * d)
            return {
                "type": "continuous-simplex",
                "dimension": d,
                "format": "comma-separated vector of length d summing to 1 (nonnegative)",
                "examples": examples
            }
        return None

    def compute_step_reward(self, action: Any):
        if self.family == 'mab':
            t = int(self.current_step - 1)
            assert self._mab_rewards is not None, "MAB rewards not pre-sampled"
            return float(self._mab_rewards[t, int(action)])

        if self.family == 'pea':
            t = int(self.current_step - 1)
            assert self._pea_labels is not None, "PEA labels not pre-sampled"
            y = int(self._pea_labels[t])
            # Map discrete action 0..10 -> probability
            p_agent = float(np.clip(action, 0, 10)) / 10.0
            # Positive reward shaping: 1 - squared error in [0,1]
            return 1.0 - (p_agent - y) ** 2

        if self.family == 'ops':
            def _parse_weights(a: Any, d: int) -> Optional[np.ndarray]:
                try:
                    if isinstance(a, (list, tuple, np.ndarray)):
                        w = np.asarray(a, dtype=float)
                    elif isinstance(a, str):
                        # remove brackets and split by comma
                        s = a.strip().strip('[]()')
                        parts = [p for p in s.replace("\n", " ").split(',') if p.strip() != ""]
                        # if no commas, try whitespace split
                        if len(parts) <= 1:
                            parts = [p for p in s.split() if p.strip() != ""]
                        w = np.asarray([float(p) for p in parts], dtype=float)
                    else:
                        return None
                except Exception as e:
                    print(f"OPS parse error: {e}")
                    return None
                if w.size != d:
                    return None
                w = np.maximum(w, 0.0)
                s = float(w.sum())
                if s <= 0.0 or not np.isfinite(s):
                    return None
                w = w / s
                return w

            t = int(self.current_step - 1)
            d = self.num_assets
            w = _parse_weights(action, d)
            if w is None:
                return -1.0
            assert self._ops_returns is not None, "OPS returns not pre-sampled"
            x = np.asarray(self._ops_returns[t], dtype=float)
            denom = max(1e-12, float(np.dot(w, x)))
            # Scale and per-instance shift to make rewards non-negative without changing regret ordering
            return float(self.config.ops_reward_scale * np.log(denom) + self._ops_reward_shift_runtime)

        raise ValueError(f"Unknown family: {self.family}")

    def _antagonist_total_reward(self) -> float:
        if self.family == 'mab':
            assert self._mab_rewards is not None, "MAB rewards not pre-sampled"
            if self.type == 'beta-bernoulli':
                return ucb_beta_bernoulli(
                    self.arm_prior_alpha, self.arm_prior_beta, self.arm_true_p, self.horizon,
                    rng=self.np_random, pre_rewards=self._mab_rewards
                )
            if self.type == 'gaussian-gaussian':
                return ucb_gaussian_gaussian(
                    self.arm_prior_mu, self.arm_prior_sigma, self.arm_true_mu, self.arm_true_sigma, self.horizon,
                    rng=self.np_random, pre_rewards=self._mab_rewards
                )
            if self.type == 'poisson-gamma':
                return ucb_poisson_gamma(
                    self.arm_prior_a, self.arm_prior_b, self.arm_true_lambda, self.horizon,
                    rng=self.np_random, pre_rewards=self._mab_rewards
                )
            raise ValueError

        if self.family == 'pea':
            assert self._pea_expert_probs is not None and self._pea_labels is not None, "PEA sequences not pre-sampled"
            return hedge_brier(self._pea_expert_probs, self._pea_labels)

        if self.family == 'ops':
            assert self._ops_returns is not None, "OPS returns not pre-sampled"
            return ons_log_loss(self._ops_returns, int(self.horizon))

        raise ValueError(f"Unknown family: {self.family}")

    def step(self, action):
        self.current_step += 1
        if self.family == 'ops':
            reward = self.compute_step_reward(action)
            valid = bool(reward > -1.0)
            msg_head = ("You provided a portfolio allocation vector and received reward "
                        f"{reward:.4f}.\n") if valid else "Your allocation was invalid and will be penalized."
        else:
            try:
                action = int(action)
            except Exception as e:
                print(f"action parse error: {e}")
                action = -1
            # validate
            if action < 0 or action >= self.ACTION_SPACE.n:
                reward = -1.0
                msg_head = "You failed to take a valid action and will be penalized."
            else:
                reward = self.compute_step_reward(action)
                msg_head = f"You took action {action} and received reward {reward:.4f}.\n"

        self.cumulative_reward += float(reward)
        terminated = self.current_step >= self.horizon
        if not terminated:
            # Craft next-round prompt details per family
            if self.family == 'pea':
                # Reveal current round (t) expert predictions for the NEXT round
                t_next = int(self.current_step)
                if self._pea_expert_probs is not None and t_next < int(self.horizon):
                    probs = ", ".join(f"{p:.2f}" for p in self._pea_expert_probs[t_next])
                    extra = f"Round {t_next+1}/{self.horizon}: expert probability forecasts for y=1 are: [{probs}].\n"
                else:
                    extra = ""
                observation = msg_head + extra + "Choose your next action."
                info = {"cumulative_reward": self.cumulative_reward, "action_is_valid": 0 <= action < self.ACTION_SPACE.n}
            elif self.family == 'ops':
                # Reveal previous round returns at the start of each round
                t_prev = int(self.current_step - 1)
                if self._ops_returns is not None and 0 <= t_prev < int(self.horizon):
                    prev_ret = ", ".join(f"{x:.4f}" for x in self._ops_returns[t_prev])
                    extra = f"Last round gross returns for each asset were: [{prev_ret}].\n"
                else:
                    extra = ""
                observation = msg_head + extra + "Choose your next action."
                info = {"cumulative_reward": self.cumulative_reward, "action_is_valid": bool(reward > -1.0)}
            else:
                observation = msg_head + ("Choose your next action.")
                info = {"cumulative_reward": self.cumulative_reward, "action_is_valid": 0 <= action < self.ACTION_SPACE.n}
        else:
            # Difficulty normalization by random or antagonist
            if self.family == 'mab':
                true_mu = self._true_means_mab()
                mu_star = float(np.max(true_mu))
                mu_bar = float(np.mean(true_mu))
                antagonist_reward = self._antagonist_total_reward()
                antagonist_regret = self.horizon * mu_star - antagonist_reward
                cumulative_regret = self.horizon * mu_star - self.cumulative_reward
                random_regret = self.horizon * (mu_star - mu_bar)
            elif self.family == 'pea':
                # Antagonist evaluated on the same pre-sampled sequence
                # Step reward is 1 - (p - y)^2 in [0,1].
                # hedge_brier() returns sum of - (p - y)^2. Convert to shaped reward by adding T.
                assert self._pea_labels is not None and self._pea_expert_probs is not None
                T = int(self.horizon)
                labels = np.asarray(self._pea_labels, dtype=float)
                P = np.asarray(self._pea_expert_probs, dtype=float)  # shape (T, N)
                antagonist_reward = float(T) + self._antagonist_total_reward()
                # Optimal comparator: best single expert in hindsight
                expert_losses = np.sum((P - labels[:, None]) ** 2, axis=0)
                opt_reward = float(T - float(np.min(expert_losses)))
                # Naive baseline: constant p = 0.5 across all rounds, evaluated on realized labels
                random_reward = float(T - np.sum((0.5 - labels) ** 2))
                antagonist_regret = opt_reward - antagonist_reward
                cumulative_regret = opt_reward - self.cumulative_reward
                random_regret = opt_reward - random_reward
            elif self.family == 'ops':
                # Antagonist evaluated on the same pre-sampled sequence
                # Scale antagonist to match step reward scaling and per-instance shift
                antagonist_reward = (
                    self.config.ops_reward_scale * self._antagonist_total_reward()
                    + self._ops_reward_shift_runtime * self.horizon
                )
                # Optimal comparator: best single asset in hindsight (CRP over vertices)
                assert self._ops_returns is not None
                per_asset_sum_log = np.sum(np.log(self._ops_returns), axis=0)  # shape (d,)
                opt_unscaled = float(np.max(per_asset_sum_log))
                opt_reward = self.config.ops_reward_scale * opt_unscaled + self._ops_reward_shift_runtime * self.horizon
                # Naive baseline: uniform portfolio every round
                uniform_wtx = np.mean(self._ops_returns, axis=1)  # shape (T,)
                rand_unscaled = float(np.sum(np.log(np.maximum(1e-12, uniform_wtx))))
                random_reward = self.config.ops_reward_scale * rand_unscaled + self._ops_reward_shift_runtime * self.horizon
                antagonist_regret = opt_reward - antagonist_reward
                cumulative_regret = opt_reward - self.cumulative_reward
                random_regret = opt_reward - random_reward
            else:
                raise ValueError

            gap = (cumulative_regret - antagonist_regret) / ((max(antagonist_regret, random_regret) - min(antagonist_regret, random_regret)) + 1e-12)

            observation = msg_head + "Game finished. Thank you for playing!"
            info = {
                "protagonist_regret": cumulative_regret,
                "antagonist_regret": antagonist_regret,
                "random_regret": random_regret,
                "protagonist_antagonist_gap": gap,
            }
            # Mark LBAT episodes as successful when terminated (non-binary task completion)
            info["success"] = True

        # Optional rephrasing via Princeton AI Sandbox (GPT-4o) with probability gate
        do_rewrite = False
        try:
            p = float(self.config.rewrite_probability)
            if p > 0.0:
                do_rewrite = bool(self.np_random.random() < p)
        except Exception as e:
            print(f"[LBATEnv.step] rewrite_probability error: {e}")
            do_rewrite = False
        if do_rewrite:
            try:
                rewritten_obs = rewrite_text_via_sandbox(observation, mode="step")
                observation = rewritten_obs if isinstance(rewritten_obs, str) and len(rewritten_obs) > 0 else observation
                print("rewrited") # for testing
            except Exception as e:
                print(f"[LBATEnv.step] prompt rewrite failed: {e}")

        self.render_cache = observation
        return observation, float(reward), bool(terminated), info

    def render(self):
        return self.render_cache

    def close(self):
        self.render_cache = None

    def set_params_for_next_reset(self, p_any: Dict[str, Any]):
        family = p_any.get('family')
        params = p_any.get('params')
        if family is None or params is None:
            return
        self._next_params_override = {"family": family, "params": params}

    def export_params(self) -> Dict[str, Any]:
        return self._as_params()