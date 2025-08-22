from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging

from ragen.env.base import BaseDiscreteActionEnv
from .config import UnderspecifiedBanditEnvConfig
from .utils import expectation, variance, sample
from .ucb import ucb_beta_bernoulli, ucb_gaussian_gaussian, ucb_poisson_gamma
from .ids import ids_beta_bernoulli, ids_gaussian_gaussian, ids_poisson_gamma
from .accel import BanditParams, AccelBuffer

logger = logging.getLogger(__name__)

class UnderspecifiedBanditEnv(BaseDiscreteActionEnv, gym.Env):

    def __init__(self, config: Optional[UnderspecifiedBanditEnvConfig] = None, *, buffer: Optional[List[Dict[str, Any]]] = None):
        BaseDiscreteActionEnv.__init__(self)
        self.config = config if config is not None else UnderspecifiedBanditEnvConfig()
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        self.buffer: List[Dict[str, Any]] = buffer if buffer is not None else []
        self.num_arms: int = 2
        self.cumulative_reward: float

        #self.type is chosen from beta-bernoulli, gaussian-gaussian, or poisson-gamma
        self.type: str

        # if bandit type is beta-bernoulli, prior is beta(alpha, beta) and true distribution is bernoulli(p)
        self.arm_prior_alpha: List[int]
        self.arm_prior_beta: List[int]
        self.arm_true_p: List[int]

        # if bandit type is gaussian-gaussian, both prior and true distributions are gaussians
        self.arm_prior_mu: List[int]
        self.arm_prior_sigma: List[int]
        self.arm_true_mu: List[int]
        self.arm_true_sigma: List[int]

        # if bandit type is poisson-gamma, prior is gamma(a, b) and true distribution is poisson(lambda)
        self.arm_prior_a: List[int]
        self.arm_prior_b: List[int]
        self.arm_true_lambda: List[int]

        # the best arm
        self._mu_star: float = 0.0

        # expected reward of pulling a random arm
        self._mu_bar: float = 0.0

        self.action_names: List[str] = []
        self.current_step: int = 0
        self.horizon: int = 20
        self.ACTION_SPACE: Optional[gym.spaces.discrete.Discrete] = None
        self.render_cache = None
        
        self._next_params_override: Optional[BanditParams] = None

    def _as_params(self) -> BanditParams:
        if self.type == "beta-bernoulli":
            return BanditParams(
                type=self.type, num_arms=self.num_arms, horizon=self.horizon,
                arm_prior_alpha=self.arm_prior_alpha, arm_prior_beta=self.arm_prior_beta,
                arm_true_p=self.arm_true_p
            )
        elif self.type == "gaussian-gaussian":
            return BanditParams(
                type=self.type, num_arms=self.num_arms, horizon=self.horizon,
                arm_prior_mu=self.arm_prior_mu, arm_prior_sigma=self.arm_prior_sigma,
                arm_true_mu=self.arm_true_mu, arm_true_sigma=self.arm_true_sigma
            )
        elif self.type == "poisson-gamma":
            return BanditParams(
                type=self.type, num_arms=self.num_arms, horizon=self.horizon,
                arm_prior_a=self.arm_prior_a, arm_prior_b=self.arm_prior_b,
                arm_true_lambda=self.arm_true_lambda
            )
        else:
            raise ValueError(f"Unknown bandit type: {self.type}")

    def _apply_params(self, p: BanditParams) -> None:
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


    def _true_means(self) -> np.ndarray:
        if self.type == "beta-bernoulli":
            return np.asarray(self.arm_true_p, dtype=float)
        elif self.type == "gaussian-gaussian":
            return np.asarray(self.arm_true_mu, dtype=float)
        elif self.type == "poisson-gamma":
            return np.asarray(self.arm_true_lambda, dtype=float)
        else:
            raise ValueError(f"Unknown bandit type: {self.type}")


    def reset(self, *, seed: Optional[int] = None, **kwargs):
        # randomly reset; for ablation (but probably it just works?)
        gym.Env.reset(self, seed=seed)
        rng: np.random.Generator = self.np_random

        if self._next_params_override is not None:
            self._apply_params(self._next_params_override)
            self._next_params_override = None
        else:
            self.num_arms = int(rng.integers(self.config.min_arms, self.config.max_arms + 1))
            self.type = rng.choice(self.config.possible_bandit_types)
            self.horizon = int(rng.integers(self.config.min_horizon, self.config.max_horizon + 1))
            
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

        self.cumulative_reward = 0.0
        intro = (
            "You need to solve a decision-making problem. "
            "In each iteration you must choose exactly one of the available arms. "
            "After pulling an arm you observe a stochastic reward drawn from that "
            "arm’s unknown distribution. Your objective is to maximise the total "
            "reward over the horizon. "
            "Wrap your choice in <answer></answer> tags, e.g. <answer>1</answer> if you choose the first arm. "
            "Do not include any other things except the number of the arm you choose. "
            "Be succinct in your reasoning, and do not use specific algorithms. Rather, reason about your choice on an abstract level."
        )

        lines: list[str] = [intro, f"This instance has {self.num_arms} arms, and in each round, you need to pull one of the arms."]   

        if self.type == "beta-bernoulli":
            for i, (alpha, beta) in enumerate(zip(self.arm_prior_alpha, self.arm_prior_beta)):
                prior = {"type": "beta", "alpha": alpha, "beta": beta}
                exp = expectation(prior)
                var = variance(prior)
                lines.append(
                    f"Your prior belief on the reward of pulling arm {i} is Beta(alpha={alpha:.2f}, beta={beta:.2f}). "
                    f"The expectation of this prior belief distribution is {exp:.2f}. "
                    f"The variance of this prior belief distribution is {var:.2f}."
                )

        elif self.type == "gaussian-gaussian":
            for i, (mu, sigma) in enumerate(zip(self.arm_prior_mu, self.arm_prior_sigma)):
                prior = {"type": "gaussian", "mu": mu, "sigma": sigma}
                exp = expectation(prior)
                var = variance(prior)
                lines.append(
                    f"Your prior belief on the reward of pulling arm {i} is Gaussian(mu={mu:.2f}, sigma={sigma:.2f}). "
                    f"The expectation of this prior belief distribution is {exp:.2f}. "
                    f"The variance of this prior belief distribution is {var:.2f}."
                )

        elif self.type == "poisson-gamma":
            for i, (a, b) in enumerate(zip(self.arm_prior_a, self.arm_prior_b)):
                prior = {"type": "gamma", "a": a, "b": b}
                exp = expectation(prior)
                var = variance(prior)
                lines.append(
                    f"Your prior belief on the reward of pulling arm {i} is Gamma(a={a:.2f}, b={b:.2f}). "
                    f"The expectation of this prior belief distribution is {exp:.2f}. "
                    f"The variance of this prior belief distribution is {var:.2f}."
                )

        else:
            raise ValueError(f"Unknown bandit type: {self.type}")        
        
        self.action_names = [f"arm_{i}" for i in range(self.num_arms)]

        # create action space with dynamic number of arms, starting at 0
        start = 0
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(self.num_arms, start=start)
        self.current_step = 0
        true_mu = self._true_means()
        self._mu_star = float(np.max(true_mu))
        self._mu_bar = float(np.mean(true_mu))

        self.render_cache = "\n".join(lines)

        return self.render_cache

    def get_all_actions(self):
        return [self.ACTION_SPACE.start, self.ACTION_SPACE.start + self.num_arms]

    def compute_step_reward(self, action: int):
        rng: np.random.Generator = self.np_random
        if self.type == "beta-bernoulli":
            dist = {"type": "bernoulli", "p": self.arm_true_p[action]}
        elif self.type == "gaussian-gaussian":
            dist = {"type": "gaussian", "mu": self.arm_true_mu[action], "sigma": self.arm_true_sigma[action]}
        elif self.type == "poisson-gamma":
            dist = {"type": "poisson", "lambda": self.arm_true_lambda[action]}
        return sample(dist, rng)

    def compute_ucb_reward(self):
        if self.type == "beta-bernoulli":
            return ucb_beta_bernoulli(self.arm_prior_alpha, self.arm_prior_beta, self.arm_true_p, self.horizon)
        elif self.type == "gaussian-gaussian":
            return ucb_gaussian_gaussian(self.arm_prior_mu, self.arm_prior_sigma, self.arm_true_mu, self.arm_true_sigma, self.horizon)
        elif self.type == "poisson-gamma":
            return ucb_poisson_gamma(self.arm_prior_a, self.arm_prior_b, self.arm_true_lambda, self.horizon)

    def compute_ids_reward(self):
        if self.type == "beta-bernoulli":
            return ids_beta_bernoulli(self.arm_prior_alpha, self.arm_prior_beta, self.arm_true_p, self.horizon)
        elif self.type == "gaussian-gaussian":
            return ids_gaussian_gaussian(self.arm_prior_mu, self.arm_prior_sigma, self.arm_true_mu, self.arm_true_sigma, self.horizon)
        elif self.type == "poisson-gamma":
            return ids_poisson_gamma(self.arm_prior_a, self.arm_prior_b, self.arm_true_lambda, self.horizon)

    def step(self, action):
        self.current_step += 1
        try:
            action = int(action)
        except:
            action = -1

        if action >= self.num_arms or action < 0:
            reward = -1 # setting to mu_bar would lead to output degeneration; should set it to some quite low value.
            msg_head = "You failed to take a valid action and will be penalized."
        else:
            reward = self.compute_step_reward(action)
            msg_head = f"You pulled arm {action} and received reward {reward:.2f}.\n"

        self.cumulative_reward += reward  
        terminated = self.current_step >= self.horizon 
        if not terminated:
            observation = (
                msg_head + 
                "Choose your next arm to pull."
            )
            info = {"cumulative_reward": self.cumulative_reward, "action_is_valid": action < self.num_arms and action >= 0}
        else:
            antagonist_reward = self.compute_ucb_reward()
            antagonist_reward_ids = self.compute_ids_reward() # just taking a look how large it is
            antagonist_regret = self.horizon * self._mu_star - antagonist_reward
            antagonist_regret_ids = self.horizon * self._mu_star - antagonist_reward_ids
            cumulative_regret = self.horizon * self._mu_star - self.cumulative_reward

            # expected regret of pulling a random arm. Used for normalizing difficulty
            random_regret = self.horizon * (self._mu_star - self._mu_bar)

            # https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf.
            gap = (cumulative_regret - antagonist_regret) / ((max(antagonist_regret, random_regret) - min(antagonist_regret, random_regret)) + 1e-12)

            observation = (
                msg_head + 
                "Game finished. Thank you for playing!"
            )
            info = {
                "protagonist_regret": cumulative_regret,
                "antagonist_regret": antagonist_regret,
                "antagonist_regret_ids": antagonist_regret_ids,
                "random_regret": random_regret,
                "protagonist_antagonist_gap": gap, 
            }
            print("info: ", info) # for testing
        
        self.render_cache = observation
        return observation, reward, terminated, info

    def render(self):
        return self.render_cache

    def close(self):
        self.render_cache = None

    def set_params_for_next_reset(self, p: BanditParams):
        self._next_params_override = p

    def export_params(self) -> Dict[str, Any]:
        return self._as_params().to_dict()