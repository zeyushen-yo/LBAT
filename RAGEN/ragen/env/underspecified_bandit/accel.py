from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import math
import copy
import bisect
import random

@dataclass
class BanditParams:
    type: str
    num_arms: int
    horizon: int
    arm_prior_alpha: Optional[List[float]] = None
    arm_prior_beta: Optional[List[float]] = None
    arm_true_p: Optional[List[float]] = None
    arm_prior_mu: Optional[List[float]] = None
    arm_prior_sigma: Optional[List[float]] = None
    arm_true_mu: Optional[List[float]] = None
    arm_true_sigma: Optional[List[float]] = None
    arm_prior_a: Optional[List[float]] = None
    arm_prior_b: Optional[List[float]] = None
    arm_true_lambda: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def canonical_key(self, decimals: int = 6) -> Tuple:
        def r(v):
            return None if v is None else tuple(round(float(x), decimals) for x in v)
        return (
            self.type,
            int(self.num_arms),
            int(self.horizon),
            r(self.arm_prior_alpha), r(self.arm_prior_beta), r(self.arm_true_p),
            r(self.arm_prior_mu), r(self.arm_prior_sigma), r(self.arm_true_mu), r(self.arm_true_sigma),
            r(self.arm_prior_a), r(self.arm_prior_b), r(self.arm_true_lambda),
        )

class AccelBuffer:
    def __init__(self, capacity: int, temperature: float = 0.5, seed: int = 0):
        self.capacity = capacity
        self.temperature = max(1e-6, float(temperature))
        self.items: List[Tuple[float, BanditParams]] = []
        self.index: dict[Tuple, int] = {}
        self.rng = np.random.default_rng(seed)

    def __len__(self): return len(self.items)

    def min_score(self) -> float:
        return min((s for s,_ in self.items), default=-math.inf)

    def add(self, score: float, params: BanditParams) -> bool:
        key = params.canonical_key()
        s = float(score)

        if key in self.index:
            i = self.index[key]
            self.items[i] = (s, copy.deepcopy(params))
            return True

        if len(self.items) < self.capacity:
            self.index[key] = len(self.items)
            self.items.append((s, copy.deepcopy(params)))
            return True

        worst_idx = min(range(len(self.items)), key=lambda j: self.items[j][0])
        if s <= self.items[worst_idx][0]:
            return False

        evicted_key = self.items[worst_idx][1].canonical_key()
        self.index.pop(evicted_key, None)
        self.items[worst_idx] = (s, copy.deepcopy(params))
        self.index[key] = worst_idx
        return True

    def sample(self, k: int = 1) -> List[BanditParams]:
        assert self.items, "Buffer empty"
        scores = np.array([s for s,_ in self.items], dtype=float)
        logits = scores / self.temperature
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()
        idx = self.rng.choice(len(self.items), size=k, p=p, replace=(k>1))
        return [copy.deepcopy(self.items[i][1]) for i in np.atleast_1d(idx)]


def _clamp_list(xs: List[float], lo: float, hi: float) -> List[float]:
    return [float(np.clip(x, lo, hi)) for x in xs]


def perturb_params(params: BanditParams,
                   rng: np.random.Generator,
                   sigma_small: float = 0.05,
                   sigma_big: float = 0.15,
                   cfg: Optional[object] = None) -> BanditParams:
    q = replace(params)
    sigma = rng.uniform(sigma_small, sigma_big)

    if q.type == "beta-bernoulli":
        q.arm_prior_alpha = [a + rng.normal(0, sigma) for a in q.arm_prior_alpha]
        q.arm_prior_beta = [b + rng.normal(0, sigma) for b in q.arm_prior_beta]
        q.arm_true_p = [p_ + rng.normal(0, sigma) for p_ in q.arm_true_p]

    elif q.type == "gaussian-gaussian":
        q.arm_prior_mu = [m + rng.normal(0, sigma) for m in q.arm_prior_mu]
        q.arm_true_mu = [m + rng.normal(0, sigma) for m in q.arm_true_mu]
        q.arm_prior_sigma = [s + rng.normal(0, sigma) for s in q.arm_prior_sigma]
        q.arm_true_sigma = [s + rng.normal(0, sigma) for s in q.arm_true_sigma]

    elif q.type == "poisson-gamma":
        q.arm_prior_a = [a + rng.normal(0, sigma) for a in q.arm_prior_a]
        q.arm_prior_b = [b + rng.normal(0, sigma) for b in q.arm_prior_b]
        q.arm_true_lambda = [l + rng.normal(0, sigma) for l in q.arm_true_lambda]

    if cfg is not None:
        if q.type == "beta-bernoulli":
            q.arm_prior_alpha = _clamp_list(q.arm_prior_alpha, cfg.min_alpha, cfg.max_alpha)
            q.arm_prior_beta = _clamp_list(q.arm_prior_beta, cfg.min_beta, cfg.max_beta)
            q.arm_true_p = _clamp_list(q.arm_true_p, cfg.min_p, cfg.max_p)

        elif q.type == "gaussian-gaussian":
            q.arm_prior_mu = _clamp_list(q.arm_prior_mu, cfg.min_mu_prior, cfg.max_mu_prior)
            q.arm_true_mu = _clamp_list(q.arm_true_mu, cfg.min_mu_true, cfg.max_mu_true)
            q.arm_prior_sigma = _clamp_list(q.arm_prior_sigma, cfg.min_sigma_prior, cfg.max_sigma_prior)
            q.arm_true_sigma = _clamp_list(q.arm_true_sigma, cfg.min_sigma_true, cfg.max_sigma_true)

        elif q.type == "poisson-gamma":
            q.arm_prior_a = _clamp_list(q.arm_prior_a, cfg.min_a, cfg.max_a)
            q.arm_prior_b = _clamp_list(q.arm_prior_b, cfg.min_b, cfg.max_b)
            q.arm_true_lambda = _clamp_list(q.arm_true_lambda, cfg.min_lambda, cfg.max_lambda)

    return q