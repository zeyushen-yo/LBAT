from __future__ import annotations
from typing import Any, Dict
import numpy as np


def expectation(dist: Dict[str, Any]):
    dtype = dist["type"]
    if dtype == "bernoulli":
        return float(dist["p"])
    if dtype == "beta":
        a, b = dist["alpha"], dist["beta"]
        return a / (a + b)
    if dtype == "gaussian":
        return float(dist["mu"])
    if dtype == "gamma":
        a, b = dist["a"], dist["b"]
        return a / b
    if dtype == "poisson":
        return float(dist["lambda"])
    raise ValueError(f"unknown distribution type: {dtype}")


def variance(dist: Dict[str, Any]):
    dtype = dist["type"]
    if dtype == "bernoulli":
        p = dist["p"]
        return (p * (1.0 - p)) ** 0.5
    if dtype == "beta":
        a, b = dist["alpha"], dist["beta"]
        return (a * b / ((a + b) ** 2 * (a + b + 1))) ** 0.5
    if dtype == "gaussian":
        return float(dist["sigma"])
    if dtype == "gamma":
        return (dist["a"] ** 0.5) / dist["b"]
    if dtype == "poisson":
        return dist["lambda"] ** 0.5
    raise ValueError(f"unknown distribution type: {dtype}")


def sample(dist: Dict[str, Any], rng: np.random.Generator):
    dtype = dist["type"]
    if dtype == "bernoulli":
        return float(rng.binomial(1, dist["p"]))
    if dtype == "beta":
        return float(rng.beta(dist["alpha"], dist["beta"]))
    if dtype == "gaussian":
        return float(rng.normal(dist["mu"], dist["sigma"]))
    if dtype == "gamma":
        return float(rng.gamma(shape=dist["a"], scale=1.0 / dist["b"]))
    if dtype == "poisson":
        return float(rng.poisson(dist["lambda"]))
    raise ValueError(f"unknown distribution type: {dtype}")


