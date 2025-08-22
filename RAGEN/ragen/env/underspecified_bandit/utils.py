from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from openai import AzureOpenAI
import os


def expectation(dist: Dict[str, Any]):
    dtype = dist["type"]
    if dtype == "bernoulli":
        return float(dist["p"])

    if dtype == "beta":
        alpha, beta = dist["alpha"], dist["beta"]
        return alpha / (alpha + beta)

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


def call_princeton_sandbox(prompt: str, model: str = "gpt-4o", temperature: float = 0.7):
    sandbox_api_key = os.environ['AI_SANDBOX_KEY']
    sandbox_endpoint = "https://api-ai-sandbox.princeton.edu/"
    sandbox_api_version = "2025-03-01-preview"    
    client = AzureOpenAI(
        api_key=sandbox_api_key,
        azure_endpoint = sandbox_endpoint,
        api_version=sandbox_api_version,
    )
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content