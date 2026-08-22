"""
dgp.py — Data-generating processes for the co-failure e-process simulation.

Two LLM judges i and j score N items. Each item has a latent difficulty theta ~ U(0,1).
Judge error probabilities are difficulty-driven and shared across judges, creating
marginal correlation via shared difficulty but (under H0) NO excess co-failure.
"""

import numpy as np


def judge_probs(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (p_i, p_j) — per-item marginal error probs for judges i and j.
    Both rise with difficulty theta, creating shared-difficulty marginal correlation.
    """
    p_i = np.clip(0.30 + 0.50 * (theta - 0.5), 0.02, 0.98)
    p_j = np.clip(0.35 + 0.50 * (theta - 0.5), 0.02, 0.98)
    return p_i, p_j


def null_dgp(N: int, rng: np.random.Generator) -> dict:
    """
    H0 — conditional independence given difficulty.
    Judges share difficulty but fail independently given theta.
    Returns dict with keys: theta, W_i, W_j, p_i, p_j.
    """
    theta = rng.uniform(0.0, 1.0, size=N)
    p_i, p_j = judge_probs(theta)
    W_i = (rng.uniform(size=N) < p_i).astype(int)
    W_j = (rng.uniform(size=N) < p_j).astype(int)
    return dict(theta=theta, W_i=W_i, W_j=W_j, p_i=p_i, p_j=p_j)


def alt_dgp(N: int, gamma: float, rng: np.random.Generator) -> dict:
    """
    H1 — genuine excess co-failure via a common shock (Marshall-Olkin atom).
    With probability gamma each item is a "shock": both judges fail together
    regardless of difficulty. Otherwise generated as H0.
    gamma in {0.05, 0.10} per the spec.
    Returns dict with keys: theta, W_i, W_j, p_i, p_j, shock (bool mask).
    """
    theta = rng.uniform(0.0, 1.0, size=N)
    p_i, p_j = judge_probs(theta)

    shock = rng.uniform(size=N) < gamma
    W_i = np.where(shock, 1, (rng.uniform(size=N) < p_i).astype(int))
    W_j = np.where(shock, 1, (rng.uniform(size=N) < p_j).astype(int))
    return dict(theta=theta, W_i=W_i, W_j=W_j, p_i=p_i, p_j=p_j, shock=shock)


def drift_null_dgp(N: int, rng: np.random.Generator) -> dict:
    """
    Drift-null variant — marginals ramp linearly over the stream:
      base_i: 0.20 -> 0.45 across N items
      base_j: 0.25 -> 0.50 across N items
    Still H0 (conditional independence given difficulty), but global sample means
    are a biased estimate of the true marginals at any given time.
    Returns dict with keys: theta, W_i, W_j, p_i, p_j.
    """
    theta = rng.uniform(0.0, 1.0, size=N)
    t_frac = np.linspace(0.0, 1.0, N)
    base_i = 0.20 + 0.25 * t_frac   # 0.20 -> 0.45
    base_j = 0.25 + 0.25 * t_frac   # 0.25 -> 0.50
    p_i = np.clip(base_i + 0.50 * (theta - 0.5), 0.02, 0.98)
    p_j = np.clip(base_j + 0.50 * (theta - 0.5), 0.02, 0.98)
    W_i = (rng.uniform(size=N) < p_i).astype(int)
    W_j = (rng.uniform(size=N) < p_j).astype(int)
    return dict(theta=theta, W_i=W_i, W_j=W_j, p_i=p_i, p_j=p_j)
