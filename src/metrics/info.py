from __future__ import annotations

import numpy as np


def entropy_base2(p: np.ndarray, eps: float = 0.0) -> float:
    p = np.asarray(p, dtype=float)
    if np.any(p < 0):
        raise ValueError("Probabilities must be nonnegative.")
    s = p.sum()
    if s <= 0:
        raise ValueError("Sum of probabilities must be positive.")
    p = p / s

    if eps > 0:
        p = np.clip(p, eps, 1.0)
        p = p / p.sum()

    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def mutual_information_joint(p_joint: np.ndarray, eps: float = 0.0) -> float:
    P = np.asarray(p_joint, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("p_joint must be square.")
    if np.any(P < 0):
        raise ValueError("p_joint must be nonnegative.")
    s = P.sum()
    if s <= 0:
        raise ValueError("p_joint sum must be positive.")
    P = P / s

    if eps > 0:
        P = np.clip(P, eps, 1.0)
        P = P / P.sum()

    p_i = P.sum(axis=1, keepdims=True)
    p_j = P.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = p_i @ p_j
        ratio = np.divide(P, denom, out=np.ones_like(P), where=(P > 0) & (denom > 0))
        term = np.where(P > 0, P * np.log2(ratio), 0.0)

    mi = float(np.sum(term))
    if mi < -1e-12:
        raise ValueError(f"MI computed negative: {mi}")
    return max(0.0, mi)
