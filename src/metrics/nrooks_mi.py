from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _h2(p: np.ndarray) -> np.ndarray:
    """
    Binary entropy h2(p) in bits:
      h2(p) = -p log2 p - (1-p) log2(1-p)
    with 0*log(0)=0 handled via clipping.
    """
    p = np.asarray(p, dtype=float)
    eps = 1e-300
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def analytic_mi_nrooks(q: np.ndarray, N: int) -> np.ndarray:
    """
    NEW analytical MI for NRooks permutation Boltzmann dynamics (symmetrized inputs),
    under the independent-bit failure model.

    Let eps = sigma(-q) = 1/(1+exp(q)) be the per-neuron failure probability.
    Then:
      I_new(N,q) = N * (1 - h2(eps))   [bits]

    This matches the observed structure of P(v|u): wrong transitions are dominated
    by low Hamming-distance errors rather than being uniform across all wrong states.
    """
    q = np.asarray(q, dtype=float)
    if N <= 0:
        raise ValueError("N must be positive.")

    eps = _sigmoid(-q)  # sigma(-q) = 1/(1+exp(q))
    return float(N) * (1.0 - _h2(eps))


def analytic_mi_nrooks_old(q: np.ndarray, N: int) -> np.ndarray:
    """
    OLD analytical MI from the uniform-wrong-transition approximation:

      preg = sigma(q)
      p1 = preg^N
      p0 = (1-p1)/(2^N-1)
      I_old = p1 log2(2^N p1) + (2^N-1) p0 log2(2^N p0)

    Kept for reference and comparison plots.
    """
    q = np.asarray(q, dtype=float)
    if N <= 0:
        raise ValueError("N must be positive.")

    NS = 2 ** N
    preg = _sigmoid(q)
    p1 = preg ** N
    p0 = (1.0 - p1) / (NS - 1.0)

    eps = 1e-300
    p1c = np.clip(p1, eps, 1.0)
    p0c = np.clip(p0, eps, 1.0)

    term1 = np.where(p1 > 0, p1 * np.log2(NS * p1c), 0.0)
    term0 = np.where(p0 > 0, p0 * np.log2(NS * p0c), 0.0)

    return term1 + (NS - 1.0) * term0
