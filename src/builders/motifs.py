from __future__ import annotations

import numpy as np


def motif_W_from_edges(N: int, edges: list[tuple[int, int, float]]) -> np.ndarray:
    """
    Build W (N,N) from directed edges (j, i, w_ji).
    Convention: W[j, i] = w_ji.
    """
    W = np.zeros((N, N), dtype=float)
    for j, i, w in edges:
        if not (0 <= i < N and 0 <= j < N):
            raise ValueError("Edge index out of bounds.")
        W[j, i] = float(w)
    return W
