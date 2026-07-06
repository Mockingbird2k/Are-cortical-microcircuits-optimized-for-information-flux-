from __future__ import annotations

import numpy as np


def nrooks_weight_matrix(N: int, q: float, sign: int = +1) -> np.ndarray:
    """
    Perfect N-rooks weight matrix: exactly one nonzero per row and one per column.
    We implement a single directed cycle:
        neuron j -> neuron (j+1 mod N)
    with magnitude q (and optional sign).

    Convention matches your core: W[j, i] = w_{j,i} (j presynaptic to i postsynaptic).
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if q < 0:
        raise ValueError("q must be nonnegative.")
    if sign not in (-1, +1):
        raise ValueError("sign must be +1 or -1.")

    W = np.zeros((N, N), dtype=float)
    for j in range(N):
        i = (j + 1) % N
        W[j, i] = float(sign) * float(q)
    return W
