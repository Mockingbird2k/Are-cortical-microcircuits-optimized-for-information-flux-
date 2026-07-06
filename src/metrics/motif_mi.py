from __future__ import annotations
import numpy as np


def all_triplets(ns: int) -> np.ndarray:
    """
    All 3-node subsets of {0..ns-1}.
    Returns shape (M, 3), M = C(ns, 3).
    """
    out = []
    for a in range(ns):
        for b in range(a + 1, ns):
            for c in range(b + 1, ns):
                out.append((a, b, c))
    return np.asarray(out, dtype=int)


def encode_motif_codes(states01: np.ndarray, triplets: np.ndarray) -> np.ndarray:
    """
    states01: (T, ns) uint8
    triplets: (M, 3) int

    Returns:
      codes: (M, T) uint8 in {0..7}
      code = s[a]*1 + s[b]*2 + s[c]*4
    """
    if states01.ndim != 2:
        raise ValueError("states01 must be (T, ns)")
    if triplets.ndim != 2 or triplets.shape[1] != 3:
        raise ValueError("triplets must be (M, 3)")

    a = triplets[:, 0]
    b = triplets[:, 1]
    c = triplets[:, 2]

    # states01[:, a] gives (T, M), transpose to (M, T)
    codes = (
        states01[:, a].T.astype(np.uint8) * 1
        + states01[:, b].T.astype(np.uint8) * 2
        + states01[:, c].T.astype(np.uint8) * 4
    )
    return codes


def pooled_joint_intra(codes: np.ndarray) -> np.ndarray:
    """
    Pooled joint counts for intra-motif transitions:
      X = motif code at t
      Y = same motif code at t+1
    pooled over all motifs and time.

    codes: (M, T)
    Returns p_joint counts as (8, 8) float (not normalized).
    """
    M, T = codes.shape
    if T < 2:
        raise ValueError("Need T >= 2")

    x = codes[:, :-1].reshape(-1).astype(np.int64)
    y = codes[:, 1:].reshape(-1).astype(np.int64)

    idx = 8 * x + y
    counts = np.bincount(idx, minlength=64).astype(np.float64)
    return counts.reshape(8, 8)


def disjoint_pairs(triplets: np.ndarray) -> np.ndarray:
    """
    All ordered pairs (i, j) of motif indices with disjoint node sets.
    Returns shape (P, 2).
    """
    M = triplets.shape[0]
    sets = [set(triplets[i].tolist()) for i in range(M)]

    pairs = []
    for i in range(M):
        si = sets[i]
        for j in range(M):
            if i == j:
                continue
            if si.isdisjoint(sets[j]):
                pairs.append((i, j))

    return np.asarray(pairs, dtype=int)


def pooled_joint_inter(codes: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """
    Pooled joint counts for inter-motif transitions:
      X = motif i code at t
      Y = motif j code at t+1
    pooled over all disjoint motif pairs and time.

    codes: (M, T)
    pairs: (P, 2)
    Returns p_joint counts as (8, 8) float (not normalized).
    """
    M, T = codes.shape
    if T < 2:
        raise ValueError("Need T >= 2")
    if pairs.size == 0:
        raise ValueError("No disjoint motif pairs")

    i = pairs[:, 0]
    j = pairs[:, 1]

    x = codes[i, :-1].reshape(-1).astype(np.int64)
    y = codes[j, 1:].reshape(-1).astype(np.int64)

    idx = 8 * x + y
    counts = np.bincount(idx, minlength=64).astype(np.float64)
    return counts.reshape(8, 8)
