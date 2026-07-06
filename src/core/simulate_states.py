from __future__ import annotations
import numpy as np


def run_recorded_states(
    W: np.ndarray,
    dynamics,
    T: int,
    burn_in: int,
    sigma: float,
    noise_mode: str,
    seed: int,
    record_idx: np.ndarray,
    init: str = "random",
) -> np.ndarray:
    """
    Run dynamics on the full network and record the substate given by record_idx.

    Returns:
      states_rec: (T, K) uint8 where K=len(record_idx)
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]

    if init == "random":
        s = rng.integers(0, 2, size=N, dtype=np.uint8)
    else:
        raise ValueError("Only init='random' supported")

    for _ in range(burn_in):
        s = dynamics.step(s=s, W=W, rng=rng, sigma=float(sigma), noise_mode=noise_mode)

    K = int(record_idx.size)
    out = np.zeros((T, K), dtype=np.uint8)
    for t in range(T):
        s = dynamics.step(s=s, W=W, rng=rng, sigma=float(sigma), noise_mode=noise_mode)
        out[t] = s[record_idx]

    return out
