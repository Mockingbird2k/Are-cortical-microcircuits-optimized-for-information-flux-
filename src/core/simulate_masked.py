from __future__ import annotations
import numpy as np

from core.dynamics_prebeck import PrebeckBoltzmannDynamics
from core.accumulators_masked import MaskedStateAccumulator


def run_masked_state_counts(
    W: np.ndarray,
    mask: np.ndarray,
    T: int,
    sigma: float,
    noise_mode: str,
    seed: int,
    t0: float,
    init: str = "random",
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dyn = PrebeckBoltzmannDynamics(t0 = float(t0), mask = mask)

    N = W.shape[0]
    if init == "random":
        s = rng.integers(0, 2, size=N, dtype=np.uint8)
    else:
        raise ValueError("Only init='random' supported")

    acc = MaskedStateAccumulator(mask)

    for _ in range(T):
        s_next = dyn.step(s=s, W=W, rng=rng, sigma=sigma, noise_mode=noise_mode)
        acc.update(s, s_next)
        s = s_next

    return acc.finalize()
