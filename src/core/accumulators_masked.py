from __future__ import annotations
import numpy as np


class MaskedStateAccumulator:
    """
    Accumulates transition statistics for a masked substate
    (e.g. skeleton nodes) while the full system evolves.
    """

    def __init__(self, mask: np.ndarray):
        if mask.ndim != 1 or mask.dtype != bool:
            raise ValueError("mask must be a 1D boolean array")

        self.mask = mask
        self.N = int(mask.sum())
        if self.N <= 0:
            raise ValueError("Mask selects zero nodes")

        self.num_states = 1 << self.N
        self.state_counts = np.zeros(self.num_states, dtype=np.int64)
        self.trans_counts = np.zeros((self.num_states, self.num_states), dtype=np.int64)
        self.steps = 0

    @staticmethod
    def _encode(bits01: np.ndarray) -> int:
        code = 0
        for i, b in enumerate(bits01.astype(int)):
            code |= (b << i)
        return code

    def update(self, s_t: np.ndarray, s_tp1: np.ndarray) -> None:
        z_t = self._encode(s_t[self.mask])
        z_tp1 = self._encode(s_tp1[self.mask])
        self.state_counts[z_t] += 1
        self.trans_counts[z_t, z_tp1] += 1
        self.steps += 1

    def finalize(self) -> dict[str, np.ndarray]:
        if self.steps == 0:
            raise ValueError("No samples collected")

        p_joint = self.trans_counts / self.trans_counts.sum()
        return {"p_joint": p_joint.astype(float)}
