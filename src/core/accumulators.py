from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def encode_state_bits(s: np.ndarray) -> int:
    """
    Encode binary vector s (length N, values 0/1) into integer in [0, 2^N - 1].
    Bit convention: s[0] is least significant bit.
    """
    if s.ndim != 1:
        raise ValueError("s must be 1D.")
    if not np.all((s == 0) | (s == 1)):
        raise ValueError("s must be binary (0/1).")

    code = 0
    for i, bit in enumerate(s.astype(int)):
        code |= (bit << i)
    return code


@dataclass
class StateAccumulator:
    """
    Counts:
      - state counts: c[i] for i in 0..(2^N-1)
      - transition counts: C[i,j] for successive states
    """
    N: int

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError("N must be positive.")
        self.num_states = 1 << self.N
        self.state_counts = np.zeros(self.num_states, dtype=np.int64)
        self.trans_counts = np.zeros((self.num_states, self.num_states), dtype=np.int64)
        self.steps = 0

    def update(self, z_t: int, z_tp1: int) -> None:
        self.state_counts[z_t] += 1
        self.trans_counts[z_t, z_tp1] += 1
        self.steps += 1

    def finalize(self) -> dict[str, np.ndarray]:
        if self.steps == 0:
            raise ValueError("No samples collected.")
        pz = self.state_counts / self.state_counts.sum()

        row_sums = self.trans_counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_cond = np.divide(
                self.trans_counts,
                row_sums,
                out=np.zeros_like(self.trans_counts, dtype=float),
                where=row_sums > 0,
            )

        p_joint = self.trans_counts / self.trans_counts.sum()
        return {"pz": pz.astype(float), "p_cond": p_cond.astype(float), "p_joint": p_joint.astype(float)}
