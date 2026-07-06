from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.accumulators import StateAccumulator, encode_state_bits
from core.dynamics import BoltzmannDynamics, NoiseMode


@dataclass(frozen=True)
class SimulationConfig:
    T: int
    sigma: float
    noise_mode: NoiseMode
    seed: int
    init: str = "random"  # "random" or "zeros"

    def validate(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.sigma < 0:
            raise ValueError("sigma must be nonnegative.")
        if self.noise_mode not in ("same_input", "different_input"):
            raise ValueError("Invalid noise_mode.")
        if self.init not in ("random", "zeros"):
            raise ValueError("Invalid init policy.")


@dataclass
class SimulationResult:
    pz: np.ndarray
    p_cond: np.ndarray
    p_joint: np.ndarray
    steps: int


class Simulator:
    def __init__(self, dynamics: BoltzmannDynamics | None = None) -> None:
        self.dynamics = dynamics if dynamics is not None else BoltzmannDynamics()

    def run_state_counts(self, W: np.ndarray, cfg: SimulationConfig) -> SimulationResult:
        cfg.validate()
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be square (N,N).")
        N = W.shape[0]

        rng = np.random.default_rng(cfg.seed)

        if cfg.init == "zeros":
            s = np.zeros(N, dtype=np.uint8)
        else:
            s = rng.integers(0, 2, size=N, dtype=np.uint8)

        acc = StateAccumulator(N=N)

        for _ in range(cfg.T):
            z_t = encode_state_bits(s)
            s_next = self.dynamics.step(s=s, W=W, rng=rng, sigma=cfg.sigma, noise_mode=cfg.noise_mode)
            z_tp1 = encode_state_bits(s_next)
            acc.update(z_t, z_tp1)
            s = s_next

        stats = acc.finalize()
        return SimulationResult(
            pz=stats["pz"],
            p_cond=stats["p_cond"],
            p_joint=stats["p_joint"],
            steps=acc.steps,
        )
