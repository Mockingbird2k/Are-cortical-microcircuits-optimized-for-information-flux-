from __future__ import annotations

from dataclasses import dataclass
import numpy as np

NoiseMode = str  # "same_input" or "different_input"


@dataclass(frozen=True)
class BoltzmannDynamics:
    """
    Binary stochastic update:
      z_i(t) = b_i(t) + sum_j s_j(t) * W[j,i]
      P(s_i(t+1)=1) = sigmoid(z_i(t))
    with Gaussian input noise b_i(t) ~ Normal(0, sigma^2).
    """

    def step(
        self,
        s: np.ndarray,          # (N,), values 0/1
        W: np.ndarray,          # (N,N), W[j,i] = w_{j,i}
        rng: np.random.Generator,
        sigma: float,
        noise_mode: NoiseMode,
    ) -> np.ndarray:
        if s.ndim != 1:
            raise ValueError("State s must be 1D of shape (N,).")
        if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] != s.shape[0]:
            raise ValueError("W must be (N,N) and match s length.")
        if sigma < 0:
            raise ValueError("sigma must be nonnegative.")
        if noise_mode not in ("same_input", "different_input"):
            raise ValueError("noise_mode must be 'same_input' or 'different_input'.")

        N = s.shape[0]
        # symmetrize input bits: 0/1 -> -1/+1
        sym = 2.0 * s.astype(float) - 1.0
        z = sym @ W

        if sigma > 0:
            if noise_mode == "same_input":
                b = rng.normal(loc=0.0, scale=sigma)
                z = z + b
            else:
                b = rng.normal(loc=0.0, scale=sigma, size=N)
                z = z + b

        p = 1.0 / (1.0 + np.exp(-z))
        u = rng.random(N)
        return (u < p).astype(np.uint8)
