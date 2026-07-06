from __future__ import annotations

from dataclasses import dataclass
import numpy as np

NoiseMode = str  # "same_input" or "different_input"


@dataclass(frozen=True)
class PrebeckBoltzmannDynamics:
    """
    Boltzmann dynamics for the Prebeck 125-node network (Goal 2).

    Follows Prebeck thesis Eq. 2.1 and 2.3 exactly:

        z_i(t) = b_i(t) + sum_j  s_j(t) * W[j, i]

    where s_j in {0, 1}  — UNIPOLAR (inactive = 0, not -1).
    An inactive neuron contributes ZERO to the input, not -1.

    This differs from BoltzmannDynamics (used in Goal 1) which maps
    s -> 2s-1 (bipolar ±1). Do NOT change BoltzmannDynamics —
    Goal 1 results depend on the bipolar convention.

    Temperature scaling (Prebeck Section 2.8):
        z[mask] /= t0
    applied only to skeleton nodes (mask=True) so that the average
    skeleton weight is effectively normalised to 1.

    Parameters
    ----------
    t0   : float  — temperature (mean skeleton weight), must be > 0
    mask : bool array shape (N,) or None — nodes where scaling is applied
    """

    t0: float
    mask: np.ndarray | None  # boolean mask of length N

    def step(
        self,
        s: np.ndarray,           # (N,)  uint8, values in {0, 1}
        W: np.ndarray,           # (N,N) float, W[j,i] = w_{j->i}
        rng: np.random.Generator,
        sigma: float,
        noise_mode: NoiseMode,
    ) -> np.ndarray:

        # ── input validation ──────────────────────────────────────────────
        if s.ndim != 1:
            raise ValueError("State s must be 1D of shape (N,).")
        if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] != s.shape[0]:
            raise ValueError("W must be (N,N) and match s length.")
        if sigma < 0:
            raise ValueError("sigma must be nonnegative.")
        if noise_mode not in ("same_input", "different_input"):
            raise ValueError("noise_mode must be 'same_input' or 'different_input'.")
        if self.t0 <= 0:
            raise ValueError("t0 must be positive.")
        if self.mask is not None:
            if (self.mask.ndim != 1
                    or self.mask.dtype != bool
                    or self.mask.shape[0] != s.shape[0]):
                raise ValueError("mask must be boolean 1-D array of length N.")

        N = s.shape[0]

        # ── unipolar weighted input  z_i = sum_j s_j * W[j,i] ───────────
        # s is {0,1} — no symmetrisation, inactive neurons contribute 0
        z = s.astype(float) @ W        # shape (N,)

        # ── temperature scaling FIRST (before noise) ──────────────────────
        # Thesis Section 2.8: replace z by z/t0.
        # Noise must be added on the NORMALISED scale so that sigma=1 means
        # "noise comparable to one average weight", matching the thesis x-axis.
        # Wrong order (noise before /t0) would require sigma~t0=4.5 to be
        # meaningful, pushing the SR peak outside the plotted 0-10 range.
        if self.mask is not None:
            z = z.copy()
            z[self.mask] = z[self.mask] / float(self.t0)

        # ── Gaussian noise (Prebeck Eq. 2.3) ─────────────────────────────
        if sigma > 0:
            if noise_mode == "same_input":
                b = rng.normal(loc=0.0, scale=sigma)          # scalar
                z = z + b
            else:                                              # "different_input"
                b = rng.normal(loc=0.0, scale=sigma, size=N)  # per-node
                z = z + b

        # ── stochastic firing (Prebeck Eq. 2.2) ──────────────────────────
        p = 1.0 / (1.0 + np.exp(-z))
        u = rng.random(N)
        return (u < p).astype(np.uint8)
