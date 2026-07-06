from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from builders.nrooks import nrooks_weight_matrix
from core.dynamics import BoltzmannDynamics  


def bits_of_state(idx: int, N: int) -> np.ndarray:
    return ((idx >> np.arange(N)) & 1).astype(int)


def state_index(bits01: np.ndarray) -> int:
    idx = 0
    for i, b in enumerate(bits01.astype(int)):
        idx |= (int(b) & 1) << i
    return idx


def hamming_distance(a_idx: int, b_idx: int, N: int) -> int:
    return int(bin(a_idx ^ b_idx).count("1"))


def estimate_p_v_given_u_simulation(
    W: np.ndarray,
    N: int,
    u_idx: int,
    q: float,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Estimate p(v|u) by repeatedly sampling one-step transitions from the same u.
    """
    rng = np.random.default_rng(seed)
    dyn = BoltzmannDynamics()

    u_bits = bits_of_state(u_idx, N).astype(float)  # 0/1
    counts = np.zeros(2**N, dtype=np.int64)

    for _ in range(n_samples):
        v_bits = dyn.step(s=u_bits, W=W, sigma=0.0, noise_mode="different_input", rng = rng)
        v_idx = state_index(v_bits)
        counts[v_idx] += 1

    p = counts / counts.sum()
    return p


def exact_p_v_given_u(
    W: np.ndarray,
    N: int,
    u_idx: int,
) -> np.ndarray:
    """
    Compute exact p(v|u) for synchronous Boltzmann updates with sigma=0:
      z = (2u-1) @ W
      p_i = sigmoid(z_i)
      p(v|u) = prod_i p_i^{v_i} (1-p_i)^{1-v_i}
    """
    u_bits = bits_of_state(u_idx, N).astype(float)
    sym = 2.0 * u_bits - 1.0
    z = sym @ W
    p_i = 1.0 / (1.0 + np.exp(-z))  # sigmoid

    S = 2**N
    p = np.zeros(S, dtype=float)
    for v_idx in range(S):
        v_bits = bits_of_state(v_idx, N).astype(float)
        prob = np.prod((p_i ** v_bits) * ((1.0 - p_i) ** (1.0 - v_bits)))
        p[v_idx] = prob
    p /= p.sum()
    return p


def plot_pvu_matrix(P: np.ndarray, outpath: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(P, aspect="equal")
    plt.colorbar(label="p(v | u)")
    plt.xlabel("v (next state index)")
    plt.ylabel("u (current state index)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".png"), dpi=220)
    plt.savefig(outpath.with_suffix(".pdf"))
    plt.close()


def plot_wrong_transition_histogram(P_row: np.ndarray, u_idx: int, N: int, outpath: Path, title: str) -> None:
    """
    For a fixed u, group probabilities by Hamming distance from the most probable v.
    """
    v_star = int(np.argmax(P_row))
    dists = np.array([hamming_distance(v_star, v, N) for v in range(2**N)], dtype=int)

    # sum probs by distance
    max_d = N
    mass = np.zeros(max_d + 1, dtype=float)
    for d in range(max_d + 1):
        mass[d] = P_row[dists == d].sum()

    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(max_d + 1), mass)
    plt.xlabel("Hamming distance d from most probable successor v*")
    plt.ylabel("total probability mass")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".png"), dpi=220)
    plt.savefig(outpath.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    out_dir = Path("data/goal1_pvu")
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 4
    q = 2.0

    # Use an NRooks matrix; if you want a random permutation version, we can swap it in.
    W = nrooks_weight_matrix(N=N, q=q, sign=+1)

    # Compute full exact transition matrix P(u->v) for N=4
    S = 2**N
    P_exact = np.zeros((S, S), dtype=float)
    for u_idx in range(S):
        P_exact[u_idx, :] = exact_p_v_given_u(W=W, N=N, u_idx=u_idx)

    # Plot exact transition matrix
    plot_pvu_matrix(
        P_exact,
        out_dir / "P_exact_N4_q2",
        title="Exact transition probabilities P(v|u), N=4, q=2",
    )

    # Also estimate one representative row by simulation and compare visually
    u_idx = 0  # you can change this (for example pick a random u)
    P_sim_row = estimate_p_v_given_u_simulation(
        W=W,
        N=N,
        u_idx=u_idx,
        q=q,
        n_samples=200000,
        seed=123,
    )

    # Save row vectors
    np.save(out_dir / "P_exact.npy", P_exact)
    np.save(out_dir / f"P_exact_row_u{u_idx}.npy", P_exact[u_idx, :])
    np.save(out_dir / f"P_sim_row_u{u_idx}.npy", P_sim_row)
    np.save(out_dir / "W.npy", W)

    # Plot exact row and simulated row as small 1x16 heatmaps for clarity
    plt.figure(figsize=(10, 1.8))
    plt.imshow(P_exact[u_idx, :][None, :], aspect="auto")
    plt.colorbar(label="p(v | u)")
    plt.xlabel("v (next state index)")
    plt.yticks([])
    plt.title(f"Exact row P(v|u={u_idx}), N=4, q=2")
    plt.tight_layout()
    plt.savefig(out_dir / f"Row_exact_u{u_idx}.png", dpi=220)
    plt.savefig(out_dir / f"Row_exact_u{u_idx}.pdf")
    plt.close()

    plt.figure(figsize=(10, 1.8))
    plt.imshow(P_sim_row[None, :], aspect="auto")
    plt.colorbar(label="p(v | u)")
    plt.xlabel("v (next state index)")
    plt.yticks([])
    plt.title(f"Simulated row P(v|u={u_idx}), N=4, q=2 (200k samples)")
    plt.tight_layout()
    plt.savefig(out_dir / f"Row_sim_u{u_idx}.png", dpi=220)
    plt.savefig(out_dir / f"Row_sim_u{u_idx}.pdf")
    plt.close()

    # Show the "one neuron failure dominates" hypothesis using Hamming-distance mass
    plot_wrong_transition_histogram(
        P_exact[u_idx, :],
        u_idx=u_idx,
        N=N,
        outpath=out_dir / f"Hamming_mass_exact_u{u_idx}",
        title=f"Exact probability mass by Hamming distance, u={u_idx}, N=4, q=2",
    )

    plot_wrong_transition_histogram(
        P_sim_row,
        u_idx=u_idx,
        N=N,
        outpath=out_dir / f"Hamming_mass_sim_u{u_idx}",
        title=f"Simulated probability mass by Hamming distance, u={u_idx}, N=4, q=2",
    )

    print("Weight matrix W used:")
    print(W)
    print("Saved plots and arrays to:", out_dir.resolve())


if __name__ == "__main__":
    main()
