from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from core.simulate import Simulator, SimulationConfig
from metrics.info import mutual_information_joint, entropy_base2
from metrics.nrooks_mi import analytic_mi_nrooks


def random_nrooks_matrix(N: int, q: float, rng: np.random.Generator) -> np.ndarray:
    """
    Create an NRooks weight matrix W with exactly one nonzero per row and per column.
    Construction: choose a random permutation perm of {0..N-1} and set:
      W[row, perm[row]] = q
    All other entries are 0.
    """
    perm = rng.permutation(N)
    W = np.zeros((N, N), dtype=float)
    rows = np.arange(N)
    W[rows, perm] = float(q)
    return W


def simulate_mi_entropy_for_W(
    W: np.ndarray,
    T: int,
    burn_in: int,
    seed: int,
) -> tuple[float, float]:
    """
    Run Boltzmann network simulation (sigma=0) and return:
      MI between successive global states (bits),
      entropy of visited global states H (bits).
    """
    sim = Simulator()

    if burn_in > 0:
        cfg_burn = SimulationConfig(
            T=burn_in,
            sigma=0.0,
            noise_mode="different_input",
            seed=seed,
            init="random",
        )
        _ = sim.run_state_counts(W=W, cfg=cfg_burn)

    cfg = SimulationConfig(
        T=T,
        sigma=0.0,
        noise_mode="different_input",
        seed=seed + 1,
        init="random",
    )
    res = sim.run_state_counts(W=W, cfg=cfg)

    mi = mutual_information_joint(res.p_joint)
    H = entropy_base2(res.pz)
    return mi, H


def run_NR_experiments(
    N: int,
    q_grid: np.ndarray,
    NR: int,
    T: int,
    burn_in: int,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      res_I shape (len(q_grid), NR)
      res_H shape (len(q_grid), NR)
    """
    res_I = np.zeros((len(q_grid), NR), dtype=float)
    res_H = np.zeros((len(q_grid), NR), dtype=float)

    # For each run r, choose one random NRooks structure (permutation),
    # then scale by q for each q in the sweep.
    for r in range(NR):
        rng = np.random.default_rng(seed_base + 10_000 * N + r)

        # Fix one permutation per run
        perm = rng.permutation(N)

        for v, q in enumerate(q_grid):
            W = np.zeros((N, N), dtype=float)
            rows = np.arange(N)
            W[rows, perm] = float(q)

            mi, H = simulate_mi_entropy_for_W(W=W, T=T, burn_in=burn_in, seed=seed_base + 1000 * r + v)
            res_I[v, r] = mi
            res_H[v, r] = H

    return res_I, res_H


def plot_MI_curves(q_grid: np.ndarray, res_I: np.ndarray, I_ana: np.ndarray | None, N: int, out_dir: Path) -> None:
    """
    Plot all NR individual MI curves in light gray and the mean in a normal color.
    Optionally include analytic curve (dashed).
    """
    mean_I = res_I.mean(axis=1)

    plt.figure(figsize=(9, 5))

    # Individual runs in light gray
    for r in range(res_I.shape[1]):
        plt.plot(q_grid, res_I[:, r], color="0.80", linewidth=1.0)

    # Mean curve on top
    plt.plot(q_grid, mean_I, linewidth=2.5, label=f"Simulation mean (N={N}, NR={res_I.shape[1]})")

    # Analytical curve (optional)
    if I_ana is not None:
        plt.plot(q_grid, I_ana, linestyle="--", linewidth=2.5, label=f"Analytical (N={N})")

    plt.xlabel("coupling magnitude q")
    plt.ylabel("mutual information I between successive states (bits)")
    plt.xlim(float(q_grid.min()), float(q_grid.max()))
    plt.ylim(0, float(N) + 0.2)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(out_dir / f"MI_NR{res_I.shape[1]}_N{N}.png", dpi=200)
    plt.savefig(out_dir / f"MI_NR{res_I.shape[1]}_N{N}.pdf")
    plt.close()


def plot_entropy_curves(q_grid: np.ndarray, res_H: np.ndarray, N: int, out_dir: Path) -> None:
    """
    Plot entropy curves in a separate figure, as requested.
    """
    mean_H = res_H.mean(axis=1)

    plt.figure(figsize=(9, 5))

    for r in range(res_H.shape[1]):
        plt.plot(q_grid, res_H[:, r], color="0.80", linewidth=1.0)

    plt.plot(q_grid, mean_H, linewidth=2.5, label=f"Entropy mean (N={N}, NR={res_H.shape[1]})")

    plt.xlabel("coupling magnitude q")
    plt.ylabel("state entropy H (bits)")
    plt.xlim(float(q_grid.min()), float(q_grid.max()))
    plt.ylim(0, float(N) + 0.2)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(out_dir / f"Entropy_NR{res_H.shape[1]}_N{N}.png", dpi=200)
    plt.savefig(out_dir / f"Entropy_NR{res_H.shape[1]}_N{N}.pdf")
    plt.close()


def main() -> None:
    out_dir = Path("data/goal1_publication")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    NR = 10
    q_grid = np.linspace(0.0, 10.0, 51)
    seed_base = 123

    # Simulation lengths: increase with N to keep mixing reasonable.
    # You can tune these if runtime is too high.
    T_map = {2: 100_000, 4: 200_000, 6: 400_000, 8: 800_000, 10: 1_200_000}
    burn_in_map = {2:1_000, 4: 10_000, 6: 20_000, 8: 30_000, 10: 40_000}

    Ns = [4, 6, 8, 10]
    # Just NS = 10 for 1 run
    Ns = [2, 8, 10]

    for N in Ns:
        T = T_map[N]
        burn_in = burn_in_map[N]

        print(f"\nRunning NRooks publication runs for N={N} with NR={NR}, T={T}, burn_in={burn_in}")

        # Run simulations across NR random matrices
        res_I, res_H = run_NR_experiments(
            N=N,
            q_grid=q_grid,
            NR=NR,
            T=T,
            burn_in=burn_in,
            seed_base=seed_base,
        )

        # Analytical curve (available for all N)
        I_ana = analytic_mi_nrooks(q_grid, N=N)

        # Save arrays
        np.save(out_dir / f"q_grid.npy", q_grid)
        np.save(out_dir / f"res_I_N{N}_NR{NR}.npy", res_I)
        np.save(out_dir / f"res_H_N{N}_NR{NR}.npy", res_H)
        np.save(out_dir / f"I_analytic_N{N}.npy", I_ana)

        # Plots
        plot_MI_curves(q_grid=q_grid, res_I=res_I, I_ana=I_ana, N=N, out_dir=out_dir)
        plot_entropy_curves(q_grid=q_grid, res_H=res_H, N=N, out_dir=out_dir)

        # Print one example weight matrix for N=4 at q=5, as your professor asked earlier
        if N == 4:
            rng = np.random.default_rng(seed_base + 10_000 * N + 0)
            W_ex = random_nrooks_matrix(N=4, q=5.0, rng=rng)
            print("\nExample 4x4 NRooks matrix used (random, q=5):")
            print(W_ex)
            print("Row nonzero counts:", (W_ex != 0).sum(axis=1))
            print("Col nonzero counts:", (W_ex != 0).sum(axis=0))

    print(f"\nAll outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
