from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from builders.nrooks import nrooks_weight_matrix
from core.simulate import Simulator, SimulationConfig
from metrics.nrooks_mi import (
    analytic_mi_nrooks,
    transition_matrix_nrooks,
    stationary_distribution,
    mutual_information_successive_states,
)


def mi_by_simulation(W: np.ndarray, T: int, seed: int) -> float:
    sim = Simulator()
    cfg = SimulationConfig(T=T, sigma=0.0, noise_mode="different_input", seed=seed, init="random")
    res = sim.run_state_counts(W=W, cfg=cfg)

    # MI from empirical joint over successive states:
    # reuse your existing MI function to avoid duplication
    from metrics.info import mutual_information_joint
    return mutual_information_joint(res.p_joint)


def main() -> None:
    out_dir = Path("data/goal1_nrooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    q_grid = np.linspace(0.0, 10.0, 51)

    Ns = [4, 10]

    # Simulation lengths: adjust if you want tighter convergence
    T_sim = {4: 100_000, 10: 800_000}
    seed_sim = 123

    results = {}

    for N in Ns:
        # Analytical
        I_analytic = analytic_mi_nrooks(q_grid, N)

        # Transition matrix
        I_tm = []
        for q in q_grid:
            W = nrooks_weight_matrix(N=N, q=float(q), sign=+1)
            P = transition_matrix_nrooks(W)
            pi = stationary_distribution(P)
            I_tm.append(mutual_information_successive_states(P, pi))
        I_tm = np.asarray(I_tm, dtype=float)

        # Simulation
        I_sim = []
        for q in q_grid:
            W = nrooks_weight_matrix(N=N, q=float(q), sign=+1)
            I_sim.append(mi_by_simulation(W, T=T_sim[N], seed=seed_sim))
        I_sim = np.asarray(I_sim, dtype=float)

        results[N] = {"analytic": I_analytic, "tm": I_tm, "sim": I_sim}

        # Save numeric artifacts
        np.save(out_dir / f"q_grid.npy", q_grid)
        np.save(out_dir / f"I_analytic_N{N}.npy", I_analytic)
        np.save(out_dir / f"I_tm_N{N}.npy", I_tm)
        np.save(out_dir / f"I_sim_N{N}.npy", I_sim)

    # Plot: color encodes method, line style encodes N
    colors = {"analytic": None, "tm": None, "sim": None}  # default matplotlib colors
    linestyles = {4: "-", 10: "--"}

    plt.figure(figsize=(9, 5))

    # Analytical
    for N in Ns:
        plt.plot(
            q_grid,
            results[N]["analytic"],
            linestyle=linestyles[N],
            linewidth=2.5,
            label=f"analytical (N={N})",
        )

    # Transition matrix

    for N in Ns:
        plt.plot(
            q_grid,
            results[N]["tm"],
            linestyle=linestyles[N],
            linewidth=2.5,
            label=f"transition matrix (N={N})",
        )

    # Simulation
    for N in Ns:
        plt.plot(
            q_grid,
            results[N]["sim"],
            linestyle=linestyles[N],
            linewidth=2.5,
            label=f"simulation (N={N})",
        )

    plt.xlabel("magnitude q of non-zero matrix elements")
    plt.ylabel("mutual information I between successive states (bits)")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", ncols=2)
    plt.tight_layout()

    plt.savefig(out_dir / "goal1_MI_vs_q_NRooks_N4_N10.png", dpi=200)
    plt.savefig(out_dir / "goal1_MI_vs_q_NRooks_N4_N10.pdf")
    plt.show()

    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
