from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from builders.nrooks import nrooks_weight_matrix
from core.simulate import Simulator, SimulationConfig
from metrics.info import entropy_base2, mutual_information_joint
from metrics.nrooks_mi import analytic_mi_nrooks


def run_simulated_n4(q_grid: np.ndarray, T: int, burn_in: int, seed: int):
    sim = Simulator()
    I_sim = []
    H_sim = []

    for q in q_grid:
        W = nrooks_weight_matrix(N=4, q=float(q), sign=+1)

        # Run a burn-in first (do not record), then record counts for T steps
        cfg_burn = SimulationConfig(T=burn_in, sigma=0.0, noise_mode="different_input", seed=seed, init="random")
        _ = sim.run_state_counts(W=W, cfg=cfg_burn)

        # Re-seed for reproducibility of the recorded phase (or keep advancing RNG by using a different seed)
        cfg = SimulationConfig(T=T, sigma=0.0, noise_mode="different_input", seed=seed + 1, init="random")
        res = sim.run_state_counts(W=W, cfg=cfg)

        I_sim.append(mutual_information_joint(res.p_joint))
        H_sim.append(entropy_base2(res.pz))

    return np.asarray(I_sim), np.asarray(H_sim)


def main() -> None:
    out_dir = Path("data/goal1_prof_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    q_grid = np.linspace(0.0, 10.0, 51)

    # Based on your professor’s note: large q needs long runs to explore attractors.
    T = 200_000
    burn_in = 10_000
    seed = 123

    # Print one example NRooks matrix used (q=5)
    W_example = nrooks_weight_matrix(N=4, q=5.0, sign=+1)
    print("Example 4x4 NRooks weight matrix used (N=4, q=5):")
    print(W_example)
    print("Row nonzero counts:", (W_example != 0).sum(axis=1))
    print("Col nonzero counts:", (W_example != 0).sum(axis=0))

    # Analytical and simulated
    I_ana = analytic_mi_nrooks(q_grid, N=4)
    I_sim, H_sim = run_simulated_n4(q_grid=q_grid, T=T, burn_in=burn_in, seed=seed)

    # Save arrays
    np.save(out_dir / "q_grid.npy", q_grid)
    np.save(out_dir / "I_analytic_N4.npy", I_ana)
    np.save(out_dir / "I_sim_N4.npy", I_sim)
    np.save(out_dir / "H_sim_N4.npy", H_sim)
    np.save(out_dir / "W_example_q5.npy", W_example)

    # Plot: MI curves + entropy on second axis
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(q_grid, I_ana, linewidth=2.5, label="Analytical (N=4)")
    ax1.plot(q_grid, I_sim, linewidth=2.5, label="Simulation (N=4)")
    ax1.set_xlabel("coupling magnitude q")
    ax1.set_ylabel("MI between successive states I (bits)")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 4.2)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(q_grid, H_sim, linewidth=2.0, linestyle="--", label="Entropy H (sim, N=4)")
    ax2.set_ylabel("State entropy H (bits)")
    ax2.set_ylim(0, 4.2)

    # One combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(out_dir / "goal1_prof_debug_N4_MI_and_H.png", dpi=200)
    plt.savefig(out_dir / "goal1_prof_debug_N4_MI_and_H.pdf")
    plt.show()

    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
