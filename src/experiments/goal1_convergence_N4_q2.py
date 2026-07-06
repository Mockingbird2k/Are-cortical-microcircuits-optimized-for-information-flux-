from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from core.simulate import Simulator, SimulationConfig
from metrics.info import mutual_information_joint, entropy_base2
from metrics.nrooks_mi import analytic_mi_nrooks


def nrooks_matrix_fixed_perm(N: int, q: float, perm: np.ndarray) -> np.ndarray:
    W = np.zeros((N, N), dtype=float)
    rows = np.arange(N)
    W[rows, perm] = float(q)
    return W


def estimate_mi_for_T(W: np.ndarray, T: int, burn_in: int, seed: int) -> tuple[float, float]:
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


def main() -> None:
    out_dir = Path("data/goal1_convergence")
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 4
    q = 2.0

    # Increasing time steps
    T_list = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
    NR = 30  # number of repeats per T, increase if you want tighter statistics

    burn_in = 2_000
    seed_base = 123

    # Fix one random NRooks permutation so the only variable is T, not W
    rng = np.random.default_rng(seed_base)
    perm = rng.permutation(N)
    W = nrooks_matrix_fixed_perm(N=N, q=q, perm=perm)

    print("Fixed 4x4 NRooks matrix used for convergence test (N=4, q=2):")
    print(W)
    print("Row nonzero counts:", (W != 0).sum(axis=1))
    print("Col nonzero counts:", (W != 0).sum(axis=0))

    I_ana = float(analytic_mi_nrooks(np.array([q]), N=N)[0])
    print(f"Analytical MI at N=4, q=2: {I_ana:.6f} bits")

    # res arrays
    res_I = np.zeros((len(T_list), NR), dtype=float)
    res_H = np.zeros((len(T_list), NR), dtype=float)

    for i, T in enumerate(T_list):
        for r in range(NR):
            seed = seed_base + 10_000 * i + r
            mi, H = estimate_mi_for_T(W=W, T=T, burn_in=burn_in, seed=seed)
            res_I[i, r] = mi
            res_H[i, r] = H

        mean_I = res_I[i].mean()
        print(f"T={T:>7d}  mean MI={mean_I:.6f}  mean H={res_H[i].mean():.6f}")

    # Save arrays
    np.save(out_dir / "T_list.npy", np.array(T_list, dtype=int))
    np.save(out_dir / "res_I.npy", res_I)
    np.save(out_dir / "res_H.npy", res_H)
    np.save(out_dir / "W.npy", W)
    with open(out_dir / "analytic_MI.txt", "w", encoding="utf-8") as f:
        f.write(f"{I_ana}\n")

    # Plot 1: MI versus T (individual points + mean), analytic line
    T_arr = np.array(T_list, dtype=float)
    mean_I = res_I.mean(axis=1)
    std_I = res_I.std(axis=1, ddof=1)
    se_I = std_I / np.sqrt(NR)

    plt.figure(figsize=(9, 5))

    # Scatter all runs
    for i, T in enumerate(T_list):
        plt.scatter([T] * NR, res_I[i], s=12)

    # Mean with error bars (standard error)
    plt.errorbar(T_arr, mean_I, yerr=se_I, fmt="o-", linewidth=2.0, capsize=4, label="Simulation mean ± SE")

    # Analytic horizontal line
    plt.axhline(I_ana, linewidth=2.0, linestyle="--", label="Analytical MI")

    plt.xscale("log")
    plt.xlabel("number of time steps T (log scale)")
    plt.ylabel("MI between successive states I (bits)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "MI_convergence_N4_q2.png", dpi=200)
    plt.savefig(out_dir / "MI_convergence_N4_q2.pdf")
    plt.close()

    # Plot 2: Entropy versus T (mean)
    mean_H = res_H.mean(axis=1)
    std_H = res_H.std(axis=1, ddof=1)
    se_H = std_H / np.sqrt(NR)

    plt.figure(figsize=(9, 5))
    plt.errorbar(T_arr, mean_H, yerr=se_H, fmt="o-", linewidth=2.0, capsize=4, label="Entropy mean ± SE")
    plt.axhline(N, linewidth=2.0, linestyle="--", label="Ideal H = N bits")
    plt.xscale("log")
    plt.xlabel("number of time steps T (log scale)")
    plt.ylabel("state entropy H (bits)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "Entropy_convergence_N4_q2.png", dpi=200)
    plt.savefig(out_dir / "Entropy_convergence_N4_q2.pdf")
    plt.close()

    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
