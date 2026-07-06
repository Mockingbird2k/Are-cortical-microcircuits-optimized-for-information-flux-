from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def h2(p: np.ndarray) -> np.ndarray:
    """
    Binary entropy h2(p) in bits.
    """
    p = np.asarray(p, dtype=float)
    eps = 1e-300
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def I_new_independent_failures(q: np.ndarray, N: int) -> np.ndarray:
    """
    New analytic MI for NRooks permutation Boltzmann dynamics with symmetrized inputs:
      eps = sigmoid(-q)
      I_new = N * (1 - h2(eps))
    """
    q = np.asarray(q, dtype=float)
    eps = sigmoid(-q)
    return N * (1.0 - h2(eps))


def find_required_files(folder: Path) -> tuple[Path, Path, Path]:
    """
    Expect these files inside folder:
      q_grid.npy
      I_sim_N4.npy
      I_analytic_N4.npy
    """
    q_path = folder / "q_grid.npy"
    sim_path = folder / "I_sim_N4.npy"
    ana_path = folder / "I_analytic_N4.npy"

    missing = [p for p in [q_path, sim_path, ana_path] if not p.exists()]
    if missing:
        msg = "Missing required files:\n" + "\n".join(str(m) for m in missing)
        raise FileNotFoundError(msg)

    return q_path, sim_path, ana_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Folder containing q_grid.npy, I_sim_N4.npy, I_analytic_N4.npy",
    )
    ap.add_argument(
        "--out_name",
        type=str,
        default="MI_N4_sim_vs_old_vs_new",
        help="Output filename stem (no extension).",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    q_path, sim_path, ana_path = find_required_files(in_dir)

    q_grid = np.load(q_path)
    I_sim = np.load(sim_path)
    I_old = np.load(ana_path)

    if q_grid.ndim != 1:
        raise ValueError("q_grid.npy must be a 1D array.")
    if I_sim.shape != q_grid.shape:
        raise ValueError(f"I_sim_N4.npy shape {I_sim.shape} does not match q_grid shape {q_grid.shape}.")
    if I_old.shape != q_grid.shape:
        raise ValueError(f"I_analytic_N4.npy shape {I_old.shape} does not match q_grid shape {q_grid.shape}.")

    N = 4
    I_new = I_new_independent_failures(q_grid, N=N)

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(q_grid, I_sim, linewidth=2.5, label="Simulation (N=4)")
    plt.plot(q_grid, I_old, linewidth=2.5, label="Old analytical (uniform wrong transitions)")
    plt.plot(q_grid, I_new, linewidth=2.5, linestyle="--", label="New analytical (independent failures)")

    plt.xlabel("coupling magnitude q")
    plt.ylabel("Mutual information I(X_t ; X_{t+1}) [bits]")
    plt.xlim(float(q_grid.min()), float(q_grid.max()))
    plt.ylim(0.0, 4.2)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    out_png = in_dir / f"{args.out_name}.png"
    out_pdf = in_dir / f"{args.out_name}.pdf"
    plt.savefig(out_png, dpi=220)
    plt.savefig(out_pdf)
    plt.show()

    print("Saved:")
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
