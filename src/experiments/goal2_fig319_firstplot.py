from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.io_paths import ensure_dir
from weights.prebeck_125node import build_prebeck_125node_weight_matrix
from plotting.prebeck_matrix_plot import plot_prebeck_weight_matrix

from core.dynamics_prebeck import PrebeckBoltzmannDynamics
from core.simulate_states import run_recorded_states

from metrics.info import mutual_information_joint


def _motif_codes(states_sk: np.ndarray, triplet: np.ndarray) -> np.ndarray:
    """
    states_sk: (T, ns) uint8
    triplet: (3,) int node indices within skeleton [0..ns-1]
    returns codes: (T,) uint8 in {0..7}
    code = s[a]*1 + s[b]*2 + s[c]*4
    """
    a, b, c = int(triplet[0]), int(triplet[1]), int(triplet[2])
    s = states_sk
    return (s[:, a] * 1 + s[:, b] * 2 + s[:, c] * 4).astype(np.uint8)


def _p_joint_from_codes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    x, y: (L,) int arrays in {0..7}
    returns p_joint: (8, 8) float, normalized
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    idx = (8 * x.astype(np.int64) + y.astype(np.int64))
    counts = np.bincount(idx, minlength=64).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("No samples for p_joint")
    return (counts / total).reshape(8, 8)


def mi_intra_inter_from_states_routeB(states_sk: np.ndarray, motifs3: np.ndarray) -> tuple[float, float]:
    """
    Route B exact method:
      - 3 motifs (motifs3 shape (3,3))
      - intra MI: compute MI per motif (t -> t+1), average over 3
      - inter MI: compute MI per ordered motif pair (i -> j), average over 6

    states_sk: (T, ns) uint8
    returns: (I_intra_avg, I_inter_avg) in bits
    """
    if motifs3.shape != (3, 3):
        raise ValueError("motifs3 must be shape (3,3)")
    T = states_sk.shape[0]
    if T < 2:
        raise ValueError("Need T >= 2")

    # motif codes
    codes = [ _motif_codes(states_sk, motifs3[m]) for m in range(3) ]

    # intra: average MI over 3 motifs
    intra_vals = []
    for m in range(3):
        x = codes[m][:-1]
        y = codes[m][1:]
        p_joint = _p_joint_from_codes(x, y)
        intra_vals.append(mutual_information_joint(p_joint))
    I_intra = float(np.mean(intra_vals))

    # inter: average MI over 6 directed pairs
    inter_vals = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            x = codes[i][:-1]   # time t
            y = codes[j][1:]    # time t+1
            p_joint = _p_joint_from_codes(x, y)
            inter_vals.append(mutual_information_joint(p_joint))
    I_inter = float(np.mean(inter_vals))

    return I_intra, I_inter


def main() -> None:
    out_root = ensure_dir(Path("data") / "goal2" / "fig3_19_like")
    arr_dir = ensure_dir(out_root / "arrays")
    plot_dir = ensure_dir(out_root / "plots")
    mat_dir = ensure_dir(out_root / "matrices")

    rng = np.random.default_rng(0)

    ns = 10
    ds_list = [0.35, 0.45]
    sigmas = np.linspace(0.0, 10.0, 51)

    # Route B: 3 disjoint motifs inside skeleton (node indices within skeleton)
    motifs3 = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ], dtype=int)
    np.save(arr_dir / "motifs3_routeB_ns10.npy", motifs3)

    # Runtime: increase to reduce MI bias
    T_iso = 500_000
    T_emb = 500_000
    burn_in = 50_000

    # Choose ONE noise model and keep it fixed while matching the thesis
    NOISE_MODE = "different_input"  # or "different_input"

    np.save(arr_dir / "sigmas.npy", sigmas)

    for ds in ds_list:
        tag = f"ns{ns}_ds{int(ds*100)}"

        # Build matrices
        W_t0, idx, t0 = build_prebeck_125node_weight_matrix(rng=rng, ns=ns, ds=ds, wi_mode="-t0")
        W_m1, idx2, t0_2 = build_prebeck_125node_weight_matrix(rng=rng, ns=ns, ds=ds, wi_mode="-1")

        sk_idx = idx["skeleton"]
        if sk_idx.size != ns:
            raise ValueError("Skeleton index size mismatch")

        # Save and plot matrix like Fig 2.7 (once per ds is enough)
        np.save(arr_dir / f"W_full_{tag}_wi_t0.npy", W_t0)
        np.save(arr_dir / f"W_full_{tag}_wi_m1.npy", W_m1)

        plot_prebeck_weight_matrix(
            W_t0, ns=ns,
            title=f"Prebeck W ({tag}, wi=-t0)",
            out_dir=mat_dir,
            fname_stem=f"W_{tag}_wi_t0"
        )

        # Isolated skeleton dynamics: run on 10x10 core, scale on all nodes
        W_core = W_t0[np.ix_(sk_idx, sk_idx)]
        mask_core = np.ones(ns, dtype=bool)
        dyn_core = PrebeckBoltzmannDynamics(t0=float(t0), mask=mask_core)

        I_intra_iso = np.zeros(sigmas.size, dtype=float)
        I_inter_iso = np.zeros(sigmas.size, dtype=float)

        for k, sigma in enumerate(sigmas):
            states = run_recorded_states(
                W=W_core,
                dynamics=dyn_core,
                T=T_iso,
                burn_in=burn_in,
                sigma=float(sigma),
                noise_mode=NOISE_MODE,
                seed=1000 + k,
                record_idx=np.arange(ns, dtype=int),
            )
            I_intra_iso[k], I_inter_iso[k] = mi_intra_inter_from_states_routeB(states, motifs3)

        np.save(arr_dir / f"I_intra_iso_{tag}.npy", I_intra_iso)
        np.save(arr_dir / f"I_inter_iso_{tag}.npy", I_inter_iso)

        # Embedded runs: dynamics on full 125 nodes, scale only on skeleton nodes
        mask_full = np.zeros(W_t0.shape[0], dtype=bool)
        mask_full[sk_idx] = True

        dyn_full_t0 = PrebeckBoltzmannDynamics(t0=float(t0), mask=mask_full)
        dyn_full_m1 = PrebeckBoltzmannDynamics(t0=float(t0_2), mask=mask_full)

        states_emb_t0 = run_recorded_states(
            W=W_t0,
            dynamics=dyn_full_t0,
            T=T_emb,
            burn_in=burn_in,
            sigma=0.0,
            noise_mode=NOISE_MODE,
            seed=2001,
            record_idx=sk_idx,
        )
        states_emb_m1 = run_recorded_states(
            W=W_m1,
            dynamics=dyn_full_m1,
            T=T_emb,
            burn_in=burn_in,
            sigma=0.0,
            noise_mode=NOISE_MODE,
            seed=2002,
            record_idx=sk_idx,
        )

        I_intra_emb_t0, I_inter_emb_t0 = mi_intra_inter_from_states_routeB(states_emb_t0, motifs3)
        I_intra_emb_m1, I_inter_emb_m1 = mi_intra_inter_from_states_routeB(states_emb_m1, motifs3)

        np.save(arr_dir / f"I_intra_emb_{tag}_wi_t0.npy", np.array([I_intra_emb_t0], dtype=float))
        np.save(arr_dir / f"I_inter_emb_{tag}_wi_t0.npy", np.array([I_inter_emb_t0], dtype=float))
        np.save(arr_dir / f"I_intra_emb_{tag}_wi_m1.npy", np.array([I_intra_emb_m1], dtype=float))
        np.save(arr_dir / f"I_inter_emb_{tag}_wi_m1.npy", np.array([I_inter_emb_m1], dtype=float))

        # Plot intra
        plt.figure(figsize=(7.5, 5.0))
        plt.plot(sigmas, I_intra_iso, linewidth=2.5, label="isolated intra I(sigma)")
        plt.axhline(I_intra_emb_t0, linestyle="--", linewidth=2.0, label="embedded intra (wi=-t0), sigma=0")
        plt.axhline(I_intra_emb_m1, linestyle=":", linewidth=2.0, label="embedded intra (wi=-1), sigma=0")
        plt.xlabel("noise sigma")
        plt.ylabel("mutual information I [bits]")
        plt.title(f"Goal 2 Fig 3.19 intra-submotif ({tag})")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(plot_dir / f"fig3_19_intra_{tag}.png", dpi=220)
        plt.savefig(plot_dir / f"fig3_19_intra_{tag}.pdf")
        plt.close()

        # Plot inter
        plt.figure(figsize=(7.5, 5.0))
        plt.plot(sigmas, I_inter_iso, linewidth=2.5, label="isolated inter I(sigma)")
        plt.axhline(I_inter_emb_t0, linestyle="--", linewidth=2.0, label="embedded inter (wi=-t0), sigma=0")
        plt.axhline(I_inter_emb_m1, linestyle=":", linewidth=2.0, label="embedded inter (wi=-1), sigma=0")
        plt.xlabel("noise sigma")
        plt.ylabel("mutual information I [bits]")
        plt.title(f"Goal 2 Fig 3.19 inter-submotif ({tag})")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(plot_dir / f"fig3_19_inter_{tag}.png", dpi=220)
        plt.savefig(plot_dir / f"fig3_19_inter_{tag}.pdf")
        plt.close()

        print(f"[Goal2] done {tag}")

    print(f"All outputs under: {out_root.resolve()}")


if __name__ == "__main__":
    main()
