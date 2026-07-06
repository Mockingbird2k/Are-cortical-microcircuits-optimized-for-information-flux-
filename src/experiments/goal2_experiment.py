"""
goal2_experiment.py
===================
Reproduces Prof's Goal 2 (all four sub-goals):

  (1) Prints a concise feature list of the Prebeck weight matrix (Fig 3.19).
  (2) build_prebeck_125node_weight_matrix() — imported from weights module.
  (3) plot_weight_matrix()  — Fig 2.7 style heatmap.
  (4) plot_fig319()         — Fig 3.19 style 2x2 MI vs sigma plot:
        rows = ds=35% (top) and ds=45% (bottom)
        cols = intra-sub-motif MI (left) and inter-sub-motif MI (right)
        blue solid = isolated skeleton + Gaussian noise sweep (sigma 0..10)
        red dashed = embedded in full 125-node network, wi=-t0, no noise
        orange dot = embedded in full 125-node network, wi=-1,  no noise

Run from project root:
    python src/experiments/goal2_experiment.py

Outputs:
    data/goal2/goal2_out/W_matrix_ns10_ds35_mt0.png
    data/goal2/goal2_out/fig319_ns10.png
    data/goal2/goal2_out/arrays/   (numpy arrays for each run)

===========================================================================
KEY METHODOLOGY NOTES (Prebeck thesis, Sections 2.6, 2.7, 2.8, 3.6)
===========================================================================

Eq. 2.1/2.3:  z_i = b_i + sum_j  s_j * w_ji   (UNIPOLAR: s in {0,1})
Eq. 2.2:      P(s_i=1) = sigmoid(z_i / t0)     (t0 on skeleton nodes only)

Sub-motif selection (Sections 2.6, 2.8):
  THREE fixed disjoint 3-node triplets are chosen from the skeleton.
  For ns=10: (0,1,2), (3,4,5), (6,7,8).  NOT all C(10,3)=120 triplets.
  Using all 120 causes memory explosion: 4200 pairs x 10^6 steps = 34 GB RAM.

MI formula (Eq. 2.9):
  MIz = sum_{i,j=1..8} p(i,j) * log2(p(i,j) / (p(i)*p(j)))
  Averaged over 3 intra-motifs and 6 ordered inter-pairs.

Timesteps: 10^6 per simulation (Section 2.8).
"""

from __future__ import annotations
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE.parent, _HERE]:
    if (_candidate / "weights").is_dir() or (_candidate / "core").is_dir():
        sys.path.insert(0, str(_candidate))
        break

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weights.prebeck_125node  import build_prebeck_125node_weight_matrix
from core.dynamics_prebeck    import PrebeckBoltzmannDynamics
from core.simulate_states     import run_recorded_states
from metrics.info             import mutual_information_joint
from metrics.motif_mi         import encode_motif_codes, pooled_joint_intra, pooled_joint_inter
from utils.io_paths           import ensure_dir

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

NS      = 10
DS_LIST = [0.35, 0.45]   # both rows of Fig 3.19
SEED    = 0

T       = 1_000_000   # timesteps per simulation (thesis Section 2.8)
BURN_IN = 10_000

# sigma grid: thesis uses steps of 0.1 for smooth curves.
# Use 0.5 here for speed; change to 0.1 for publication quality.
SIGMAS = np.arange(0.0, 10.1, 0.2)

OUT_DIR   = Path("data/goal2/goal2_out")
ARRAY_DIR = OUT_DIR / "arrays"

# Fixed sub-motif triplets (Section 2.6, 2.8): 3 disjoint triplets from ns=10
TRIPLETS_NS10 = np.array([[0, 1, 2],
                           [3, 4, 5],
                           [6, 7, 8]], dtype=int)

def get_triplets(ns: int) -> np.ndarray:
    if ns == 10:
        return TRIPLETS_NS10
    raise ValueError(f"Add fixed triplets for ns={ns} above.")

# ══════════════════════════════════════════════════════════════════════════════
# (1) FEATURE LIST
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_LIST = """
+------------------------------------------------------------------------------+
|     FEATURES OF THE PREBECK 125-NODE WEIGHT MATRIX  (Fig 3.19 / 2.7)        |
+------------------------------------------------------------------------------+
|  Node layout (125 nodes, no self-connections):                               |
|    0..ns-1       Skeleton  excitatory, densely + strongly connected          |
|    ns..99        Sea       excitatory, sparse + weakly connected             |
|    100..124      Inter     inhibitory, 25 nodes                              |
|                                                                              |
|  Dynamics (Eq. 2.1-2.3, unipolar s in {0,1}):                               |
|    z_i = b_i + sum_j s_j * w_ji   (inactive neuron contributes 0, not -1)   |
|    P(s_i=1) = sigmoid(z_i / t0)   (t0 normalises skeleton inputs)           |
|                                                                              |
|  Region A [skeleton->skeleton]:                                              |
|    density ds (varied), TOP ds*ns*(ns-1) lognormal weights assigned here    |
|    t0 = mean(region A weights)  -> skeleton average weight = 1 after /t0   |
|                                                                              |
|  Regions B+C [all excitatory->sea/inhibitory]:                              |
|    overall A+B+C density = 11.6% -> ~1438 edges total                       |
|    remaining lognormal weights placed randomly in B+C                       |
|                                                                              |
|  Region D [inhibitory->skeleton+sea]:                                       |
|    density 11.6%, weight wi = -t0 (balanced) or -1 (fixed)                 |
|                                                                              |
|  Region E [inhibitory->inhibitory]:  NO connections (all zero)              |
|                                                                              |
|  Lognormal fit (Song et al. 2005): ln(w) ~ N(mu=-0.702, sigma=0.9355)       |
+------------------------------------------------------------------------------+
"""

# ══════════════════════════════════════════════════════════════════════════════
# MI HELPER
# ══════════════════════════════════════════════════════════════════════════════

def compute_mi_intra_inter(states: np.ndarray,
                           triplets: np.ndarray) -> tuple[float, float]:
    """
    Compute averaged intra- and inter-sub-motif MI.

    Parameters
    ----------
    states   : (T, ns) uint8 -- recorded skeleton states
    triplets : (K, 3) int   -- K fixed disjoint sub-motif triplets

    Returns
    -------
    mi_intra : mean intra-motif MI over K triplets (bits)
    mi_inter : mean inter-motif MI over K*(K-1) ordered pairs (bits)
    """
    K     = triplets.shape[0]
    codes = encode_motif_codes(states, triplets)   # (K, T)

    # Intra: one MI per triplet, then average
    intra_vals = []
    for k in range(K):
        c_k  = codes[k:k+1, :]                    # (1, T) -- pooled_joint_intra expects (M,T)
        jnt  = pooled_joint_intra(c_k)
        intra_vals.append(mutual_information_joint(jnt))
    mi_intra = float(np.mean(intra_vals))

    # Inter: all ordered pairs (i->j, i!=j); all disjoint by construction
    inter_vals = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            pair = np.array([[i, j]])              # (1, 2)
            jnt  = pooled_joint_inter(codes, pair)
            inter_vals.append(mutual_information_joint(jnt))
    mi_inter = float(np.mean(inter_vals)) if inter_vals else float("nan")

    return mi_intra, mi_inter

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def sim_isolated_skeleton(W_full: np.ndarray, idx: dict, t0: float,
                          triplets: np.ndarray, sigmas: np.ndarray,
                          T: int, burn_in: int, seed: int,
                          label: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Isolated skeleton sub-network with noise sweep."""
    sk    = idx["skeleton"]
    ns    = len(sk)
    W_sk  = W_full[np.ix_(sk, sk)]
    mask  = np.ones(ns, dtype=bool)
    dyn   = PrebeckBoltzmannDynamics(t0=t0, mask=mask)
    rec   = np.arange(ns)

    mi_intra = np.zeros(len(sigmas))
    mi_inter = np.zeros(len(sigmas))

    for k, sigma in enumerate(sigmas):
        st = run_recorded_states(W=W_sk, dynamics=dyn, T=T, burn_in=burn_in,
                                 sigma=float(sigma), noise_mode="different_input",
                                 seed=seed + k, record_idx=rec)
        mi_intra[k], mi_inter[k] = compute_mi_intra_inter(st, triplets)
        if k % 5 == 0 or k == len(sigmas) - 1:
            print(f"    {label}sigma={sigma:.1f} "
                  f"intra={mi_intra[k]:.5f}  inter={mi_inter[k]:.5f}")

    return mi_intra, mi_inter


def sim_embedded_network(W_full: np.ndarray, idx: dict, t0: float,
                         triplets: np.ndarray,
                         T: int, burn_in: int, seed: int,
                         label: str = "") -> tuple[float, float]:
    """Full 125-node network, sigma=0, record skeleton states only."""
    sk       = idx["skeleton"]
    nt       = W_full.shape[0]
    mask     = np.zeros(nt, dtype=bool)
    mask[sk] = True
    dyn      = PrebeckBoltzmannDynamics(t0=t0, mask=mask)
    rec      = sk.astype(int)

    st = run_recorded_states(W=W_full, dynamics=dyn, T=T, burn_in=burn_in,
                             sigma=0.0, noise_mode="different_input",
                             seed=seed, record_idx=rec)
    mi_intra, mi_inter = compute_mi_intra_inter(st, triplets)
    print(f"    {label}intra={mi_intra:.5f}  inter={mi_inter:.5f}")
    return mi_intra, mi_inter

# ══════════════════════════════════════════════════════════════════════════════
# (3) WEIGHT MATRIX PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_weight_matrix(W: np.ndarray, ns: int, t0: float, ds: float,
                       wi_mode: str, out_dir: Path) -> None:
    ensure_dir(out_dir)
    ne, nt = 100, 125
    vmax = float(np.percentile(np.abs(W[W != 0]), 95)) if np.any(W != 0) else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             gridspec_kw={"width_ratios": [1, 2.5]})

    ax = axes[0]
    im = ax.imshow(W[:ns, :ns], aspect="equal", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(f"Region A (skeleton->skeleton)\nns={ns}, ds={ds:.0%}", fontsize=10)
    ax.set_xlabel("target j"); ax.set_ylabel("source i")
    plt.colorbar(im, ax=ax, label="weight")

    ax = axes[1]
    im = ax.imshow(W, aspect="equal", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(f"Full 125-node matrix  ns={ns}, ds={ds:.0%}, wi={wi_mode}, t0={t0:.2f}",
                 fontsize=10)
    ax.set_xlabel("target j"); ax.set_ylabel("source i")

    for b in [ns - 0.5, ne - 0.5]:
        ax.axvline(b, color="white", linewidth=1.2, linestyle="--")
        ax.axhline(b, color="white", linewidth=1.2, linestyle="--")
    ax.axvline(ne - 0.5, color="white", linewidth=2.5)
    ax.axhline(ne - 0.5, color="white", linewidth=2.5)
    for (col, row, lbl) in [(ns/2, ns/2, "A"), ((ns+ne)/2, (ns+ne)/2, "B"),
                             ((ns+ne)/2, (ne+nt)/2, "C"), ((ne+nt)/2, (ns+ne)/2, "D"),
                             ((ne+nt)/2, (ne+nt)/2, "E")]:
        ax.text(col, row, lbl, color="white", fontsize=12,
                ha="center", va="center", fontweight="bold")
    plt.colorbar(im, ax=ax, label="weight")
    plt.tight_layout()

    stem = f"W_matrix_ns{ns}_ds{int(ds*100):02d}_{wi_mode.replace('-','m')}"
    fig.savefig(out_dir / f"{stem}.png", dpi=200)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  [saved] {out_dir/stem}.png")

# ══════════════════════════════════════════════════════════════════════════════
# (4) FIG 3.19 PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig319(results: dict, sigmas: np.ndarray, ns: int, out_dir: Path) -> None:
    """
    2x2 grid matching Fig 3.19:
      rows: ds=35% (top), ds=45% (bottom)
      cols: intra-MI (left), inter-MI (right)
      3 lines per panel: blue solid (RI), red dashed (wi=-t0), orange dotted (wi=-1)
    """
    ensure_dir(out_dir)
    ds_list = sorted(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Fig 3.19 — ns={ns}, isolated skeleton (RI) vs embedded network",
                 fontsize=13, fontweight="bold")

    row_titles = [f"ds = {ds:.0%}" for ds in ds_list]
    col_titles = ["Intra-sub-motif MI (bits)", "Inter-sub-motif MI (bits)"]

    for row_i, ds in enumerate(ds_list):
        r = results[ds]
        for col_i, (mi_noise, mi_mt0, mi_m1) in enumerate([
            (r["mi_intra_noise"], r["mi_intra_mt0"], r["mi_intra_m1"]),
            (r["mi_inter_noise"], r["mi_inter_mt0"], r["mi_inter_m1"]),
        ]):
            ax = axes[row_i, col_i]

            ax.plot(sigmas, mi_noise, color="steelblue", linewidth=2,
                    label="Isolated skeleton + random input")
            ax.axhline(mi_mt0, color="crimson", linewidth=2, linestyle="--",
                       label=f"Embedded, wi=-t0  (t0={r['t0']:.2f})")
            ax.axhline(mi_m1,  color="darkorange", linewidth=2, linestyle=":",
                       label="Embedded, wi=-1")

            ax.set_title(f"{row_titles[row_i]}, {col_titles[col_i]}", fontsize=10)
            ax.set_xlabel("Noise SD (sigma)", fontsize=10)
            ax.set_ylabel(col_titles[col_i], fontsize=9)
            ax.set_xlim(sigmas[0], sigmas[-1])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / f"fig319_ns{ns}.png", dpi=200)
    fig.savefig(out_dir / f"fig319_ns{ns}.pdf")
    plt.close(fig)
    print(f"  [saved] {out_dir}/fig319_ns{ns}.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(ARRAY_DIR)

    print(FEATURE_LIST)

    triplets = get_triplets(NS)
    results  = {}

    for ds in DS_LIST:
        print(f"\n{'='*65}")
        print(f"  Running ds = {ds:.0%}")
        print(f"{'='*65}")

        # Build wi=-t0 network
        rng       = np.random.default_rng(SEED)
        W_mt0, idx, t0 = build_prebeck_125node_weight_matrix(
            rng=rng, ns=NS, ds=ds, wi_mode="-t0")
        print(f"  W (wi=-t0): t0={t0:.4f}, nonzero={np.count_nonzero(W_mt0)}")

        # Build wi=-1 network (same excitatory weights, same seed)
        rng2 = np.random.default_rng(SEED)
        W_m1, _, _ = build_prebeck_125node_weight_matrix(
            rng=rng2, ns=NS, ds=ds, wi_mode="-1")

        np.save(ARRAY_DIR / f"W_mt0_ns{NS}_ds{int(ds*100):02d}.npy", W_mt0)
        np.save(ARRAY_DIR / f"W_m1_ns{NS}_ds{int(ds*100):02d}.npy",  W_m1)

        # Weight matrix plot (once, for first ds)
        if ds == DS_LIST[0]:
            print("\n[Step 3] Weight matrix plot …")
            plot_weight_matrix(W_mt0, ns=NS, t0=t0, ds=ds, wi_mode="-t0", out_dir=OUT_DIR)

        # Isolated skeleton noise sweep
        print(f"\n[Step 4a] Noise sweep, ds={ds:.0%} …")
        mi_intra_noise, mi_inter_noise = sim_isolated_skeleton(
            W_full=W_mt0, idx=idx, t0=t0, triplets=triplets,
            sigmas=SIGMAS, T=T, burn_in=BURN_IN, seed=SEED,
            label=f"ds={ds:.0%} ")

        np.save(ARRAY_DIR / f"sigmas.npy",                              SIGMAS)
        np.save(ARRAY_DIR / f"mi_intra_noise_ns{NS}_ds{int(ds*100)}.npy", mi_intra_noise)
        np.save(ARRAY_DIR / f"mi_inter_noise_ns{NS}_ds{int(ds*100)}.npy", mi_inter_noise)

        # Embedded wi=-t0
        print(f"\n[Step 4b] Embedded wi=-t0, ds={ds:.0%} …")
        mi_intra_mt0, mi_inter_mt0 = sim_embedded_network(
            W_full=W_mt0, idx=idx, t0=t0, triplets=triplets,
            T=T, burn_in=BURN_IN, seed=SEED + 1000, label="wi=-t0 ")

        # Embedded wi=-1
        print(f"\n[Step 4c] Embedded wi=-1, ds={ds:.0%} …")
        mi_intra_m1, mi_inter_m1 = sim_embedded_network(
            W_full=W_m1, idx=idx, t0=t0, triplets=triplets,
            T=T, burn_in=BURN_IN, seed=SEED + 2000, label="wi=-1 ")

        results[ds] = dict(
            mi_intra_noise=mi_intra_noise, mi_inter_noise=mi_inter_noise,
            mi_intra_mt0=mi_intra_mt0,    mi_inter_mt0=mi_inter_mt0,
            mi_intra_m1=mi_intra_m1,      mi_inter_m1=mi_inter_m1,
            t0=t0, ds=ds,
        )

    # Plot Fig 3.19
    print("\n[Step 4d] Plotting Fig 3.19 …")
    plot_fig319(results, SIGMAS, NS, OUT_DIR)

    print(f"\n Done. Outputs in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
