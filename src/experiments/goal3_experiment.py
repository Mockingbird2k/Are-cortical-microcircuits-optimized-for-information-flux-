"""
Goal 3: Plot sub-motif MI of the embedded 125-node network as a function
of the inhibitory weight wi, sweeping from -2*t0 to 0 in 11 steps.

Reproduces the experiment described in Prebeck thesis Section 3.6.2,
asking whether wi = -t0 is truly the optimal inhibitory strength.

Usage (from project root):
    python src/experiments/goal3_experiment.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from weights.prebeck_125node import build_prebeck_125node_weight_matrix
from core.dynamics_prebeck   import PrebeckBoltzmannDynamics
from core.simulate_states    import run_recorded_states
from metrics.info            import mutual_information_joint
from metrics.motif_mi        import encode_motif_codes, pooled_joint_intra, pooled_joint_inter
from utils.io_paths          import ensure_dir

# ── Configuration ────────────────────────────────────────────────────────────
NS         = 10
DS_LIST    = [0.35, 0.45]
T          = 1_000_000       # timesteps per simulation
BURN_IN    = 10_000        # burn-in steps discarded
SEED       = 42
N_STEPS    = 21            # number of wi values between -2*t0 and 0
TRIPLETS   = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
OUT_DIR    = Path('data/goal3')
ensure_dir(OUT_DIR)

# ── Helper functions ──────────────────────────────────────────────────────────
def mi_from_states(states: np.ndarray, triplets: np.ndarray):
    """Return (intra_MI, inter_MI) in units of 10^-3 bits."""
    K     = triplets.shape[0]
    codes = encode_motif_codes(states, triplets)
    intra = np.mean([
        mutual_information_joint(pooled_joint_intra(codes[k:k+1, :]))
        for k in range(K)
    ])
    inter = np.mean([
        mutual_information_joint(pooled_joint_inter(codes, np.array([[i, j]])))
        for i in range(K) for j in range(K) if i != j
    ])
    return float(intra) * 1000, float(inter) * 1000


def run_embedded(W: np.ndarray, t0: float, seed: int = SEED):
    """Run the full 125-node network at sigma=0, return (intra_MI, inter_MI)."""
    nt  = W.shape[0]
    dyn = PrebeckBoltzmannDynamics(t0=t0, mask=np.ones(nt, dtype=bool))
    st  = run_recorded_states(
        W=W, dynamics=dyn, T=T, burn_in=BURN_IN,
        sigma=0.0, noise_mode='different_input',
        seed=seed, record_idx=np.arange(NS)
    )
    return mi_from_states(st, TRIPLETS)


# ── Main experiment ───────────────────────────────────────────────────────────
results = {}

for ds in DS_LIST:
    print(f"\n=== ds = {ds:.0%} ===")

    rng = np.random.default_rng(SEED)
    W_base, idx, t0 = build_prebeck_125node_weight_matrix(
        rng=rng, ns=NS, ds=ds, wi_mode='-t0'
    )
    sk  = idx['skeleton']
    sea = idx['sea']
    inh = idx['inhibitory']
    print(f"  t0 = {t0:.4f}")

    # wi sweep: -2*t0 to 0
    wi_fracs = np.linspace(-2.0, 0.0, N_STEPS)   # multiples of t0
    wi_vals  = wi_fracs * t0
    intra_list, inter_list = [], []

    for wi in wi_vals:
        W = W_base.copy()
        # Replace every inhibitory->excitatory weight with the new wi value
        exc = list(sk) + list(sea)
        for inh_node in inh:
            for exc_node in exc:
                if W_base[inh_node, exc_node] != 0:
                    W[inh_node, exc_node] = wi

        intra, inter = run_embedded(W, t0)
        intra_list.append(intra)
        inter_list.append(inter)
        print(f"  wi = {wi:7.3f}  ({wi / t0:+.2f}*t0) "
              f"| intra = {intra:7.3f}  inter = {inter:7.3f}  (x10-3 bits)")

    results[ds] = dict(
        wi_fracs=wi_fracs,
        wi_vals=wi_vals,
        intra=np.array(intra_list),
        inter=np.array(inter_list),
        t0=t0
    )

    # Save arrays
    np.save(OUT_DIR / f'goal3_ds{int(ds*100):02d}_intra.npy', np.array(intra_list))
    np.save(OUT_DIR / f'goal3_ds{int(ds*100):02d}_inter.npy', np.array(inter_list))
    np.save(OUT_DIR / f'goal3_ds{int(ds*100):02d}_wi_fracs.npy', wi_fracs)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    'Goal 3 — Intra & Inter Sub-motif MI vs Inhibitory Weight wᵢ\n'
    '(embedded 125-node network, σ = 0, ns = 10)',
    fontsize=13, fontweight='bold'
)

colors = {0.35: '#2166ac', 0.45: '#d6604d'}

for ds, r in results.items():
    for ax_i, (mi_vals, mi_label) in enumerate([
        (r['intra'], 'Intra-sub-motif MI'),
        (r['inter'], 'Inter-sub-motif MI')
    ]):
        ax = axes[ax_i]
        ax.plot(r['wi_fracs'], mi_vals, 'o-', color=colors[ds],
                linewidth=2.2, markersize=6, label=f'ds = {ds:.0%}')

for ax_i, title in enumerate(['Intra-sub-motif MI', 'Inter-sub-motif MI']):
    ax = axes[ax_i]
    ax.axvline(-1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(-1.0 + 0.04, ax.get_ylim()[1] * 0.97, 'wᵢ = −t₀',
            fontsize=8, color='gray', va='top')
    ax.set_xlabel('wᵢ  (multiples of t₀)', fontsize=11)
    ax.set_ylabel('MI  (×10⁻³ bits)', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-2.05, 0.05)
    ax.set_xticks(np.linspace(-2, 0, N_STEPS))
    ax.set_xticklabels([f'{v:.1f}' for v in np.linspace(-2, 0, N_STEPS)], fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
out_path = OUT_DIR / 'goal3_mi_vs_wi.png'
fig.savefig(out_path, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved to {out_path}")
