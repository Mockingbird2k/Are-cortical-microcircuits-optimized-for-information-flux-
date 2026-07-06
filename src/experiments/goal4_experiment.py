"""
Goal 4: Systematically cut each of the 6 cuttable connection types in the
125-node network and measure the resulting sub-motif MI as bar plots.

The 7 connection types in the network (from the diagram):
  1. Skeleton self-loop  (++ strongly excitatory) -- NEVER CUT, essential for RNN
  2. Skeleton  -> inter  (+ excitatory)
  3. Inter     -> skeleton  (-- strongly inhibitory)
  4. Sea       -> inter  (+ excitatory)
  5. Inter     -> sea    (-- strongly inhibitory)
  6. Skeleton  -> sea    (+ excitatory)
  7. Sea       -> skeleton  (+ excitatory)

Each of the 6 cuttable connections is zeroed out in turn and MI is measured,
then all results are shown in a single bar plot per ds.

Usage (from project root):
    python src/experiments/goal4_experiment.py
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
NS       = 10
DS_LIST  = [0.35, 0.45]
T        = 400_000
BURN_IN  = 10_000
SEED     = 42
TRIPLETS = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
OUT_DIR  = Path('data/goal4')
ensure_dir(OUT_DIR)

# Each cut: (bar label, source region key, target region key)
# None means baseline (no cut)
CUTS = [
    ('Baseline\n(no cut)',    None,          None),
    ('Cut\nsk→inter',         'skeleton',    'inhibitory'),
    ('Cut\ninter→sk',         'inhibitory',  'skeleton'),
    ('Cut\nsea→inter',        'sea',         'inhibitory'),
    ('Cut\ninter→sea',        'inhibitory',  'sea'),
    ('Cut\nsk→sea',           'skeleton',    'sea'),
    ('Cut\nsea→sk',           'sea',         'skeleton'),
]

BAR_COLORS = [
    '#4dac26',   # baseline       - green
    '#b8e186',   # sk->inter      - light green
    '#d01c8b',   # inter->sk      - strong pink (most dramatic)
    '#f1b6da',   # sea->inter     - light pink
    '#f4a582',   # inter->sea     - light orange
    '#92c5de',   # sk->sea        - light blue
    '#2166ac',   # sea->sk        - blue
]

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
    region = {
        'skeleton':   idx['skeleton'],
        'sea':        idx['sea'],
        'inhibitory': idx['inhibitory'],
    }
    print(f"  t0 = {t0:.4f}")

    intra_list, inter_list, labels = [], [], []

    for label, src_key, tgt_key in CUTS:
        W = W_base.copy()
        if src_key is not None:
            src = region[src_key]
            tgt = region[tgt_key]
            W[np.ix_(src, tgt)] = 0.0

        intra, inter = run_embedded(W, t0)
        intra_list.append(intra)
        inter_list.append(inter)
        labels.append(label)

        tag = label.replace('\n', ' ')
        print(f"  {tag:22s} | intra = {intra:7.3f}  inter = {inter:7.3f}  (x10-3 bits)")

    results[ds] = dict(
        labels=labels,
        intra=np.array(intra_list),
        inter=np.array(inter_list),
        t0=t0
    )

    np.save(OUT_DIR / f'goal4_ds{int(ds*100):02d}_intra.npy', np.array(intra_list))
    np.save(OUT_DIR / f'goal4_ds{int(ds*100):02d}_inter.npy', np.array(inter_list))


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Goal 4 — Effect of Cutting Each Connection Type on Sub-motif MI\n'
    '(embedded 125-node network, σ = 0, ns = 10, wᵢ = −t₀)',
    fontsize=13, fontweight='bold'
)

for row_i, ds in enumerate(DS_LIST):
    r = results[ds]
    for col_i, (mi_vals, mi_label) in enumerate([
        (r['intra'], 'Intra-sub-motif MI'),
        (r['inter'], 'Inter-sub-motif MI')
    ]):
        ax   = axes[row_i, col_i]
        x    = np.arange(len(r['labels']))
        bars = ax.bar(x, mi_vals, color=BAR_COLORS,
                      edgecolor='white', linewidth=0.8, width=0.65)

        # Baseline reference line
        ax.axhline(mi_vals[0], color='#4dac26', linestyle='--',
                   linewidth=1.2, alpha=0.5)

        # Value labels on bars
        for bar, val in zip(bars, mi_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(mi_vals) * 0.01,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(r['labels'], fontsize=8.5)
        ax.set_ylabel('MI  (×10⁻³ bits)', fontsize=10)
        ax.set_title(f'ds = {ds:.0%}  —  {mi_label}', fontsize=10)
        ax.set_ylim(0, max(mi_vals) * 1.18)
        ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / 'goal4_cuts_barplot.png'
fig.savefig(out_path, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved to {out_path}")
