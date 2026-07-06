"""
Goal 5: Simultaneously cut sk->inter AND sk->sea to test whether the skeleton
participates in a feedback loop with the inter-nodes, or whether the inter-nodes
act purely as a one-way structured noise source.

Hypothesis (supervisor): if cutting both sk->inter and sk->sea has no effect on MI,
the skeleton is not driving the inter-nodes at all — the inter-nodes are running
on sea activity alone and projecting back as sophisticated noise, not as feedback.

Usage (from project root):
    python src/experiments/goal5_experiment.py
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
NS      = 10
DS_LIST = [0.35, 0.45]
T       = 1_000_000
BURN_IN = 10_000
SEED    = 42
TRIPLETS = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
OUT_DIR  = Path('data/goal5')
ensure_dir(OUT_DIR)

CONDITIONS = [
    ('Baseline\n(no cut)',           None,         None),
    ('Cut\nsk→inter',                'skeleton',   'inhibitory'),
    ('Cut\nsk→sea',                  'skeleton',   'sea'),
    ('Cut sk→inter\n+ sk→sea',       'both',       None),
]

BAR_COLORS = ['#4dac26', '#b8e186', '#92c5de', '#d01c8b']

# ── Helpers ──────────────────────────────────────────────────────────────────
def mi_from_states(states, triplets):
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


def run_embedded(W, t0, seed=SEED):
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

    n_sk_inter = np.count_nonzero(W_base[np.ix_(sk, inh)])
    n_sk_sea   = np.count_nonzero(W_base[np.ix_(sk, sea)])
    n_sea_inter = np.count_nonzero(W_base[np.ix_(sea, inh)])
    print(f"  t0={t0:.3f}")
    print(f"  sk->inter edges:  {n_sk_inter}  (vs sea->inter: {n_sea_inter})")
    print(f"  sk->sea edges:    {n_sk_sea}")

    intra_list, inter_list, labels = [], [], []

    for label, src_key, tgt_key in CONDITIONS:
        W = W_base.copy()
        if src_key == 'both':
            W[np.ix_(sk, inh)] = 0.0
            W[np.ix_(sk, sea)] = 0.0
        elif src_key is not None:
            region = {'skeleton': sk, 'sea': sea, 'inhibitory': inh}
            W[np.ix_(region[src_key], region[tgt_key])] = 0.0

        intra, inter = run_embedded(W, t0)
        intra_list.append(intra)
        inter_list.append(inter)
        labels.append(label)

        tag = label.replace('\n', ' ')
        diff_str = f"  (Δ intra={intra-intra_list[0]:+.3f})" if len(intra_list) > 1 else ""
        print(f"  {tag:<30}  intra={intra:7.3f}  inter={inter:7.3f}{diff_str}")

    results[ds] = dict(
        labels=labels,
        intra=np.array(intra_list),
        inter=np.array(inter_list),
        t0=t0
    )
    np.save(OUT_DIR / f'goal5_ds{int(ds*100):02d}_intra.npy', np.array(intra_list))
    np.save(OUT_DIR / f'goal5_ds{int(ds*100):02d}_inter.npy', np.array(inter_list))


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    'Goal 5 — Does the Skeleton Drive the Inter-nodes? (Feedback vs Noise Input)\n'
    'Cutting sk→inter and sk→sea simultaneously  |  ns=10, σ=0, wᵢ=−t₀',
    fontsize=12, fontweight='bold'
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
                      edgecolor='white', linewidth=0.8, width=0.55)

        # Baseline reference line
        ax.axhline(mi_vals[0], color='#4dac26', linestyle='--',
                   linewidth=1.2, alpha=0.5)

        # Value labels
        for bar, val in zip(bars, mi_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(mi_vals) * 0.01,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        # Percentage change labels below bars
        for i, (bar, val) in enumerate(zip(bars, mi_vals)):
            if i > 0:
                pct = (val - mi_vals[0]) / mi_vals[0] * 100
                ax.text(bar.get_x() + bar.get_width() / 2,
                        max(mi_vals) * 0.04,
                        f'{pct:+.1f}%', ha='center', va='bottom',
                        fontsize=8, color='#555555')

        ax.set_xticks(x)
        ax.set_xticklabels(r['labels'], fontsize=9)
        ax.set_ylabel('MI  (×10⁻³ bits)', fontsize=10)
        ax.set_title(f'ds = {ds:.0%}  —  {mi_label}', fontsize=10)
        ax.set_ylim(0, max(mi_vals) * 1.22)
        ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / 'goal5_feedback_test.png'
fig.savefig(out_path, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved to {out_path}")
