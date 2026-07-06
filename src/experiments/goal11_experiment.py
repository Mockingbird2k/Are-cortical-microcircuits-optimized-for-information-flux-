"""
Goal 11: Compute and plot the four joint pair probabilities p(0,0), p(0,1),
p(1,0), p(1,1) for real data vs statistically independent expectation.

For each pair of neurons (i,j) within a population, the four joint probabilities
are computed from the time series and averaged over all pairs i≠j. These are
then compared to the independent expectation:
  p00_ind = (1-pi)(1-pj)
  p01_ind = (1-pi)*pj
  p10_ind = pi*(1-pj)
  p11_ind = pi*pj

where pi and pj are the individual mean firing rates.

Any deviation between real and independent bars reveals genuine pairwise
correlations beyond what mean firing rates alone would predict.

Computed for:
  - Sea population X:   averaged over all 90*89 pairs
  - Inter population Y: averaged over all 25*24 pairs

Usage (from project root):
    python src/experiments/goal11_experiment.py
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
from utils.io_paths          import ensure_dir

# ── Configuration ─────────────────────────────────────────────────────────────
NS      = 10
DS      = 0.35
SEED    = 42
BURN_IN = 10_000
T       = 100_000
OUT_DIR = Path('data/goal11')
ensure_dir(OUT_DIR)

STATES_FILE = Path('data/goal7/states_100k.npy')
IDX_FILE    = Path('data/goal7/idx_100k.npy')

# ── Load or simulate ──────────────────────────────────────────────────────────
if STATES_FILE.exists():
    print("Loading states from Goal 7 data...")
    states = np.load(STATES_FILE)
    idx    = np.load(IDX_FILE, allow_pickle=True).item()
else:
    print(f"Running simulation: T={T}, ns={NS}, ds={DS:.0%}...")
    rng = np.random.default_rng(SEED)
    W, idx, t0 = build_prebeck_125node_weight_matrix(
        rng=rng, ns=NS, ds=DS, wi_mode='-t0'
    )
    nt  = W.shape[0]
    dyn = PrebeckBoltzmannDynamics(t0=t0, mask=np.ones(nt, dtype=bool))
    states = run_recorded_states(
        W=W, dynamics=dyn, T=T, burn_in=BURN_IN,
        sigma=0.0, noise_mode='different_input',
        seed=SEED, record_idx=np.arange(nt)
    )

sea = idx['sea']
inh = idx['inhibitory']

X = states[:, sea].astype(float)   # (T, 90)
Y = states[:, inh].astype(float)   # (T, 25)
print(f"X shape: {X.shape},  Y shape: {Y.shape}")

# ── Compute pair probabilities ────────────────────────────────────────────────
def pair_probs(Z):
    """
    Compute p(0,0), p(0,1), p(1,0), p(1,1) averaged over all pairs i≠j.
    Also returns the independent expectation based on individual firing rates.
    """
    N      = Z.shape[1]
    p_real = np.zeros(4)
    p_ind  = np.zeros(4)
    n_pairs = 0
    rates  = Z.mean(axis=0)   # individual firing rates, shape (N,)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Real joint probabilities from data
            p00 = ((1 - Z[:, i]) * (1 - Z[:, j])).mean()
            p01 = ((1 - Z[:, i]) *      Z[:, j] ).mean()
            p10 = (     Z[:, i]  * (1 - Z[:, j])).mean()
            p11 = (     Z[:, i]  *      Z[:, j] ).mean()
            p_real += [p00, p01, p10, p11]

            # Independent expectation
            pi = rates[i]; pj = rates[j]
            p_ind += [(1-pi)*(1-pj), (1-pi)*pj, pi*(1-pj), pi*pj]
            n_pairs += 1

    return p_real / n_pairs, p_ind / n_pairs

print(f"Computing pair probabilities for sea  ({len(sea)}×{len(sea)-1} pairs)...")
p_real_X, p_ind_X = pair_probs(X)

print(f"Computing pair probabilities for inter ({len(inh)}×{len(inh)-1} pairs)...")
p_real_Y, p_ind_Y = pair_probs(Y)

print(f"\nSea   — real: {np.round(p_real_X, 5)}  ind: {np.round(p_ind_X, 5)}")
print(f"Inter — real: {np.round(p_real_Y, 5)}  ind: {np.round(p_ind_Y, 5)}")

np.save(OUT_DIR / 'pair_probs_real_X.npy', p_real_X)
np.save(OUT_DIR / 'pair_probs_ind_X.npy',  p_ind_X)
np.save(OUT_DIR / 'pair_probs_real_Y.npy', p_real_Y)
np.save(OUT_DIR / 'pair_probs_ind_Y.npy',  p_ind_Y)

# ── Plot ──────────────────────────────────────────────────────────────────────
labels = ['p(0,0)', 'p(0,1)', 'p(1,0)', 'p(1,1)']
x      = np.arange(4)
width  = 0.35

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    'Goal 11 — Pair Probabilities: Real Data vs Statistically Independent\n'
    f'Full Prebeck 125-node Network  |  ds={DS:.0%},  wᵢ=−t₀,  σ=0',
    fontsize=12, fontweight='bold'
)

for ax, p_real, p_ind, color, title in [
    (axes[0], p_real_X, p_ind_X, '#2166ac',
     f'Sea population (X)\naveraged over all {len(sea)}×{len(sea)-1} pairs i≠j'),
    (axes[1], p_real_Y, p_ind_Y, '#d6604d',
     f'Inter-node population (Y)\naveraged over all {len(inh)}×{len(inh)-1} pairs i≠j'),
]:
    bars_real = ax.bar(x - width/2, p_real, width, label='Real data',
                       color=color, alpha=0.85, edgecolor='white')
    bars_ind  = ax.bar(x + width/2, p_ind,  width, label='Independent',
                       color=color, alpha=0.35, edgecolor='white', hatch='///')

    for bar in bars_real:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f'{bar.get_height():.4f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in bars_ind:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f'{bar.get_height():.4f}',
                ha='center', va='bottom', fontsize=8, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(p_real), max(p_ind)) * 1.25)

plt.tight_layout()
out_path = OUT_DIR / 'goal11_pair_probabilities.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved to {out_path}")
