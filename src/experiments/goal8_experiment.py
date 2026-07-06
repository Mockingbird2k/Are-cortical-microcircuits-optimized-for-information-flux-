"""
Goal 8: Compute Pearson correlation matrices between neuron time series.

Naming convention:
  X[t,i] = time series of sea neuron i        shape (T, 90)
  Y[t,j] = time series of inter-node j        shape (T, 25)

Three matrices computed using professor's Pearson_Matrix function:
  C(X,X)  shape (90, 90)  — sea vs sea
  C(Y,Y)  shape (25, 25)  — inter vs inter
  C(X,Y)  shape (90, 25)  — sea vs inter

Usage (from project root):
    python src/experiments/goal8_experiment.py
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
OUT_DIR = Path('data/goal8')
ensure_dir(OUT_DIR)

STATES_FILE = Path('data/goal7/states_100k.npy')
IDX_FILE    = Path('data/goal7/idx_100k.npy')
T0_FILE     = Path('data/goal7/t0_100k.npy')

# ── Load or simulate ──────────────────────────────────────────────────────────
if STATES_FILE.exists():
    print("Loading states from Goal 7 data...")
    states = np.load(STATES_FILE)
    idx    = np.load(IDX_FILE, allow_pickle=True).item()
    t0     = float(np.load(T0_FILE))
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

# X[t,i]: sea activations,   shape (T, 90)
# Y[t,j]: inter activations, shape (T, 25)
X = states[:, sea].astype(float)
Y = states[:, inh].astype(float)
print(f"X shape: {X.shape},  Y shape: {Y.shape}")

# ── Professor's Pearson_Matrix function ───────────────────────────────────────
def Pearson_Matrix(X, Y):
    from numpy import mean, sqrt, sum, where
    Xc = X - mean(X, axis=0, keepdims=True)
    Yc = Y - mean(Y, axis=0, keepdims=True)
    Xn = sqrt(sum(Xc*Xc, axis=0))
    Yn = sqrt(sum(Yc*Yc, axis=0))
    D  = Xn[:, None] * Yn[None, :]
    D  = where(D == 0, 1.0, D)
    C  = (Xc.T @ Yc) / D
    return C

# ── Compute ───────────────────────────────────────────────────────────────────
print("Computing correlation matrices...")
C_XX = Pearson_Matrix(X, X)   # (90, 90)
C_YY = Pearson_Matrix(Y, Y)   # (25, 25)
C_XY = Pearson_Matrix(X, Y)   # (90, 25)

print(f"C(X,X): shape={C_XX.shape}  "
      f"mean_offdiag={C_XX[~np.eye(len(sea), dtype=bool)].mean():.4f}")
print(f"C(Y,Y): shape={C_YY.shape}  "
      f"mean_offdiag={C_YY[~np.eye(len(inh), dtype=bool)].mean():.4f}")
print(f"C(X,Y): shape={C_XY.shape}  "
      f"mean={C_XY.mean():.4f}")

np.save(OUT_DIR / 'C_XX.npy', C_XX)
np.save(OUT_DIR / 'C_YY.npy', C_YY)
np.save(OUT_DIR / 'C_XY.npy', C_XY)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 3, figsize=(16, 12),
    gridspec_kw={'width_ratios': [len(sea), len(inh), len(inh)], 'wspace': 0.35}
)
fig.suptitle(
    'Goal 8 — Pearson Correlation Matrices\n'
    f'Full Prebeck 125-node Network  |  ds={DS:.0%},  wᵢ=−t₀,  σ=0',
    fontsize=13, fontweight='bold'
)

panels = [
    (axes[0], C_XX, f'C(X,X)  —  Sea × Sea\n({len(sea)}×{len(sea)})',
     'Neuron i  (sea)', 'Neuron j  (sea)'),
    (axes[1], C_YY, f'C(Y,Y)  —  Inter × Inter\n({len(inh)}×{len(inh)})',
     'Neuron i  (inter)', 'Neuron j  (inter)'),
    (axes[2], C_XY, f'C(X,Y)  —  Sea × Inter\n({len(sea)}×{len(inh)})',
     'Neuron i  (sea)', 'Neuron j  (inter)'),
]

for ax, C, title, xlabel, ylabel in panels:
    im = ax.imshow(C, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                   interpolation='nearest')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Pearson r', fontsize=8)

out_path = OUT_DIR / 'goal8_correlations.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved to {out_path}")
