"""
Goal 9: Compute Pearson correlation matrices for shuffled versions of X and Y.

Shuffling destroys all temporal and spatial structure while preserving the
marginal distribution of values. The resulting correlation matrices serve as
a null baseline — any structure in Goal 8 that survives shuffling is purely
distributional, not dynamical.

Naming convention:
  X'  = shuffled version of X  (sea activations)
  Y'  = shuffled version of Y  (inter activations)

Three shuffled matrices:
  C(X',X')  shape (90, 90)
  C(Y',Y')  shape (25, 25)
  C(X',Y')  shape (90, 25)

Plotted alongside original Goal 8 matrices for direct comparison.

Usage (from project root):
    python src/experiments/goal9_experiment.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.io_paths import ensure_dir

# ── Configuration ─────────────────────────────────────────────────────────────
DS      = 0.35
OUT_DIR = Path('data/goal9')
ensure_dir(OUT_DIR)

# ── Load states from Goal 7 ───────────────────────────────────────────────────
print("Loading states from Goal 7 data...")
states = np.load(Path('data/goal7/states_100k.npy'))
idx    = np.load(Path('data/goal7/idx_100k.npy'), allow_pickle=True).item()
sea    = idx['sea']
inh    = idx['inhibitory']

X = states[:, sea].astype(float)   # (T, 90)
Y = states[:, inh].astype(float)   # (T, 25)
print(f"X shape: {X.shape},  Y shape: {Y.shape}")

# ── Professor's functions ─────────────────────────────────────────────────────
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

def Shuffle_Matrix(X):
    from numpy import reshape
    from numpy.random import shuffle
    flat = X.reshape(-1)
    shuffle(flat)
    Xs = reshape(flat, X.shape)
    return Xs

# ── Original correlation matrices (Goal 8) ────────────────────────────────────
C_XX = np.load(Path('data/goal8/C_XX.npy'))
C_YY = np.load(Path('data/goal8/C_YY.npy'))
C_XY = np.load(Path('data/goal8/C_XY.npy'))

# ── Shuffled versions ─────────────────────────────────────────────────────────
np.random.seed(42)
Xs = Shuffle_Matrix(X.copy())
Ys = Shuffle_Matrix(Y.copy())

print("Computing shuffled correlation matrices...")
C_XsXs = Pearson_Matrix(Xs, Xs)
C_YsYs = Pearson_Matrix(Ys, Ys)
C_XsYs = Pearson_Matrix(Xs, Ys)

print(f"C(X',X'): mean_offdiag={C_XsXs[~np.eye(len(sea), dtype=bool)].mean():.6f}")
print(f"C(Y',Y'): mean_offdiag={C_YsYs[~np.eye(len(inh), dtype=bool)].mean():.6f}")
print(f"C(X',Y'): mean={C_XsYs.mean():.6f}")

np.save(OUT_DIR / 'C_XsXs.npy', C_XsXs)
np.save(OUT_DIR / 'C_YsYs.npy', C_YsYs)
np.save(OUT_DIR / 'C_XsYs.npy', C_XsYs)

# ── Plot: 2 rows (original top, shuffled bottom) ──────────────────────────────
fig, axes = plt.subplots(
    2, 3, figsize=(16, 11),
    gridspec_kw={
        'width_ratios': [len(sea), len(inh), len(inh)],
        'wspace': 0.35, 'hspace': 0.4
    }
)
fig.suptitle(
    'Goal 8 & 9 — Pearson Correlation Matrices: Original vs Shuffled\n'
    f'Full Prebeck 125-node Network  |  ds={DS:.0%},  wᵢ=−t₀,  σ=0',
    fontsize=13, fontweight='bold'
)

rows = [
    ('Original',           C_XX,   C_YY,   C_XY),
    ("Shuffled (X', Y')",  C_XsXs, C_YsYs, C_XsYs),
]
col_info = [
    ("C(X,X) / C(X',X')\nSea × Sea",       'Neuron i  (sea)',   'Neuron j  (sea)'),
    ("C(Y,Y) / C(Y',Y')\nInter × Inter",    'Neuron i  (inter)', 'Neuron j  (inter)'),
    ("C(X,Y) / C(X',Y')\nSea × Inter",      'Neuron i  (sea)',   'Neuron j  (inter)'),
]

for row_i, (row_label, Cxx, Cyy, Cxy) in enumerate(rows):
    for col_i, (C, (title, xlabel, ylabel)) in enumerate(
        zip([Cxx, Cyy, Cxy], col_info)
    ):
        ax = axes[row_i, col_i]
        # Scale each panel to its own off-diagonal range so structure is visible
        if C.shape[0] == C.shape[1]:
            mask = ~np.eye(C.shape[0], dtype=bool)
            vmax = max(np.abs(C[mask]).max(), 1e-6)
        else:
            vmax = max(np.abs(C).max(), 1e-6)
        im = ax.imshow(C, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_title(f'{title}\n[{row_label}]  vmax={vmax:.5f}',
                     fontsize=8, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Pearson r', fontsize=7)

out_path = OUT_DIR / 'goal9_shuffled_correlations.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved to {out_path}")
