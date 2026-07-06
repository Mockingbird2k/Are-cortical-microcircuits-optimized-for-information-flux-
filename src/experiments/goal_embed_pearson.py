"""
goal_embed_pearson.py — Pearson correlation matrices of U_{i,embed} time series

Panels (b) and (c) of the supervisor's figure:
  (b) 10×10 Pearson correlation matrix of U_{i,embed}(t),  lagtime dt=0
  (c) 10×10 Pearson correlation matrix of U_{i,embed}(t) vs U_{j,embed}(t+1), dt=1

U_embed shape: (T, 10) — loaded from goal_embed_signal output.

Usage (from project root):
    python src/experiments/goal_embed_pearson.py
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.io_paths import ensure_dir

OUT_DIR = Path('data/goal_embed_signal')
ensure_dir(OUT_DIR)

# ── Load U_embed time series ──────────────────────────────────────────────────
U = np.load(OUT_DIR / 'U_embed_timeseries.npy')   # (T, 10)
T, ns = U.shape
print(f"Loaded U_embed: shape={U.shape}")

# ── Pearson correlation at dt=0 ───────────────────────────────────────────────
# Standard (T, 10) correlation matrix — np.corrcoef gives (10, 10)
C0 = np.corrcoef(U.T)   # (10, 10)

# ── Pearson correlation at dt=1 ───────────────────────────────────────────────
# C1[i,j] = corr( U_i(t), U_j(t+1) )  — NOT symmetric
U_ref = U[:-1]   # (T-1, 10)   at time t
U_lag = U[1:]    # (T-1, 10)   at time t+1

# Standardise each column
def standardise(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

U_ref_z = standardise(U_ref)   # (T-1, 10)
U_lag_z = standardise(U_lag)   # (T-1, 10)

# C1[i,j] = mean over t of z_i(t) * z_j(t+1)
C1 = (U_ref_z.T @ U_lag_z) / (T - 1)   # (10, 10)

print(f"\nC0 (dt=0) diagonal (should be 1.0): {np.diag(C0).round(4)}")
print(f"C0 range off-diag: [{C0[~np.eye(ns,dtype=bool)].min():.4f}, "
      f"{C0[~np.eye(ns,dtype=bool)].max():.4f}]")
print(f"\nC1 (dt=1) diagonal: {np.diag(C1).round(4)}")
print(f"C1 range: [{C1.min():.4f}, {C1.max():.4f}]")

np.save(OUT_DIR / 'pearson_C0.npy', C0)
np.save(OUT_DIR / 'pearson_C1.npy', C1)

# ── Plot ───────────────────────────────────────────────────────────────────────
tick_labels = [str(i) for i in range(ns)]
vmax = max(abs(C0[~np.eye(ns,dtype=bool)]).max(), abs(C1).max())
vmax = np.ceil(vmax * 10) / 10   # round up to nearest 0.1

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                          gridspec_kw=dict(wspace=0.35))

for ax, C, title, label in zip(
    axes,
    [C0, C1],
    [r'(b)  Pearson corr. of $U_{i,\mathrm{emb}}$   (dt=0)',
     r'(c)  Pearson corr. of $U_{i,\mathrm{emb}}$   (dt=1)'],
    ['(b)', '(c)']
):
    im = ax.imshow(C, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='equal', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Pearson r', fraction=0.046, pad=0.04)

    # Annotate each cell
    for i in range(ns):
        for j in range(ns):
            ax.text(j, i, f'{C[i,j]:.2f}', ha='center', va='center',
                    fontsize=6.5,
                    color='white' if abs(C[i,j]) > 0.5 * vmax else 'black')

    ax.set_xticks(range(ns)); ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(range(ns)); ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel('core neuron  j', fontsize=9)
    ax.set_ylabel('core neuron  i', fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)

fig.suptitle(
    r'Pearson correlation matrices of $U_{i,\mathrm{emb}}(t)$ time series'
    '\n(10 core neurons, T=100k, full embedded network)',
    fontsize=10
)

out_png = OUT_DIR / 'goal_embed_pearson.png'
out_pdf = OUT_DIR / 'goal_embed_pearson.pdf'
plt.savefig(out_png, dpi=180, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
