"""
goal_embed_signal.py  —  PDFs of U_i,embed for the 10 core neurons

For each core neuron i, the total embedding signal at time t is:
    U_i,embed(t) = U_i,inter(t) + U_i,peri(t)
                 = sum_m  s_m(t) * W[m,i]   (m in inter)
                 + sum_n  s_n(t) * W[n,i]   (n in peri)

We plot the PDF of U_i,embed as a 2×5 grid (one panel per core neuron).
We also save the 10 time series for later Pearson correlation analysis.

Convention: W[j,i] = weight from neuron j to neuron i.

Usage (from project root):
    python src/experiments/goal_embed_signal.py
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from weights.prebeck_125node import build_prebeck_125node_weight_matrix
from utils.io_paths          import ensure_dir

# ── Config ────────────────────────────────────────────────────────────────────
NS   = 10
DS   = 0.35
SEED = 42

OUT_DIR = Path('data/goal_embed_signal')
ensure_dir(OUT_DIR)

# ── Build network & extract weight blocks ─────────────────────────────────────
rng = np.random.default_rng(SEED)
W, idx, t0 = build_prebeck_125node_weight_matrix(rng=rng, ns=NS, ds=DS, wi_mode='-t0')

sk  = idx['skeleton'].astype(int)    # (10,)
inh = idx['inhibitory'].astype(int)  # (25,)
sea = idx['sea'].astype(int)         # (90,)
ns  = len(sk)

# W[j, i] = weight from j to i
# W_inter_to_core: shape (25, 10)  — columns = core neurons
W_IC = W[np.ix_(inh, sk)]   # inter → core
W_PC = W[np.ix_(sea, sk)]   # peri  → core

print(f't0 = {t0:.4f}')
print(f'W_IC shape: {W_IC.shape},  nonzero: {(W_IC!=0).sum()},  mean nonzero: {W_IC[W_IC!=0].mean():.4f}' if (W_IC!=0).any() else 'W_IC all zero')
print(f'W_PC shape: {W_PC.shape},  nonzero: {(W_PC!=0).sum()},  mean nonzero: {W_PC[W_PC!=0].mean():.4f}' if (W_PC!=0).any() else 'W_PC all zero')

# ── Load 100k states ──────────────────────────────────────────────────────────
states = np.load('data/goal_pairwise_mi/states_100k.npy')   # (T, 125)
T = states.shape[0]
print(f'\nLoaded states: {states.shape}')

s_inter = states[:, inh].astype(np.float64)   # (T, 25)
s_peri  = states[:, sea].astype(np.float64)   # (T, 90)

# ── Compute U_i,embed for all 10 core neurons ─────────────────────────────────
# U_embed shape: (T, 10)
U_inter = s_inter @ W_IC   # (T, 25) @ (25, 10) = (T, 10)
U_peri  = s_peri  @ W_PC   # (T, 90) @ (90, 10) = (T, 10)
U_embed = U_inter + U_peri  # (T, 10)

print(f'\nU_embed shape: {U_embed.shape}')
print(f'Per-neuron mean (bias):')
for i in range(ns):
    print(f'  core[{i}]: mean={U_embed[:,i].mean():.4f}  std={U_embed[:,i].std():.4f}'
          f'  min={U_embed[:,i].min():.2f}  max={U_embed[:,i].max():.2f}')

# Save time series for later Pearson correlation analysis
np.save(OUT_DIR / 'U_embed_timeseries.npy', U_embed)
print(f'\nSaved U_embed time series → {OUT_DIR}/U_embed_timeseries.npy')

# ── Plot: 2×5 grid of PDFs ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(13, 5.5),
                          gridspec_kw=dict(hspace=0.45, wspace=0.35))

for i, ax in enumerate(axes.flat):
    u = U_embed[:, i]

    # KDE for smooth PDF
    kde = gaussian_kde(u, bw_method='scott')
    x_grid = np.linspace(u.min() - 0.5, u.max() + 0.5, 400)
    pdf    = kde(x_grid)

    ax.plot(x_grid, pdf, color='black', lw=1.5)
    ax.fill_between(x_grid, pdf, alpha=0.15, color='black')

    # Mean marker
    mu  = u.mean()
    sig = u.std()
    ax.axvline(mu, color='red', lw=1.0, linestyle='--', label=f'μ={mu:.2f}')

    ax.set_title(f'core neuron {i}', fontsize=8.5)
    ax.set_xlabel(r'$U_{i,\mathrm{embed}}$', fontsize=8)
    ax.set_ylabel('PDF', fontsize=8)
    ax.tick_params(labelsize=7.5)

    # Annotate mean and std
    ax.text(0.97, 0.95, f'μ={mu:.2f}\nσ={sig:.2f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=7.5, color='red')

fig.suptitle(r'PDFs of total embedding signal $U_{i,\mathrm{embed}}(t) = U_{i,\mathrm{inter}} + U_{i,\mathrm{peri}}$'
             '\nfor each of the 10 core neurons  (T=100k, full embedded network)',
             fontsize=10, y=1.01)

out_png = OUT_DIR / 'goal_embed_signal.png'
out_pdf = OUT_DIR / 'goal_embed_signal.pdf'
plt.savefig(out_png, dpi=180, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f'Saved: {out_png}')
print(f'Saved: {out_pdf}')
