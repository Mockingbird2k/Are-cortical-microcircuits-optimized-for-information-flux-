"""
goal_raster_allpops.py — Raster plot of ALL 3 populations (embedded network)

Reproduces the supervisor's sketch:
  - 3 stacked panels: core (top), inter (middle), perip/sea (bottom)
  - y-axis: individual neuron indices
  - x-axis: time step
  - Black dot = neuron fired at that timestep
  - Embedded full network, sigma=0, wi=-t0

Usage (from project root):
    python src/experiments/goal_raster_allpops.py
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

# ── Configuration ─────────────────────────────────────────────────────────────
T_SHOW  = 100      # timesteps to display (matches the sketch x-axis: 0 to 99)
T_START = 0        # starting timestep from the 100k run

OUT_DIR = Path('data/goal_raster')
ensure_dir(OUT_DIR)

# ── Load states and node indices ──────────────────────────────────────────────
states  = np.load('soep/results/states_full.npy')   # (100000, 125)
idx     = np.load('data/goal6/node_idx.npy', allow_pickle=True).item()
sk      = idx['skeleton']    # (10,)
inh     = idx['inhibitory']  # (25,)
sea     = idx['sea']         # (90,)

# Take the display window
win = states[T_START : T_START + T_SHOW, :]   # (T_SHOW, 125)

print(f"Display window: t={T_START} to {T_START + T_SHOW - 1}")
print(f"Mean fire rates in window:")
print(f"  Core  (n={len(sk)}):  {win[:, sk].mean():.3f}")
print(f"  Inter (n={len(inh)}): {win[:, inh].mean():.3f}")
print(f"  Perip (n={len(sea)}): {win[:, sea].mean():.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
# Proportions matched to sketch: taller panels, stacked vertically
# Heights proportional to neuron counts: core=10, inter=25, peri=90
n_core  = len(sk)
n_inter = len(inh)
n_peri  = len(sea)

fig, axes = plt.subplots(
    3, 1,
    figsize=(7, 5.5),
    gridspec_kw=dict(
        hspace=0.15,
        height_ratios=[n_core, n_inter, n_peri]
    )
)

pop_data = [
    (win[:, sk],  sk,  n_core,  'core\nneuron'),
    (win[:, inh], inh, n_inter, 'inter\nneuron'),
    (win[:, sea], sea, n_peri,  'perip.\nneuron'),
]

for ax, (pop_win, pop_idx, n_pop, label) in zip(axes, pop_data):
    # Raster: for each (t, neuron_within_pop), plot dot if fired
    t_fired, n_fired = np.where(pop_win == 1)
    ax.scatter(t_fired, n_fired, s=4, c='black', marker='.', linewidths=0)

    ax.set_xlim(-0.5, T_SHOW - 0.5)
    ax.set_ylim(-0.5, n_pop - 0.5)

    # y-axis: show 0 at top, n-1 at bottom  (as in sketch)
    ax.set_yticks([0, n_pop - 1])
    ax.set_yticklabels(['0', str(n_pop - 1)], fontsize=8)
    ax.set_ylabel(label, fontsize=8.5, labelpad=4)

    ax.tick_params(axis='x', labelbottom=False)

    # Box border only (no grid)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

# x-axis only on bottom panel
axes[-1].tick_params(axis='x', labelbottom=True, labelsize=8)
axes[-1].set_xticks([0, T_SHOW - 1])
axes[-1].set_xticklabels(['0', str(T_SHOW - 1)], fontsize=8)
axes[-1].set_xlabel('time step', fontsize=9)

out_png = OUT_DIR / 'goal_raster_allpops.png'
out_pdf = OUT_DIR / 'goal_raster_allpops.pdf'
plt.savefig(out_png, dpi=200, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
