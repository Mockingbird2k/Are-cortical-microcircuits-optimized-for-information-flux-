"""
Goal 6: Plot the signals flowing into the skeleton from the rest of the system,
split into two panels side by side:
  - Left:  Sea nodes  (90) — source of excitatory input to skeleton
  - Right: Inter-nodes (25) — source of inhibitory input to skeleton

Red = firing (1), Blue = silent (0). Binary spike matrix.
Fully connected Prebeck system, no cuts, sigma=0, wi=-t0.

Usage (from project root):
    python src/experiments/goal6_experiment.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from weights.prebeck_125node import build_prebeck_125node_weight_matrix
from core.dynamics_prebeck   import PrebeckBoltzmannDynamics
from core.simulate_states    import run_recorded_states
from utils.io_paths          import ensure_dir

# ── Configuration ─────────────────────────────────────────────────────────────
NS       = 10
DS       = 0.35
SEED     = 42
BURN_IN  = 10_000
T_RECORD = 300
T_SHOW   = 200
OUT_DIR  = Path('data/goal6')
ensure_dir(OUT_DIR)

# ── Build network ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
W, idx, t0 = build_prebeck_125node_weight_matrix(
    rng=rng, ns=NS, ds=DS, wi_mode='-t0'
)
sk  = idx['skeleton']
sea = idx['sea']
inh = idx['inhibitory']
nt  = W.shape[0]

print(f"=== SANITY CHECK ===")
print(f"Fully connected: {nt} nodes (no cuts)")
print(f"wi = -t0 = {-t0:.4f}")
print(f"t0 mask: all {nt} nodes")
print(f"Noise: sigma=0  (no external noise)")
print(f"Dynamics: unipolar s in {{0,1}}, noise added AFTER /t0")
print(f"Total edges: {np.count_nonzero(W)}")
print(f"All inh->exc weights = -{t0:.3f}? "
      f"{np.allclose(W[np.ix_(inh, list(sk)+list(sea))][W[np.ix_(inh, list(sk)+list(sea))] != 0], -t0)}")

# ── Run simulation, record all 125 nodes ──────────────────────────────────────
dyn = PrebeckBoltzmannDynamics(t0=t0, mask=np.ones(nt, dtype=bool))
states = run_recorded_states(
    W=W, dynamics=dyn, T=T_RECORD, burn_in=BURN_IN,
    sigma=0.0, noise_mode='different_input',
    seed=SEED, record_idx=np.arange(nt)
)
# states shape: (T_RECORD, 125), values in {0, 1}

print(f"\nStates shape: {states.shape}")
print(f"Values in {{0,1}} only: {np.all(np.isin(states, [0, 1]))}")
print(f"Mean fire rates:")
print(f"  Skeleton: {states[:, sk].mean():.3f}  (thesis target: ~0.52)")
print(f"  Sea:      {states[:, sea].mean():.3f}  (thesis target: ~0.29)")
print(f"  Inter:    {states[:, inh].mean():.3f}  (thesis target: ~0.62)")

# Save
np.save(OUT_DIR / 'all_states.npy', states)
np.save(OUT_DIR / 'node_idx.npy',   idx, allow_pickle=True)
np.save(OUT_DIR / 't0.npy',         np.array(t0))

# ── Plot: two panels side by side ─────────────────────────────────────────────
cmap = ListedColormap(['#3a6bbf', '#c0392b'])   # blue=silent, red=firing

fig, axes = plt.subplots(
    1, 2, figsize=(30, 7),
    gridspec_kw={'wspace': 0.06}
)
fig.suptitle(
    'Goal 6 — Signals Flowing into the Skeleton\n'
    f'Full Prebeck 125-node Network  |  ds={DS:.0%},  wᵢ=−t₀,  σ=0',
    fontsize=13, fontweight='bold'
)

panels = [
    (axes[0], sea, f'Sea → Skeleton\n(excitatory input source,  {len(sea)} nodes)'),
    (axes[1], inh, f'Inter-nodes → Skeleton\n(inhibitory input source,  {len(inh)} nodes)'),
]

for ax, node_idx, title in panels:
    n   = len(node_idx)
    mat = states[:T_SHOW, node_idx].T   # (n_nodes, T_SHOW)
    ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1,
              interpolation='nearest')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Time step', fontsize=10)
    ax.set_ylabel('Neuron', fontsize=10)
    ax.set_yticks([0, n - 1])
    ax.set_yticklabels(['0', f'{n-1}'], fontsize=8)
    ax.set_xticks(np.arange(0, T_SHOW + 1, 25))
    ax.set_xticklabels(np.arange(0, T_SHOW + 1, 25), fontsize=8)

# Shared colorbar on the right
cbar_ax = fig.add_axes([0.93, 0.12, 0.013, 0.72])
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax, ticks=[0.25, 0.75])
cb.ax.set_yticklabels(['0  silent', '1  firing'], fontsize=8)

out_path = OUT_DIR / 'goal6_signal_matrix.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved to {out_path}")
