"""
goal_raster_core.py — Core neuron activation raster for 4 conditions

Shows only the skeleton/core neuron activations over time
(the top panel of the supervisor's sketch), for:
  a) isolated core
  b) isolated core with optimal noise
  c) embedded core
  d) isolated core with optimal bias  [update BIAS_OPT after bias sweep]

4 panels arranged vertically, each matching the style of the sketch top panel.

Usage (from project root):
    python src/experiments/goal_raster_core.py
"""

from __future__ import annotations
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
NS        = 10
DS        = 0.35
SEED      = 42
T_VIS     = 100      # timesteps shown (matches sketch x-axis: 0 to 99)
BURN_IN   = 2_000

SIGMA_OPT = 2.0      # optimal sigma from SR curve (Goal 2, ds=35%)
BIAS_OPT  = 0.5      # placeholder — update after bias sweep

OUT_DIR = Path('data/goal_raster')
ensure_dir(OUT_DIR)

# ── Build network ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
W, idx, t0 = build_prebeck_125node_weight_matrix(rng=rng, ns=NS, ds=DS, wi_mode='-t0')
sk  = idx['skeleton']
ns  = len(sk)   # 10
nt  = W.shape[0]

# Isolated skeleton: extract (ns × ns) subblock
W_sk = W[np.ix_(sk, sk)]

print(f"Network: nt={nt}, ns={ns}, t0={t0:.4f}")

# ── Dynamics ──────────────────────────────────────────────────────────────────
dyn_iso = PrebeckBoltzmannDynamics(t0=t0, mask=np.ones(ns, dtype=bool))
mask_emb = np.zeros(nt, dtype=bool)
mask_emb[sk] = True
dyn_emb = PrebeckBoltzmannDynamics(t0=t0, mask=mask_emb)

# ── Isolated with bias: manual loop (bias added to normalised input) ──────────
def run_isolated_with_bias(W_sk, t0, sigma, bias_norm, seed, T=T_VIS, burn=BURN_IN):
    rng_r = np.random.default_rng(seed)
    n = W_sk.shape[0]
    s = rng_r.integers(0, 2, size=n, dtype=np.uint8)
    for _ in range(burn):
        z = s.astype(float) @ W_sk / t0 + bias_norm
        if sigma > 0:
            z += rng_r.normal(0.0, sigma, size=n)
        s = (rng_r.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.uint8)
    out = np.zeros((T, n), dtype=np.uint8)
    for t in range(T):
        z = s.astype(float) @ W_sk / t0 + bias_norm
        if sigma > 0:
            z += rng_r.normal(0.0, sigma, size=n)
        s = (rng_r.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.uint8)
        out[t] = s
    return out   # (T, ns)

# ── Run all 4 conditions ──────────────────────────────────────────────────────
print("\nRunning condition a: isolated core, σ=0, bias=0 ...")
states_a = run_recorded_states(
    W=W_sk, dynamics=dyn_iso, T=T_VIS, burn_in=BURN_IN,
    sigma=0.0, noise_mode='different_input',
    seed=SEED, record_idx=np.arange(ns)
)   # (T_VIS, ns)

print(f"Running condition b: isolated core + σ={SIGMA_OPT} ...")
states_b = run_recorded_states(
    W=W_sk, dynamics=dyn_iso, T=T_VIS, burn_in=BURN_IN,
    sigma=SIGMA_OPT, noise_mode='different_input',
    seed=SEED + 1, record_idx=np.arange(ns)
)

print("Running condition c: embedded core ...")
states_c = run_recorded_states(
    W=W, dynamics=dyn_emb, T=T_VIS, burn_in=BURN_IN,
    sigma=0.0, noise_mode='different_input',
    seed=SEED + 2, record_idx=sk.astype(int)
)   # (T_VIS, ns)

print(f"Running condition d: isolated core + bias={BIAS_OPT} ...")
states_d = run_isolated_with_bias(
    W_sk=W_sk, t0=t0, sigma=0.0, bias_norm=BIAS_OPT, seed=SEED + 3
)

all_states = [states_a, states_b, states_c, states_d]
cond_labels = [
    f"a)  isolated core",
    f"b)  isolated core + optimal noise  (σ={SIGMA_OPT})",
    f"c)  embedded core",
    f"d)  isolated core + optimal bias  (bias={BIAS_OPT})",
]

for lab, s in zip(cond_labels, all_states):
    print(f"  {lab[:30]}:  ⟨s⟩={s.mean():.3f}")

# ── Plot: 4 stacked panels, each like the sketch top panel ────────────────────
fig, axes = plt.subplots(
    4, 1,
    figsize=(7, 6.5),
    gridspec_kw=dict(hspace=0.12)
)

for ax, states, label in zip(axes, all_states, cond_labels):
    t_fired, n_fired = np.where(states == 1)
    ax.scatter(t_fired, n_fired, s=4, c='black', marker='.', linewidths=0)

    ax.set_xlim(-0.5, T_VIS - 0.5)
    ax.set_ylim(-0.5, ns - 0.5)
    ax.set_yticks([0, ns - 1])
    ax.set_yticklabels(['0', str(ns - 1)], fontsize=8)
    ax.set_ylabel('core\nneuron', fontsize=8.5, labelpad=4)
    ax.tick_params(axis='x', labelbottom=False)

    # Condition label in top-left corner
    ax.text(0.01, 0.93, label, transform=ax.transAxes,
            fontsize=8, va='top', ha='left')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

axes[-1].tick_params(axis='x', labelbottom=True, labelsize=8)
axes[-1].set_xticks([0, T_VIS - 1])
axes[-1].set_xticklabels(['0', str(T_VIS - 1)], fontsize=8)
axes[-1].set_xlabel('time step', fontsize=9)

out_png = OUT_DIR / 'goal_raster_core.png'
out_pdf = OUT_DIR / 'goal_raster_core.pdf'
plt.savefig(out_png, dpi=200, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
