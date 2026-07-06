"""
goal_beta_convergence.py — Bias convergence under Triesch self-organisation.

Runs the Triesch update rule on the isolated skeleton:
    b_i(t) = b_i(t-1) - epsilon * (z_i(t) - 0.5)

Starting from b_i = 0, shows how each bias converges to its p=0.5 fixed point.
No MI computation — just the bias trajectories.

Reference lines: p=0.5 mean-field target per neuron (dashed).

Usage (from project root):
    python src/experiments/goal_beta_convergence.py
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
from utils.io_paths          import ensure_dir

# ── Configuration ─────────────────────────────────────────────────────────────
NS      = 10
DS      = 0.35
SEED    = 42

EPSILON = 0.01
T_TOTAL = 10_000    # short — biases converge well before this

RECORD_EVERY = 100   # record bias every N steps (for smooth curves)

OUT_DIR = Path('data/goal_beta')
ensure_dir(OUT_DIR)

# ── Build isolated skeleton ───────────────────────────────────────────────────
rng_build = np.random.default_rng(SEED)
W_full, idx, t0 = build_prebeck_125node_weight_matrix(
    rng=rng_build, ns=NS, ds=DS, wi_mode='-t0'
)
sk   = idx['skeleton'].astype(int)
ns   = len(sk)
W_sk = W_full[np.ix_(sk, sk)]

print(f"Isolated skeleton: n={ns}, t0={t0:.4f}")
print(f"epsilon={EPSILON}, T_total={T_TOTAL}")

# ── p=0.5 mean-field target biases ───────────────────────────────────────────
v_half = -(W_sk.T @ np.full(ns, 0.5))
print(f"\np=0.5 mean-field targets: {v_half.round(4)}")

# ── Adaptive simulation ───────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
s   = rng.integers(0, 2, size=ns, dtype=np.uint8)
b   = np.zeros(ns, dtype=np.float64)

n_records    = T_TOTAL // RECORD_EVERY
bias_history = np.zeros((n_records, ns), dtype=np.float64)
t_recorded   = np.zeros(n_records, dtype=np.int64)

for t in range(T_TOTAL):
    z = s.astype(np.float64) @ W_sk / t0 + b / t0
    s = (rng.random(ns) < 1.0 / (1.0 + np.exp(-z))).astype(np.uint8)
    b -= EPSILON * (s.astype(np.float64) - 0.5)

    if t % RECORD_EVERY == 0:
        k = t // RECORD_EVERY
        bias_history[k] = b
        t_recorded[k]   = t

print(f"\nFinal biases:   {b.round(4)}")
print(f"MF targets:     {v_half.round(4)}")
print(f"Difference:     {(b - v_half).round(4)}")

np.save(OUT_DIR / 'bias_convergence.npy',   bias_history)
np.save(OUT_DIR / 't_convergence.npy',      t_recorded)
np.save(OUT_DIR / 'mf_target_biases.npy',   v_half)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

cmap = plt.cm.tab10
t_plot = t_recorded / 1000   # x-axis in thousands of steps

for i in range(ns):
    color = cmap(i / ns)
    ax.plot(t_plot, bias_history[:, i], lw=1.5, color=color,
            label=f'b_{i}', alpha=0.9)
    ax.axhline(v_half[i], color=color, lw=0.8, ls='--', alpha=0.5)

ax.axhline(np.nan, color='gray', lw=0.8, ls='--', alpha=0.7,
           label='p=0.5 MF target (dashed)')
ax.axvline(0, color='black', lw=0.5)

ax.set_xlabel('time  (×10³ steps)', fontsize=10)
ax.set_ylabel('adaptive bias  bᵢ', fontsize=10)
ax.set_title(
    f'Goal β — Triesch bias convergence  (isolated skeleton, ds={DS:.0%})\n'
    f'ε={EPSILON},  starting from bᵢ=0  |  dashed = p=0.5 mean-field target',
    fontsize=10
)
ax.legend(fontsize=8, ncol=2, loc='lower right')
ax.grid(True, alpha=0.3)

out_png = OUT_DIR / 'goal_beta_convergence.png'
out_pdf = OUT_DIR / 'goal_beta_convergence.pdf'
plt.tight_layout()
plt.savefig(out_png, dpi=180, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
