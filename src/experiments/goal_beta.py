"""
Goal BETA: Triesch-like self-organisation of isolated core biases.

Each core neuron i adapts its bias b_i slowly to maintain average firing ≈ 0.5:
    b_i(t) = b_i(t-1) - epsilon * (z_i(t) - 0.5)

i.e.:
    z_i(t) = 1  →  b_i -= epsilon/2   (fired: reduce excitability)
    z_i(t) = 0  →  b_i += epsilon/2   (silent: increase excitability)

Starting from b_i = 0, biases converge to their p=0.5 fixed points.
MI is computed in a running window to show flux increasing over time.

Reference lines:
  - Full embedded core MI (Goal 3)
  - p=0.5 bias MI (Goal Alpha)
  - Mean-bias-only MI (Goal Alpha)

Usage (from project root):
    python src/experiments/goal_beta.py
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
from metrics.info            import mutual_information_joint
from metrics.motif_mi        import encode_motif_codes, pooled_joint_intra, pooled_joint_inter
from utils.io_paths          import ensure_dir

# ── Configuration ─────────────────────────────────────────────────────────────
NS      = 10
DS      = 0.35
SEED    = 42

EPSILON = 0.01       # learning rate
T_TOTAL = 2_000_000  # total timesteps of adaptive simulation
BURN_IN = 1_000      # initial burn-in with b=0 before adaptation starts

# Running MI window parameters
WIN_SIZE  = 50_000   # timesteps per MI window
WIN_STEP  = 25_000   # step between windows (50% overlap)

TRIPLETS = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
OUT_DIR  = Path('data/goal_beta')
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
print(f"epsilon={EPSILON}, T_total={T_TOTAL}, win_size={WIN_SIZE}, win_step={WIN_STEP}")

# ── Reference lines from Goal 3 and Goal Alpha ────────────────────────────────
def motif_mi(states):
    """Returns (intra, inter) in 1e-3 bits from (T, ns) uint8 states."""
    K     = TRIPLETS.shape[0]
    codes = encode_motif_codes(states, TRIPLETS)
    intra = float(np.mean([
        mutual_information_joint(pooled_joint_intra(codes[k:k+1, :]))
        for k in range(K)
    ])) * 1000
    inter = float(np.mean([
        mutual_information_joint(pooled_joint_inter(codes, np.array([[i, j]])))
        for i in range(K) for j in range(K) if i != j
    ])) * 1000
    return intra, inter

# Embedded reference
try:
    ref_emb_intra = float(np.load('data/goal3/goal3_ds35_intra.npy').max())
    ref_emb_inter = float(np.load('data/goal3/goal3_ds35_inter.npy').max())
    ref_emb = ref_emb_intra + ref_emb_inter
    print(f"\nEmbedded reference:   {ref_emb:.4f}  ×10⁻³ bits")
except FileNotFoundError:
    ref_emb = None
    print("Goal 3 data not found.")

# p=0.5 reference from Goal Alpha results
try:
    alpha_res = np.load('data/goal_alpha/results.npy', allow_pickle=True)
    ref_p05   = float(alpha_res[3]['intra'] + alpha_res[3]['inter'])
    ref_mu    = float(alpha_res[1]['intra'] + alpha_res[1]['inter'])
    print(f"p=0.5 bias reference: {ref_p05:.4f}  ×10⁻³ bits")
    print(f"Mean-bias reference:  {ref_mu:.4f}  ×10⁻³ bits")
except Exception:
    ref_p05 = ref_mu = None
    print("Goal Alpha data not found — recomputing p=0.5 reference ...")
    v_half = -(W_sk.T @ np.full(ns, 0.5))
    rng_r  = np.random.default_rng(SEED + 1)
    s_r    = rng_r.integers(0, 2, size=ns, dtype=np.uint8)
    bs_r   = v_half / t0
    for _ in range(10_000):
        z = s_r.astype(np.float64) @ W_sk / t0 + bs_r
        s_r = (rng_r.random(ns) < 1/(1+np.exp(-z))).astype(np.uint8)
    out_r = np.zeros((500_000, ns), dtype=np.uint8)
    for t in range(500_000):
        z   = s_r.astype(np.float64) @ W_sk / t0 + bs_r
        s_r = (rng_r.random(ns) < 1/(1+np.exp(-z))).astype(np.uint8)
        out_r[t] = s_r
    ref_p05_i, ref_p05_e = motif_mi(out_r)
    ref_p05 = ref_p05_i + ref_p05_e
    print(f"p=0.5 reference (recomputed): {ref_p05:.4f}")

# ── Adaptive simulation ───────────────────────────────────────────────────────
print(f"\nRunning adaptive simulation (T={T_TOTAL}) ...")

rng  = np.random.default_rng(SEED)
s    = rng.integers(0, 2, size=ns, dtype=np.uint8)
b    = np.zeros(ns, dtype=np.float64)   # adaptive biases, start at 0

# Burn-in without adaptation
for _ in range(BURN_IN):
    z = s.astype(np.float64) @ W_sk / t0 + b / t0
    s = (rng.random(ns) < 1.0 / (1.0 + np.exp(-z))).astype(np.uint8)

# Storage: full binary states and bias history
# (store bias only every RECORD_BIAS_EVERY steps to save memory)
RECORD_BIAS_EVERY = 1_000
states_all   = np.zeros((T_TOTAL, ns), dtype=np.uint8)
bias_history = np.zeros((T_TOTAL // RECORD_BIAS_EVERY, ns), dtype=np.float64)

for t in range(T_TOTAL):
    z    = s.astype(np.float64) @ W_sk / t0 + b / t0
    s    = (rng.random(ns) < 1.0 / (1.0 + np.exp(-z))).astype(np.uint8)
    # Triesch update: b_i -= epsilon * (z_i - 0.5)
    b   -= EPSILON * (s.astype(np.float64) - 0.5)
    states_all[t]  = s
    if t % RECORD_BIAS_EVERY == 0:
        bias_history[t // RECORD_BIAS_EVERY] = b

    if (t + 1) % 200_000 == 0:
        print(f"  t={t+1:>8d}  b={b.round(3)}")

print("Simulation done.")
print(f"Final biases: {b.round(4)}")
print(f"Final ⟨p₁⟩:  {states_all[-100_000:].mean(axis=0).round(4)}")

np.save(OUT_DIR / 'final_biases.npy', b)
np.save(OUT_DIR / 'bias_history.npy', bias_history)

# ── Running MI in sliding windows ─────────────────────────────────────────────
print(f"\nComputing running MI (window={WIN_SIZE}, step={WIN_STEP}) ...")

window_starts = np.arange(0, T_TOTAL - WIN_SIZE + 1, WIN_STEP)
n_windows     = len(window_starts)
mi_intra_run  = np.zeros(n_windows)
mi_inter_run  = np.zeros(n_windows)
mi_total_run  = np.zeros(n_windows)
t_centres     = window_starts + WIN_SIZE // 2

for k, t_start in enumerate(window_starts):
    win_states = states_all[t_start : t_start + WIN_SIZE]
    intra, inter = motif_mi(win_states)
    mi_intra_run[k] = intra
    mi_inter_run[k] = inter
    mi_total_run[k] = intra + inter
    if (k + 1) % 10 == 0:
        print(f"  window {k+1}/{n_windows}: t={t_centres[k]}  "
              f"MI_total={mi_total_run[k]:.4f}")

np.save(OUT_DIR / 'mi_total_running.npy',  mi_total_run)
np.save(OUT_DIR / 'mi_intra_running.npy',  mi_intra_run)
np.save(OUT_DIR / 'mi_inter_running.npy',  mi_inter_run)
np.save(OUT_DIR / 't_centres.npy',         t_centres)

print(f"\nInitial MI (first window): {mi_total_run[0]:.4f}")
print(f"Final   MI (last window):  {mi_total_run[-1]:.4f}")
if ref_emb:
    print(f"Embedded reference:        {ref_emb:.4f}")
if ref_p05:
    print(f"p=0.5 reference:           {ref_p05:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                          gridspec_kw=dict(wspace=0.35))

# ── Panel 1: Running MI ───────────────────────────────────────────────────────
ax = axes[0]
t_plot = t_centres / 1000   # convert to thousands of timesteps

ax.plot(t_plot, mi_total_run, color='#2166ac', lw=2.0,
        label='MI_total (running window)')
ax.fill_between(t_plot, mi_intra_run, alpha=0.3, color='#2166ac',
                label='intra-MI component')
ax.fill_between(t_plot, mi_intra_run, mi_total_run, alpha=0.3,
                color='#d6604d', label='inter-MI component')

if ref_emb:
    ax.axhline(ref_emb, color='black', lw=1.5, ls='--',
               label=f'full embedded ({ref_emb:.1f})')
if ref_p05:
    ax.axhline(ref_p05, color='green', lw=1.5, ls='-.',
               label=f'p=0.5 bias ({ref_p05:.1f})')
if ref_mu:
    ax.axhline(ref_mu, color='gray', lw=1.2, ls=':',
               label=f'mean bias μ_emb ({ref_mu:.1f})')

ax.set_xlabel('time  (×10³ steps)', fontsize=10)
ax.set_ylabel('MI_total  (×10⁻³ bits)', fontsize=10)
ax.set_title(f'Running motif MI  (window={WIN_SIZE//1000}k steps)\n'
             f'ε={EPSILON}, starting from b=0', fontsize=10)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

# ── Panel 2: Bias evolution ───────────────────────────────────────────────────
ax = axes[1]
t_bias = np.arange(len(bias_history)) * RECORD_BIAS_EVERY / 1000

cmap   = plt.cm.tab10
for i in range(ns):
    ax.plot(t_bias, bias_history[:, i], lw=1.2,
            color=cmap(i / ns), label=f'b_{i}', alpha=0.85)

# Final bias reference: p=0.5 mean-field
v_half = -(W_sk.T @ np.full(ns, 0.5))
for i in range(ns):
    ax.axhline(v_half[i], color=cmap(i / ns), lw=0.8, ls='--', alpha=0.5)

ax.set_xlabel('time  (×10³ steps)', fontsize=10)
ax.set_ylabel('adaptive bias  bᵢ', fontsize=10)
ax.set_title(f'Bias evolution  (dashed = p=0.5 mean-field target)\n'
             f'ε={EPSILON}', fontsize=10)
ax.legend(fontsize=7, ncol=2, loc='lower right')
ax.grid(True, alpha=0.3)

fig.suptitle(
    f'Goal β — Triesch self-organisation of isolated core\n'
    f'(ds={DS:.0%}, T={T_TOTAL//1000}k, ε={EPSILON})',
    fontsize=11
)

out_png = OUT_DIR / 'goal_beta.png'
out_pdf = OUT_DIR / 'goal_beta.pdf'
plt.tight_layout()
plt.savefig(out_png, dpi=180, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
