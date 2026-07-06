"""
Goal ALPHA: Bar diagram comparing core info flux (motif MI) for 4 conditions.

Conditions:
  1. Full embedded core  (reference from Goal 3)
  2. Isolated core + mean bias  mu_i  (from U_i,emb PDFs)
  3. Isolated core + mean bias  mu_i  + Gaussian noise  sigma_i  (from U_i,emb PDFs)
  4. Isolated core + p=0.5 bias per neuron  (mean-field)

Each bar shows intra-MI and inter-MI stacked, total MI annotated on top.
Reference line: full embedded MI.

Usage (from project root):
    python src/experiments/goal_alpha.py
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
T_EVAL  = 1_000_000   # timesteps per condition
BURN_IN = 10_000

TRIPLETS = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
OUT_DIR  = Path('data/goal_alpha')
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

# ── Load U_embed statistics (mu and sigma per core neuron) ────────────────────
U = np.load('data/goal_embed_signal/U_embed_timeseries.npy')  # (T, 10)
mu_emb  = U.mean(axis=0)   # (10,)  mean embedding signal per neuron
sig_emb = U.std(axis=0)    # (10,)  std  embedding signal per neuron

print(f"\nU_embed statistics per core neuron:")
print(f"  mu  = {mu_emb.round(4)}")
print(f"  sig = {sig_emb.round(4)}")

# ── p=0.5 bias via mean-field ─────────────────────────────────────────────────
# We want sigmoid((W_sk.T @ p + v) / t0) = 0.5  ⟹  W_sk.T @ p + v = 0
# At p=0.5: v = -W_sk.T @ 0.5
v_half = -(W_sk.T @ np.full(ns, 0.5))
print(f"\np=0.5 mean-field bias: {v_half.round(4)}")

# ── Simulation function ───────────────────────────────────────────────────────
def sim(W, t0, v_bias, sigma_noise, T, burn, seed):
    """
    Boltzmann dynamics with per-neuron constant bias and optional Gaussian noise.
    z_i = (sum_j s_j * W[j,i] + v_i) / t0  +  N(0, sigma_i)   if sigma_i > 0
    """
    n   = W.shape[0]
    rng = np.random.default_rng(seed)
    s   = rng.integers(0, 2, size=n, dtype=np.uint8)
    bs  = v_bias / t0
    use_noise = np.any(sigma_noise > 0)

    for _ in range(burn):
        z = s.astype(np.float64) @ W / t0 + bs
        if use_noise:
            z += rng.normal(0.0, 1.0, size=n) * sigma_noise
        s = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.uint8)

    out = np.zeros((T, n), dtype=np.uint8)
    for t in range(T):
        z = s.astype(np.float64) @ W / t0 + bs
        if use_noise:
            z += rng.normal(0.0, 1.0, size=n) * sigma_noise
        s = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.uint8)
        out[t] = s
    return out


def motif_mi(states):
    """Returns (intra, inter) in 1e-3 bits."""
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

# ── Load embedded reference (Goal 3) ─────────────────────────────────────────
try:
    emb_intra = float(np.load('data/goal3/goal3_ds35_intra.npy').max())
    emb_inter = float(np.load('data/goal3/goal3_ds35_inter.npy').max())
    print(f"\nEmbedded reference (Goal 3): "
          f"intra={emb_intra:.4f}  inter={emb_inter:.4f}  "
          f"total={emb_intra+emb_inter:.4f}  ×10⁻³ bits")
except FileNotFoundError:
    # Recompute embedded MI from the 100k states
    print("\nGoal 3 data not found — computing embedded MI from 100k states ...")
    states_emb = np.load('data/goal_pairwise_mi/states_100k.npy')[:, sk]
    emb_intra, emb_inter = motif_mi(states_emb)
    print(f"Embedded: intra={emb_intra:.4f}  inter={emb_inter:.4f}  ×10⁻³ bits")

# ── Run 3 isolated conditions ─────────────────────────────────────────────────
conditions = [
    # (label, v_bias, sigma_noise)
    ('mean bias\n(μ_emb)',
     mu_emb,
     np.zeros(ns)),

    ('mean bias + noise\n(μ_emb, σ_emb)',
     mu_emb,
     sig_emb),

    ('p=0.5 bias\n(mean-field)',
     v_half,
     np.zeros(ns)),
]

results = []

# Condition 1: full embedded (already loaded)
results.append({
    'label': 'full embedded\ncore',
    'intra': emb_intra,
    'inter': emb_inter,
})

for k, (label, v_bias, sigma_noise) in enumerate(conditions):
    print(f"\nRunning: {label.replace(chr(10), ' ')} ...")
    states = sim(W_sk, t0, v_bias, sigma_noise, T_EVAL, BURN_IN, SEED + k)
    intra, inter = motif_mi(states)
    p1 = states.mean(axis=0)
    print(f"  intra={intra:.4f}  inter={inter:.4f}  total={intra+inter:.4f}  ×10⁻³ bits")
    print(f"  ⟨p₁⟩ = {p1.round(3)}")
    results.append({'label': label, 'intra': intra, 'inter': inter})

# ── Save ──────────────────────────────────────────────────────────────────────
np.save(OUT_DIR / 'results.npy', np.array(results, dtype=object))

print("\n── Summary ──────────────────────────────────────────────────────")
for r in results:
    tot = r['intra'] + r['inter']
    print(f"  {r['label'].replace(chr(10),' '):35s}  "
          f"intra={r['intra']:.4f}  inter={r['inter']:.4f}  total={tot:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
n_cond   = len(results)
x        = np.arange(n_cond)
labels   = [r['label'] for r in results]
intras   = np.array([r['intra'] for r in results])
inters   = np.array([r['inter'] for r in results])
totals   = intras + inters

fig, ax = plt.subplots(figsize=(10, 5.5))

bar_w = 0.55
b1 = ax.bar(x, intras, bar_w, label='intra-MI',
            color='#2166ac', alpha=0.85, edgecolor='black', lw=0.6)
b2 = ax.bar(x, inters, bar_w, bottom=intras, label='inter-MI',
            color='#d6604d', alpha=0.85, edgecolor='black', lw=0.6)

# Annotate total on top of each bar
for xi, tot in zip(x, totals):
    ax.text(xi, tot + 0.5, f'{tot:.1f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

# Reference line: full embedded total
emb_total = emb_intra + emb_inter
ax.axhline(emb_total, color='black', lw=1.5, ls='--', alpha=0.7,
           label=f'full embedded total ({emb_total:.1f})')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel('Motif MI  (×10⁻³ bits)', fontsize=11)
ax.set_title(
    'Goal α — Core info flux for different embedding conditions\n'
    f'(isolated skeleton, ds={DS:.0%}, T={T_EVAL//1000}k)',
    fontsize=11
)
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, max(totals) * 1.18)
ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

out_png = OUT_DIR / 'goal_alpha.png'
out_pdf = OUT_DIR / 'goal_alpha.pdf'
plt.tight_layout()
plt.savefig(out_png, dpi=180, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
