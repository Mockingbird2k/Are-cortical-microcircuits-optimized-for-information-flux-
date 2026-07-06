"""
fig6a_5bars.py — Figure 6(a): flux indicator bar chart, 5 conditions

Reads all values from the actual simulation result files.

Data sources:
  - Full signals:          soep/results/mi_embedded.npy
  - Indiv. biases:         data/goal_alpha/alpha_intra.npy  [index 1]
  - Biases + noise:        data/goal_alpha/alpha_intra.npy  [index 2]
  - Uniform optimal:       soep/results/intra_bias.npy + inter_bias.npy (peak)
  - Indiv. optimized:      data/goal13/g13_intra.npy + g13_inter.npy
                           (falls back to goal_alpha index 3 if not found)

Usage (from project root):
    python src/experiments/fig6a_5bars.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path('data/goal_alpha')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Full signals (full embedded core) ──────────────────────────────────────
mi_emb     = np.load('soep/results/mi_embedded.npy')   # [intra, inter] x10^-3 bits
intra_full = float(mi_emb[0])
inter_full = float(mi_emb[1])

# ── 2 & 3. Indiv. biases and biases + noise (Goal Alpha conditions 1 & 2) ────
alpha_intra = np.load('data/goal_alpha/alpha_intra.npy')  # [full, bias, bias+noise, p05]
alpha_inter = np.load('data/goal_alpha/alpha_inter.npy')
intra_bias_only  = float(alpha_intra[1])
inter_bias_only  = float(alpha_inter[1])
intra_bias_noise = float(alpha_intra[2])
inter_bias_noise = float(alpha_inter[2])

# ── 4. Uniform optimal biases (Goal 12 peak) ──────────────────────────────────
intra_b = np.load('data/goal12/mi_intra.npy')
inter_b = np.load('data/goal12/mi_inter.npy')
pk      = int(np.argmax(intra_b + inter_b))
intra_unif = float(intra_b[pk])
inter_unif = float(inter_b[pk])

# ── 5. Indiv. optimized biases (Goal 13) ──────────────────────────────────────
try:
    intra_g13 = float(np.load('data/goal13/g13_intra.npy'))
    inter_g13 = float(np.load('data/goal13/g13_inter.npy'))
    print("Loaded Goal 13 MI from data/goal13/")
except FileNotFoundError:
    # Fall back to p=0.5 bias result from Goal Alpha (index 3)
    intra_g13 = float(alpha_intra[3])
    inter_g13 = float(alpha_inter[3])
    print("Goal 13 MI not found — using Goal Alpha p=0.5 result as proxy")

# ── Assemble ──────────────────────────────────────────────────────────────────
labels = [
    'full signals',
    'indiv. biases',
    'indiv. biases\nplus normal noise',
    'uniform optimal\nbiases',
    'indiv. optimized\nbiases',
]
intras = np.array([intra_full, intra_bias_only, intra_bias_noise,
                   intra_unif, intra_g13])
inters = np.array([inter_full, inter_bias_only, inter_bias_noise,
                   inter_unif, inter_g13])
totals = intras + inters

emb_ref = intra_full + inter_full   # dashed reference line

print("\nFlux indicators:")
for lbl, tot in zip(labels, totals):
    print(f"  {lbl.replace(chr(10),' '):35s}: {tot:.1f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
BLUE   = '#4472C4'
SALMON = '#E07060'

fig, ax = plt.subplots(figsize=(8.5, 5.2))

x = np.arange(len(labels))
w = 0.55

ax.bar(x, intras, w, color=BLUE,   label='intra-triplet', zorder=3)
ax.bar(x, inters, w, color=SALMON, label='inter-triplet',
       bottom=intras, zorder=3)

ax.axhline(emb_ref, color='black', linewidth=1.4, linestyle='--', zorder=2)

for xi, tot in zip(x, totals):
    ax.text(xi, tot + 2.5, f'{tot:.1f}', ha='center', va='bottom',
            fontsize=9.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel('flux indicator', fontsize=11)
ax.set_ylim(0, max(totals) * 1.14)
ax.set_xlim(-0.5, len(labels) - 0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', alpha=0.35, zorder=0)
ax.legend(fontsize=9.5, loc='upper left', frameon=False)
ax.tick_params(axis='x', length=0)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig6a_5bars.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(OUT_DIR / 'fig6a_5bars.pdf', bbox_inches='tight')
plt.close()
print(f"\nSaved to {OUT_DIR}/")
