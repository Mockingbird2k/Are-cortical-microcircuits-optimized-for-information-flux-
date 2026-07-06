"""
Goal 10: Compute lag-time dependent autocorrelation functions for sea (X) and
inter-node (Y) populations using the professor's corrFu function.

For each neuron i:
  Cxx,i(dt) = Pearson correlation between X[t,i] and X[t+dt,i]
  Cyy,j(dt) = Pearson correlation between Y[t,j] and Y[t+dt,j]

Lag range: dt = -10 to +10 timesteps.

Outputs:
  - ACF_XX: matrix of shape (90, 21) — one ACF per sea neuron
  - ACF_YY: matrix of shape (25, 21) — one ACF per inter neuron
  - ACF_XX_av: population-averaged sea ACF (21,)
  - ACF_YY_av: population-averaged inter ACF (21,)

Usage (from project root):
    python src/experiments/goal10_experiment.py
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
DT_MIN  = -10
DT_MAX  =  10
OUT_DIR = Path('data/goal10')
ensure_dir(OUT_DIR)

# Example neurons to show in individual plots
EXAMPLE_SEA = [0, 10, 20, 30, 40]
EXAMPLE_INH = [0, 5, 10, 15, 20]

# ── Load states ───────────────────────────────────────────────────────────────
print("Loading states from Goal 7 data...")
states = np.load(Path('data/goal7/states_100k.npy'))
idx    = np.load(Path('data/goal7/idx_100k.npy'), allow_pickle=True).item()
sea    = idx['sea']
inh    = idx['inhibitory']

X = states[:, sea].astype(float)   # (T, 90)
Y = states[:, inh].astype(float)   # (T, 25)
print(f"X shape: {X.shape},  Y shape: {Y.shape}")

# ── Professor's corrFu function ───────────────────────────────────────────────
def corrFu(a, b, dtMin, dtMax):
    from numpy import zeros, mean, std
    NDT = 1 + dtMax - dtMin
    DT  = zeros(shape=(NDT,), dtype=float)
    CF  = zeros(shape=(NDT,), dtype=float)
    k = -1
    for dt in range(dtMin, dtMax + 1):
        k += 1
        DT[k] = dt
        if dt < 0:
            aS = a[-dt:]
            bS = b[:dt]
        elif dt == 0:
            aS = a
            bS = b
        else:
            aS = a[:-dt]
            bS = b[dt:]
        stA = std(aS)
        stB = std(bS)
        if stA * stB != 0:
            aSN = (aS - mean(aS)) / stA
            bSN = (bS - mean(bS)) / stB
            CF[k] = mean(aSN * bSN)
        else:
            CF[k] = 0
    return DT, CF

# ── Compute ACFs for all neurons ──────────────────────────────────────────────
NDT = 1 + DT_MAX - DT_MIN
ACF_XX = np.zeros((X.shape[1], NDT))   # (90, 21)
ACF_YY = np.zeros((Y.shape[1], NDT))   # (25, 21)

print(f"Computing ACFs for {X.shape[1]} sea neurons...")
for i in range(X.shape[1]):
    DT, CF = corrFu(X[:, i], X[:, i], DT_MIN, DT_MAX)
    ACF_XX[i] = CF

print(f"Computing ACFs for {Y.shape[1]} inter neurons...")
for j in range(Y.shape[1]):
    DT, CF = corrFu(Y[:, j], Y[:, j], DT_MIN, DT_MAX)
    ACF_YY[j] = CF

# Population averages
ACF_XX_av = ACF_XX.mean(axis=0)
ACF_YY_av = ACF_YY.mean(axis=0)

print(f"\nACF_XX_av at dt=+1: {ACF_XX_av[DT_MAX + 1]:.5f}")
print(f"ACF_YY_av at dt=+1: {ACF_YY_av[DT_MAX + 1]:.5f}")

np.save(OUT_DIR / 'ACF_XX.npy',    ACF_XX)
np.save(OUT_DIR / 'ACF_YY.npy',    ACF_YY)
np.save(OUT_DIR / 'ACF_XX_av.npy', ACF_XX_av)
np.save(OUT_DIR / 'ACF_YY_av.npy', ACF_YY_av)
np.save(OUT_DIR / 'DT.npy',        DT)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Goal 10 — Lag-time Autocorrelation Functions\n'
    f'Full Prebeck 125-node Network  |  ds={DS:.0%},  wᵢ=−t₀,  σ=0',
    fontsize=13, fontweight='bold'
)

# Top left: example individual sea neurons
ax = axes[0, 0]
for i in EXAMPLE_SEA:
    ax.semilogy(DT, np.abs(ACF_XX[i]), 'o-', alpha=0.7, linewidth=1.2,
                markersize=4, label=f'sea node {i}')
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Lag time  Δt', fontsize=10)
ax.set_ylabel('|Cₓₓ,ᵢ(Δt)|  (log scale)', fontsize=10)
ax.set_title('Example individual sea neurons\nCₓₓ,ᵢ(Δt)', fontsize=10, fontweight='bold')
ax.set_xticks(np.arange(int(DT.min()), int(DT.max()) + 1, 1))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

# Top right: example individual inter neurons
ax = axes[0, 1]
for j in EXAMPLE_INH:
    ax.semilogy(DT, np.abs(ACF_YY[j]), 'o-', alpha=0.7, linewidth=1.2,
                markersize=4, label=f'inter node {j}')
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Lag time  Δt', fontsize=10)
ax.set_ylabel('|C_yy,j(Δt)|  (log scale)', fontsize=10)
ax.set_title('Example individual inter-nodes\nC_yy,j(Δt)', fontsize=10, fontweight='bold')
ax.set_xticks(np.arange(int(DT.min()), int(DT.max()) + 1, 1))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

# Bottom left: population-averaged sea
ax = axes[1, 0]
ax.semilogy(DT, np.abs(ACF_XX_av), 'o-', color='#2166ac', linewidth=2.2, markersize=5)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Lag time  Δt', fontsize=10)
ax.set_ylabel('|Cₓₓ,av(Δt)|  (log scale)', fontsize=10)
ax.set_title('Population-averaged sea autocorrelation\nCₓₓ,av(Δt)', fontsize=10, fontweight='bold')
ax.set_xticks(np.arange(int(DT.min()), int(DT.max()) + 1, 1))
ax.grid(True, alpha=0.3, which='both')

# Bottom right: population-averaged inter
ax = axes[1, 1]
ax.semilogy(DT, np.abs(ACF_YY_av), 'o-', color='#d6604d', linewidth=2.2, markersize=5)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Lag time  Δt', fontsize=10)
ax.set_ylabel('|C_yy,av(Δt)|  (log scale)', fontsize=10)
ax.set_title('Population-averaged inter autocorrelation\nC_yy,av(Δt)', fontsize=10, fontweight='bold')
ax.set_xticks(np.arange(int(DT.min()), int(DT.max()) + 1, 1))
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
out_path = OUT_DIR / 'goal10_autocorr.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved to {out_path}")
