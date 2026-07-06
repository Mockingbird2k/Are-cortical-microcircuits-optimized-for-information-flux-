"""
goal_pairwise_mi.py — 125×125 pairwise mutual information matrix

Computes M[i,j] = MI(s_i(t), s_j(t+dt)) for every pair of neurons (binary),
then collapses to a 3×3 subpopulation matrix by averaging within blocks.

As the supervisor noted: for binary units, pairwise MI characterises
correlations more naturally than Pearson, since Pearson is optimal for
continuous variables.

MI values are in [0, 1] bits (max entropy of a binary variable = 1 bit).

Outputs:
  data/goal_pairwise_mi/pairwise_mi_dt0.npy   — (125,125) matrix
  data/goal_pairwise_mi/pairwise_mi_dt1.npy   — (125,125) matrix
  data/goal_pairwise_mi/goal_pairwise_mi.png  — figure
  data/goal_pairwise_mi/goal_pairwise_mi.pdf

Usage (from project root):
    python src/experiments/goal_pairwise_mi.py
"""

from __future__ import annotations
import sys, time
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
NS      = 10
DS      = 0.35
SEED    = 42
T_SIM   = 100_000   # timesteps
BURN_IN = 10_000

OUT_DIR  = Path('data/goal_pairwise_mi')
CACHE_ST = Path('data/goal_pairwise_mi/states_100k.npy')   # 100k full simulation
ensure_dir(OUT_DIR)

# ── Build network ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
W, idx, t0 = build_prebeck_125node_weight_matrix(rng=rng, ns=NS, ds=DS, wi_mode='-t0')
sk  = idx['skeleton']
sea = idx['sea']
inh = idx['inhibitory']
nt  = W.shape[0]   # 125

print(f"Network: nt={nt}, ns={len(sk)}, n_sea={len(sea)}, n_inh={len(inh)}")
print(f"t0={t0:.4f}")

# ── Load or run simulation ────────────────────────────────────────────────────
if CACHE_ST.exists():
    print(f"\nLoading cached states from {CACHE_ST} ...")
    states = np.load(CACHE_ST)   # (T, 125) from goal6
    print(f"  Loaded: {states.shape}")
else:
    print(f"\nNo cached states found. Running simulation T={T_SIM} ...")
    mask = np.zeros(nt, dtype=bool)
    mask[sk] = True
    dyn = PrebeckBoltzmannDynamics(t0=t0, mask=mask)
    states = run_recorded_states(
        W=W, dynamics=dyn, T=T_SIM, burn_in=BURN_IN,
        sigma=0.0, noise_mode='different_input',
        seed=SEED, record_idx=np.arange(nt)
    )   # (T_SIM, 125)
    np.save(OUT_DIR / 'states_cache.npy', states)
    print(f"  Saved cache to {OUT_DIR / 'states_cache.npy'}")

T_actual = states.shape[0]
print(f"States: ({T_actual}, {nt}), dtype={states.dtype}")
print(f"Mean firing rates:  sk={states[:, sk].mean():.3f}  "
      f"sea={states[:, sea].mean():.3f}  inh={states[:, inh].mean():.3f}")

# ── Pairwise MI function (vectorised) ─────────────────────────────────────────
def pairwise_mi_binary(states: np.ndarray, dt: int = 0) -> np.ndarray:
    """
    Compute N×N matrix of pairwise MI(s_i(t), s_j(t+dt)).

    dt=0 : instantaneous MI (symmetric)
    dt=1 : lagged MI (not necessarily symmetric)

    Algorithm:
      For binary neurons, joint distribution has 4 cells: (00, 01, 10, 11).
      p(1,1) = (s_ref.T @ s_lag) / T_use  computed as a single matrix multiply.
      The other three cells follow from marginals.
      MI = sum_{a,b in {0,1}} p(a,b) * log2( p(a,b) / (p_ref(a) * p_lag(b)) )

    Returns (N, N) float64 MI matrix in bits.  Values in [0, 1].
    """
    T, N   = states.shape
    eps    = 1e-15

    if dt == 0:
        s_ref  = states.astype(np.float64)
        s_lag  = states.astype(np.float64)
        T_use  = T
    else:
        s_ref  = states[:T - dt].astype(np.float64)
        s_lag  = states[dt:].astype(np.float64)
        T_use  = T - dt

    p1_ref = s_ref.mean(axis=0)   # (N,)  marginal of reference frame
    p0_ref = 1.0 - p1_ref
    p1_lag = s_lag.mean(axis=0)   # (N,)  marginal of lagged frame
    p0_lag = 1.0 - p1_lag

    # Joint counts via single matrix multiply: (N,N)
    p11 = (s_ref.T @ s_lag) / T_use   # p(ref=1, lag=1)
    p10 = p1_ref[:, None] - p11        # p(ref=1, lag=0)
    p01 = p1_lag[None, :] - p11        # p(ref=0, lag=1)
    p00 = 1.0 - p11 - p10 - p01

    def kl_cell(p_joint, p_r, p_l):
        """Contribution p(a,b) * log2(p(a,b) / (p_r * p_l)) across all pairs."""
        denom = p_r[:, None] * p_l[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where((p_joint > eps) & (denom > eps),
                             p_joint / denom, 1.0)
            contrib = np.where(p_joint > eps, p_joint * np.log2(ratio), 0.0)
        return contrib

    M  = kl_cell(p00, p0_ref, p0_lag)
    M += kl_cell(p01, p0_ref, p1_lag)
    M += kl_cell(p10, p1_ref, p0_lag)
    M += kl_cell(p11, p1_ref, p1_lag)

    np.clip(M, 0.0, None, out=M)   # remove floating-point negatives
    return M

# ── Feasibility timing test ───────────────────────────────────────────────────
print("\n─── Feasibility timing test ───")
t_start = time.time()
M0 = pairwise_mi_binary(states, dt=0)
t_dt0 = time.time() - t_start
print(f"  dt=0 completed in {t_dt0:.2f}s")

t_start = time.time()
M1 = pairwise_mi_binary(states, dt=1)
t_dt1 = time.time() - t_start
print(f"  dt=1 completed in {t_dt1:.2f}s")
print(f"  Total: {t_dt0 + t_dt1:.2f}s  → feasibility confirmed ✓")

# Diagonal of M0 = MI(s_i, s_i) = H(s_i) — should equal binary entropy of firing rate
h_sk_expected = np.array([
    -p * np.log2(p + 1e-15) - (1-p) * np.log2(1-p + 1e-15)
    for p in states[:, sk].mean(axis=0)
])
print(f"\n  Diagonal sanity (M0 diagonal == H(s_i) for skeleton):")
print(f"    M0 diagonal:  {M0[np.ix_(sk,sk)].diagonal().mean():.4f}  bits")
print(f"    H(s_i) mean:  {h_sk_expected.mean():.4f}  bits")

# ── Save matrices ─────────────────────────────────────────────────────────────
np.save(OUT_DIR / 'pairwise_mi_dt0.npy', M0)
np.save(OUT_DIR / 'pairwise_mi_dt1.npy', M1)
print(f"\nM0: off-diag range [{M0[~np.eye(nt,dtype=bool)].min():.4f}, "
      f"{M0[~np.eye(nt,dtype=bool)].max():.4f}]  "
      f"mean={M0[~np.eye(nt,dtype=bool)].mean():.4f}  bits")
print(f"M1: off-diag range [{M1.min():.4f}, {M1.max():.4f}]  "
      f"mean={M1.mean():.4f}  bits")

# ── Collapse to 3×3 subpopulation block averages ──────────────────────────────
subpops    = [sk.astype(int), inh.astype(int), sea.astype(int)]
pop_labels = ['core', 'inter', 'peri']

def collapse_3x3(M, subpops):
    """Average pairwise MI within/between subpopulations.
    For diagonal blocks (same population), self-pairs (i==j) are excluded
    so the result matches what the 125x125 plot shows (diagonal zeroed)."""
    K   = len(subpops)
    out = np.zeros((K, K))
    for i, pi in enumerate(subpops):
        for j, pj in enumerate(subpops):
            block = M[np.ix_(pi, pj)]
            if i == j:
                mask = ~np.eye(len(pi), dtype=bool)
                out[i, j] = block[mask].mean()
            else:
                out[i, j] = block.mean()
    return out

M0_3x3 = collapse_3x3(M0, subpops)
M1_3x3 = collapse_3x3(M1, subpops)

print("\n3×3 averaged MI (dt=0)  [bits]:")
header = "        " + "  ".join(f"{l:>7}" for l in pop_labels)
print(header)
for i, lab in enumerate(pop_labels):
    row = "  ".join(f"{M0_3x3[i,j]:7.4f}" for j in range(3))
    print(f"  {lab:5s}: {row}")

print("\n3×3 averaged MI (dt=1)  [bits]:")
print(header)
for i, lab in enumerate(pop_labels):
    row = "  ".join(f"{M1_3x3[i,j]:7.4f}" for j in range(3))
    print(f"  {lab:5s}: {row}")

# ── Plot: 2×2 — full matrix and 3×3 for each dt ──────────────────────────────
off_diag_mask = ~np.eye(nt, dtype=bool)
vmax0 = M0[off_diag_mask].max()
vmax1 = M1.max()

fig, axes = plt.subplots(2, 2, figsize=(11, 9),
                          gridspec_kw=dict(hspace=0.38, wspace=0.38))

# ── Reorder matrix rows/cols to core→inter→peri for consistent visual layout ──
# Raw matrix indices: core=0-9, sea=10-99, inter=100-124
# We reorder so the 125x125 plot shows them as: core | inter | peri
order      = np.concatenate([sk.astype(int), inh.astype(int), sea.astype(int)])
M0_ordered = M0[np.ix_(order, order)]
M1_ordered = M1[np.ix_(order, order)]

# Boundaries in the reordered matrix: at len(core), len(core)+len(inter)
boundaries = [0, len(sk), len(sk) + len(inh), len(sk) + len(inh) + len(sea)]

def draw_full(ax, M_ord, title, vmax):
    M_disp = M_ord.copy()
    np.fill_diagonal(M_disp, 0.0)
    im = ax.imshow(M_disp, aspect='auto', cmap='gray_r', vmin=0, vmax=vmax,
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label='MI  (bits)', fraction=0.046, pad=0.04)
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='white', lw=0.7, alpha=0.8)
        ax.axvline(b - 0.5, color='white', lw=0.7, alpha=0.8)
    centres = [(boundaries[k] + boundaries[k+1]) / 2 for k in range(len(subpops))]
    ax.set_xticks(centres); ax.set_xticklabels(pop_labels, fontsize=9)
    ax.set_yticks(centres); ax.set_yticklabels(pop_labels, fontsize=9)
    ax.set_title(title, fontsize=10)

def draw_3x3(ax, M3, title):
    im = ax.imshow(M3, aspect='equal', cmap='gray_r', vmin=0, vmax=M3.max())
    plt.colorbar(im, ax=ax, label='mean MI  (bits)', fraction=0.046, pad=0.04)
    ax.set_xticks(range(3)); ax.set_xticklabels(pop_labels, fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels(pop_labels, fontsize=9)
    for i in range(3):
        for j in range(3):
            val = M3[i, j]
            ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                    fontsize=9.5, color='red',
                    fontweight='bold')
    ax.set_title(title, fontsize=10)

# ── Version A: self-pairs EXCLUDED (diagonal zeroed in 125x125 and 3x3) ──────
draw_full(axes[0, 0], M0_ordered, '125×125 pairwise MI  (dt=0,  diagonal zeroed)', vmax0)
draw_full(axes[0, 1], M1_ordered, '125×125 pairwise MI  (dt=1)', vmax1)
draw_3x3(axes[1, 0], M0_3x3, '3×3 mean MI, excl. self-pairs  (dt=0)')
draw_3x3(axes[1, 1], M1_3x3, '3×3 mean MI, excl. self-pairs  (dt=1)')

out_png = OUT_DIR / 'goal_pairwise_mi_excl_self.png'
out_pdf = OUT_DIR / 'goal_pairwise_mi_excl_self.pdf'
plt.savefig(out_png, dpi=180, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"Saved: {out_pdf}")

# ── Version B: self-pairs INCLUDED (diagonal shown in 125x125 and 3x3) ───────
# Recompute 3x3 with diagonal included
def collapse_3x3_with_diag(M, subpops):
    K   = len(subpops)
    out = np.zeros((K, K))
    for i, pi in enumerate(subpops):
        for j, pj in enumerate(subpops):
            out[i, j] = M[np.ix_(pi, pj)].mean()
    return out

M0_3x3_diag = collapse_3x3_with_diag(M0, subpops)
M1_3x3_diag = collapse_3x3_with_diag(M1, subpops)

def draw_full_with_diag(ax, M_ord, title):
    # Include diagonal — scale to full matrix max
    vmax = M_ord.max()
    im = ax.imshow(M_ord, aspect='auto', cmap='gray_r', vmin=0, vmax=vmax,
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label='MI  (bits)', fraction=0.046, pad=0.04)
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='red', lw=0.7, alpha=0.8)
        ax.axvline(b - 0.5, color='red', lw=0.7, alpha=0.8)
    centres = [(boundaries[k] + boundaries[k+1]) / 2 for k in range(len(subpops))]
    ax.set_xticks(centres); ax.set_xticklabels(pop_labels, fontsize=9)
    ax.set_yticks(centres); ax.set_yticklabels(pop_labels, fontsize=9)
    ax.set_title(title, fontsize=10)

fig2, axes2 = plt.subplots(2, 2, figsize=(11, 9),
                            gridspec_kw=dict(hspace=0.38, wspace=0.38))

draw_full_with_diag(axes2[0, 0], M0_ordered, '125×125 pairwise MI  (dt=0,  diagonal included)')
draw_full_with_diag(axes2[0, 1], M1_ordered, '125×125 pairwise MI  (dt=1,  diagonal included)')
draw_3x3(axes2[1, 0], M0_3x3_diag, '3×3 mean MI, incl. self-pairs  (dt=0)')
draw_3x3(axes2[1, 1], M1_3x3_diag, '3×3 mean MI, incl. self-pairs  (dt=1)')

out_png2 = OUT_DIR / 'goal_pairwise_mi_incl_self.png'
out_pdf2 = OUT_DIR / 'goal_pairwise_mi_incl_self.pdf'
plt.savefig(out_png2, dpi=180, bbox_inches='tight')
plt.savefig(out_pdf2, bbox_inches='tight')
plt.close()
print(f"Saved: {out_pdf2}")

print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
