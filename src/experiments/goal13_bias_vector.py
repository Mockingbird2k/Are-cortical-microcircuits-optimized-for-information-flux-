"""
Goal 13: Hill-climbing optimisation of individual bias vector for isolated skeleton.

Each of the 10 skeleton neurons receives its own constant bias v_i.
We optimise v = (v_1,...,v_10) via hill-climbing to maximise total motif MI
(intra + inter sub-motif MI) — same metric as Goals 3 & 12.

4 restart strategies:
  0: init at v_opt (best global scalar from Goal 12 sweep, done inline)
  1: init at p=0.5 bias per neuron (supervisor's hypothesis)
  2: init at zero
  3: random init

Reference line: best Prebeck embedded MI (Goal 3, ds=35%).
Reference line: best Goal 12 scalar-bias MI.

Usage (from project root):
    python src/experiments/goal13_bias_vector.py
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

# ══════════════════════════════════════════════════════════════════════════════
# Configuration  — tune these for your PC
# ══════════════════════════════════════════════════════════════════════════════
NS    = 10
DS    = 0.35
SEED  = 42

# Hill-climbing evaluation budget
T_HC    = 100_000   # timesteps per MI evaluation during hill-climbing
                    # (use 50_000 if you want faster runs, 200_000 for more precision)
T_FINAL = 1_000_000 # timesteps for final evaluation of best solution
BURN_IN = 5_000     # burn-in discarded before recording

# Hill-climbing hyperparameters
N_ITER      = 500   # perturbation steps per restart
PERTURB_STD = 0.3   # Gaussian std for perturbation (smaller = finer search)
                    # after convergence, will automatically halve and continue

# Goal 12 scalar sweep — run inline to find v_opt
# (if you already have data/goal12/ with saved results, script will load them)
N_SCALAR_STEPS = 41  # number of v values in scalar sweep (-10 to +10)
V_RANGE        = (-10.0, 10.0)

TRIPLETS = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
OUT_DIR  = Path('data/goal13')
ensure_dir(OUT_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Build isolated skeleton
# ══════════════════════════════════════════════════════════════════════════════
rng_build = np.random.default_rng(SEED)
W_full, idx, t0 = build_prebeck_125node_weight_matrix(
    rng=rng_build, ns=NS, ds=DS, wi_mode='-t0'
)
sk   = idx['skeleton'].astype(int)
ns   = len(sk)
W_sk = W_full[np.ix_(sk, sk)]   # (10, 10) isolated skeleton weights

print(f"Isolated skeleton: n={ns}, t0={t0:.4f}")
print(f"W_sk nonzero: {(W_sk!=0).sum()},  mean nonzero: {W_sk[W_sk!=0].mean():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Core simulation & MI functions
# ══════════════════════════════════════════════════════════════════════════════
def sim(W, t0, v_bias, T, burn, seed):
    """
    Boltzmann dynamics on isolated skeleton with constant per-neuron bias.
    z_i = (sum_j s_j * W[j,i] + v_i) / t0
    Returns (T, n) uint8 binary states.
    """
    n   = W.shape[0]
    rng = np.random.default_rng(seed)
    s   = rng.integers(0, 2, size=n, dtype=np.uint8)
    bs  = v_bias / t0   # pre-scale bias once

    for _ in range(burn):
        z = s.astype(np.float64) @ W / t0 + bs
        s = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.uint8)

    out = np.zeros((T, n), dtype=np.uint8)
    for t in range(T):
        z    = s.astype(np.float64) @ W / t0 + bs
        s    = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.uint8)
        out[t] = s
    return out


def motif_mi(states):
    """
    Compute motif MI (intra + inter sub-motif) from (T, ns) binary states.
    Returns (intra, inter) in units of 1e-3 bits — same as Goals 3 & 12.
    """
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


def eval_mi(v_bias, seed):
    """Evaluate motif MI for a given bias vector. Returns total = intra + inter."""
    states = sim(W_sk, t0, v_bias, T_HC, BURN_IN, seed)
    intra, inter = motif_mi(states)
    return intra + inter, intra, inter

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load Prebeck reference (Goal 3, ds=35%)
# ══════════════════════════════════════════════════════════════════════════════
try:
    pb_intra = float(np.load('data/goal3/goal3_ds35_intra.npy').max())
    pb_inter = float(np.load('data/goal3/goal3_ds35_inter.npy').max())
    pb_total = pb_intra + pb_inter
    print(f"\nPrebeck embedded reference: total={pb_total:.4f}  "
          f"(intra={pb_intra:.4f}, inter={pb_inter:.4f})  ×10⁻³ bits")
except FileNotFoundError:
    pb_total = None
    print("Goal 3 data not found — Prebeck reference line will be omitted.")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Goal 12 scalar sweep to find v_opt
# ══════════════════════════════════════════════════════════════════════════════
g12_path_total = Path('data/goal12/mi_total.npy')
g12_path_v     = Path('data/goal12/v_values.npy')

if g12_path_total.exists() and g12_path_v.exists():
    print("\nLoading Goal 12 scalar sweep from data/goal12/ ...")
    g12_v      = np.load(g12_path_v)
    g12_total  = np.load(g12_path_total)
else:
    print(f"\nRunning Goal 12 scalar sweep  "
          f"({N_SCALAR_STEPS} values, T={T_HC}) ...")
    ensure_dir(Path('data/goal12'))
    g12_v     = np.linspace(V_RANGE[0], V_RANGE[1], N_SCALAR_STEPS)
    g12_intra = np.zeros(N_SCALAR_STEPS)
    g12_inter = np.zeros(N_SCALAR_STEPS)
    for k, v_scalar in enumerate(g12_v):
        v_vec = np.full(ns, v_scalar)
        tot, intr, intr2 = eval_mi(v_vec, seed=SEED + k)
        g12_intra[k] = intr
        g12_inter[k] = intr2
        print(f"  v={v_scalar:+6.2f}  MI_total={tot:.4f}")
    g12_total = g12_intra + g12_inter
    np.save(g12_path_v,     g12_v)
    np.save(g12_path_total, g12_total)
    np.save('data/goal12/mi_intra.npy', g12_intra)
    np.save('data/goal12/mi_inter.npy', g12_inter)

best_g12_idx = int(np.argmax(g12_total))
v_opt_scalar = float(g12_v[best_g12_idx])
mi_g12_best  = float(g12_total[best_g12_idx])
print(f"Goal 12 best: v_opt = {v_opt_scalar:.4f},  MI_total = {mi_g12_best:.4f}  ×10⁻³ bits")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Compute p=0.5 bias per neuron (supervisor's hypothesis)
#
# For each neuron i: we want p_i = 0.5  ⟹  sigmoid(z_i/t0) = 0.5  ⟹  z_i = 0
# z_i = sum_j p_j * W[j,i] + v_i  (mean-field approximation)
# ⟹  v_i = -sum_j p_j * W[j,i]
# Solve self-consistently: iterate p = sigmoid((W^T p + v) / t0)
# with v chosen so that fixed point gives p_i = 0.5 for all i.
# Simplest approximation: start with p = 0.5, compute v = -W_sk.T @ p
# ══════════════════════════════════════════════════════════════════════════════
p_half   = np.full(ns, 0.5)
v_p_half = -(W_sk.T @ p_half)   # mean-field bias to achieve p=0.5
print(f"\nP=0.5 hypothesis bias vector (mean-field):")
print(f"  {v_p_half.round(4)}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Hill-climbing with 4 restart strategies
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nHill-climbing: 4 restarts × {N_ITER} iters")
print(f"  T_HC={T_HC}   BURN_IN={BURN_IN}   PERTURB_STD={PERTURB_STD}\n")

rng_hc = np.random.default_rng(SEED + 100)

# 4 initialisation strategies
v_inits = [
    ('v_opt scalar',     np.full(ns, v_opt_scalar)),
    ('p=0.5 mean-field', v_p_half.copy()),
    ('zero',             np.zeros(ns)),
    ('random ±5',        rng_hc.uniform(-5.0, 5.0, size=ns)),
]

best_tot_g   = -np.inf
best_v_g     = np.zeros(ns)
all_hist     = []
restart_labs = []

for restart, (label, v_init) in enumerate(v_inits):
    v              = v_init.copy()
    tot, intr, itr = eval_mi(v, seed=SEED + restart)
    cur            = tot
    hist           = [cur]
    restart_labs.append(label)

    print(f"  Restart {restart} [{label}]: init total={cur:.4f}")

    perturb_std = PERTURB_STD

    for it in range(N_ITER):
        # Adaptive perturbation: halve std halfway through if not improving
        if it == N_ITER // 2:
            perturb_std *= 0.5
            print(f"    [restart {restart}] halving perturb_std → {perturb_std:.3f}")

        v_new          = v.copy()
        # Occasionally (20% of steps) perturb 2 neurons simultaneously
        k_perturb = 2 if rng_hc.random() < 0.2 else 1
        neurons   = rng_hc.choice(ns, size=k_perturb, replace=False)
        v_new[neurons] += rng_hc.normal(0.0, perturb_std, size=k_perturb)

        tot_new, intr_new, itr_new = eval_mi(v_new, seed=SEED + restart * 100000 + it)

        if tot_new > cur:
            v, cur        = v_new, tot_new
            intr, itr     = intr_new, itr_new

        hist.append(cur)

        if (it + 1) % 100 == 0:
            print(f"    iter {it+1:4d}: total={cur:.4f}  "
                  f"intra={intr:.4f}  inter={itr:.4f}")

    all_hist.append(np.array(hist))

    if cur > best_tot_g:
        best_tot_g = cur
        best_v_g   = v.copy()

    print(f"  Restart {restart} done: total={cur:.4f}  "
          f"v={v.round(3)}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Final high-T evaluation of best bias vector
# ══════════════════════════════════════════════════════════════════════════════
print(f"Running final evaluation at T={T_FINAL} ...")
states_final            = sim(W_sk, t0, best_v_g, T_FINAL, BURN_IN, SEED + 9999)
best_intra_f, best_inter_f = motif_mi(states_final)
best_tot_f              = best_intra_f + best_inter_f
p1_opt                  = states_final.mean(axis=0)

print(f"\n{'='*60}")
print(f"FINAL RESULT (T={T_FINAL})")
print(f"  MI_total = {best_tot_f:.4f}  "
      f"(intra={best_intra_f:.4f}, inter={best_inter_f:.4f})  ×10⁻³ bits")
if pb_total:
    print(f"  Prebeck reference = {pb_total:.4f}  "
          f"→ gap = {pb_total - best_tot_f:.4f}")
print(f"  Goal 12 best      = {mi_g12_best:.4f}  "
      f"→ improvement = {best_tot_f - mi_g12_best:.4f}")
print(f"\nOptimal bias vector:")
for i in range(ns):
    print(f"  core[{i}]: v={best_v_g[i]:+7.4f}   ⟨p₁⟩={p1_opt[i]:.4f}")

print(f"\nSupervisor hypothesis check — are firing probs near 0.5?")
print(f"  mean |p1 - 0.5| = {np.abs(p1_opt - 0.5).mean():.4f}")

# Save
np.save(OUT_DIR / 'best_bias_vector.npy',        best_v_g)
np.save(OUT_DIR / 'optimal_p1.npy',              p1_opt)
np.save(OUT_DIR / 'hill_climbing_histories.npy',  np.array(all_hist, dtype=object))
np.save(OUT_DIR / 'goal12_v_values.npy',          g12_v)
np.save(OUT_DIR / 'goal12_mi_total.npy',          g12_total)
print(f"\nAll results saved to {OUT_DIR}/")

# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35)

ax_g12   = fig.add_subplot(gs[0, 0])   # Goal 12 scalar sweep
ax_conv  = fig.add_subplot(gs[0, 1:])  # Hill-climbing convergence (wide)
ax_bias  = fig.add_subplot(gs[1, 0])   # Optimal bias vector
ax_p1    = fig.add_subplot(gs[1, 1])   # Firing probs
ax_scat  = fig.add_subplot(gs[1, 2])   # bias vs p1 scatter

colors = ['#2166ac', '#d6604d', '#4dac26', '#984ea3']

# ── Panel: Goal 12 scalar sweep ───────────────────────────────────────────────
ax_g12.plot(g12_v, g12_total, 'o-', color='#2166ac', lw=2, ms=4)
ax_g12.axvline(v_opt_scalar, color='red', lw=1.2, ls='--',
               label=f'v_opt={v_opt_scalar:.2f}')
if pb_total:
    ax_g12.axhline(pb_total, color='black', lw=1.2, ls='--',
                   label=f'Prebeck ({pb_total:.1f})')
ax_g12.set_xlabel('global bias  v', fontsize=10)
ax_g12.set_ylabel('MI_total  (×10⁻³ bits)', fontsize=10)
ax_g12.set_title('Goal 12 — scalar bias sweep', fontsize=10)
ax_g12.legend(fontsize=8); ax_g12.grid(True, alpha=0.3)

# ── Panel: Hill-climbing convergence ─────────────────────────────────────────
for r, (hist, lab) in enumerate(zip(all_hist, restart_labs)):
    ax_conv.plot(hist, color=colors[r], lw=1.5, label=f'restart {r}: {lab}')
if pb_total:
    ax_conv.axhline(pb_total, color='black', lw=1.5, ls='--',
                    label=f'Prebeck embedded ({pb_total:.1f})')
ax_conv.axhline(mi_g12_best, color='gray', lw=1.2, ls=':',
                label=f'Goal 12 best ({mi_g12_best:.1f})')
ax_conv.set_xlabel('iteration', fontsize=10)
ax_conv.set_ylabel('MI_total  (×10⁻³ bits)', fontsize=10)
ax_conv.set_title('Goal 13 — hill-climbing convergence  (4 restart strategies)', fontsize=10)
ax_conv.legend(fontsize=8); ax_conv.grid(True, alpha=0.3)

# ── Panel: Optimal bias vector ────────────────────────────────────────────────
bars = ax_bias.bar(range(ns), best_v_g, color='#2166ac', alpha=0.8,
                    edgecolor='black', lw=0.5)
ax_bias.axhline(0, color='black', lw=0.8)
ax_bias.axhline(v_opt_scalar, color='red', lw=1.0, ls='--',
                label=f'Goal 12 v_opt={v_opt_scalar:.2f}')
ax_bias.set_xlabel('core neuron index', fontsize=10)
ax_bias.set_ylabel('optimal bias  vᵢ', fontsize=10)
ax_bias.set_title('Optimal bias vector', fontsize=10)
ax_bias.set_xticks(range(ns))
ax_bias.legend(fontsize=8); ax_bias.grid(True, alpha=0.3, axis='y')

# ── Panel: Firing probs ───────────────────────────────────────────────────────
ax_p1.bar(range(ns), p1_opt, color='#d6604d', alpha=0.8,
           edgecolor='black', lw=0.5)
ax_p1.axhline(0.5, color='black', lw=1.5, ls='--', label='p=0.5 (max entropy)')
ax_p1.axhline(DS,  color='gray',  lw=1.2, ls=':',
              label=f'ds={DS:.0%} (Prebeck target)')
ax_p1.set_xlabel('core neuron index', fontsize=10)
ax_p1.set_ylabel('⟨p₁⟩  at optimal bias', fontsize=10)
ax_p1.set_title('Firing probs at optimal bias', fontsize=10)
ax_p1.set_ylim(0, 1); ax_p1.set_xticks(range(ns))
ax_p1.legend(fontsize=8); ax_p1.grid(True, alpha=0.3, axis='y')

# ── Panel: bias vs p1 scatter ─────────────────────────────────────────────────
ax_scat.scatter(best_v_g, p1_opt, c='#2166ac', s=60, zorder=3)
for i in range(ns):
    ax_scat.annotate(str(i), (best_v_g[i], p1_opt[i]),
                     textcoords='offset points', xytext=(4, 3), fontsize=7.5)
# Theoretical sigmoid curve (no recurrent input)
v_line = np.linspace(best_v_g.min() - 1, best_v_g.max() + 1, 200)
ax_scat.plot(v_line, 1.0 / (1.0 + np.exp(-v_line / t0)),
             color='gray', lw=1.2, ls='--', label='sigmoid(v/t₀)  [no recurrence]')
ax_scat.axhline(0.5, color='black', lw=0.8, ls=':')
ax_scat.axvline(0,   color='black', lw=0.8, ls=':')
ax_scat.set_xlabel('optimal bias  vᵢ', fontsize=10)
ax_scat.set_ylabel('⟨p₁⟩', fontsize=10)
ax_scat.set_title('Bias vs firing prob  (sigmoid reference)', fontsize=10)
ax_scat.legend(fontsize=8); ax_scat.grid(True, alpha=0.3)

fig.suptitle(
    f'Goal 13 — Individual bias vector optimisation  '
    f'(isolated skeleton, ds={DS:.0%}, T_HC={T_HC}, T_final={T_FINAL})\n'
    f'Best MI_total = {best_tot_f:.4f} ×10⁻³ bits'
    + (f'   [Prebeck = {pb_total:.4f}]' if pb_total else ''),
    fontsize=11
)

out_png = OUT_DIR / 'goal13_bias_vector.png'
out_pdf = OUT_DIR / 'goal13_bias_vector.pdf'
plt.savefig(out_png, dpi=180, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
