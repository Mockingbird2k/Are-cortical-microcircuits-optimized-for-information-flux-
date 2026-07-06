import numpy as np


def build_prebeck_125node_weight_matrix(
    rng: np.random.Generator,
    ns: int,
    ds: float,
    overall_density_ABC: float = 0.116,
    ne: int = 100,
    nt: int = 125,
    mu_logn: float = -0.702,
    sigma_logn: float = 0.9355,
    inhibitory_density_D: float = 0.116,
    wi_mode: str = "-t0",  # "-t0" or "-1"
):
    """
    Exact construction from Prebeck thesis Methods Section 2.7 + continuous skeleton temperature t0.
    Returns:
      W: (nt, nt) weight matrix with no self connections
      idx: dict with index arrays for skeleton, sea, inhibitory
      t0: temperature computed as mean weight of realized skeleton-internal edges (region A)

    Conventions:
      - Nodes 0..ns-1: skeleton (excitatory)
      - Nodes ns..ne-1: sea (excitatory)
      - Nodes ne..nt-1: inhibitory ("inter") nodes

    Regions (Fig. 2.7):
      A: skeleton -> skeleton (excluding diagonal)
      B: (skeleton+sea) -> sea
      C: (skeleton+sea) -> inhibitory
      D: inhibitory -> (skeleton+sea)
      E: inhibitory -> inhibitory  (no edges)
    """
    if nt != 125 or ne != 100 or (nt - ne) != 25:
        raise ValueError("This routine is defined for nt=125, ne=100, ni=25 exactly.")
    if not (0 < ns < ne):
        raise ValueError("ns must satisfy 0 < ns < ne.")
    if not (0.0 <= ds <= 1.0):
        raise ValueError("ds must be in [0,1].")

    ni = nt - ne
    nsea = ne - ns

    sk = np.arange(0, ns, dtype=int)
    sea = np.arange(ns, ne, dtype=int)
    inh = np.arange(ne, nt, dtype=int)

    W = np.zeros((nt, nt), dtype=float)

    # ---------- Excitatory candidate edges (regions A+B+C) ----------
    # Possible connections from excitatory nodes (0..ne-1) to:
    # - excitatory targets (0..ne-1) excluding self
    # - inhibitory targets (ne..nt-1)
    exc_sources = np.arange(0, ne, dtype=int)

    # Region A candidates: skeleton->skeleton excluding self
    A_candidates = [(i, j) for i in sk for j in sk if i != j]

    # Regions B and C candidates:
    # All excitatory sources to all excitatory targets excluding self, minus region A edges
    # plus all excitatory sources to all inhibitory targets
    A_set = set(A_candidates)

    BC_candidates = []
    # excitatory -> excitatory excluding self, excluding A
    for i in exc_sources:
        for j in exc_sources:
            if i == j:
                continue
            if (i in sk) and (j in sk):
                # this is region A, skip from BC
                continue
            BC_candidates.append((i, j))
    # excitatory -> inhibitory (region C)
    for i in exc_sources:
        for j in inh:
            BC_candidates.append((i, j))

    # Total possible edges in A+B+C
    total_possible_ABC = ne * (ne - 1) + ne * ni  # 12400
    # Number of realized excitatory edges in A+B+C
    n_realized_ABC = int(round(overall_density_ABC * total_possible_ABC))  # 1438

    # Draw weights w_ex (lognormal, Song fit)
    w_ex = rng.lognormal(mean=mu_logn, sigma=sigma_logn, size=n_realized_ABC)
    # Sort descending so we can take largest for region A
    w_ex_sorted = np.sort(w_ex)[::-1]

    # Number of realized skeleton edges in region A
    nA_possible = ns * (ns - 1)
    nA_realized = int(round(ds * nA_possible))

    if nA_realized > n_realized_ABC:
        raise ValueError("Requested skeleton edges exceed total excitatory edges. Reduce ds or adjust densities.")

    # Assign top weights to region A edges, randomly placed among A candidates
    if nA_realized > 0:
        A_edges = rng.choice(len(A_candidates), size=nA_realized, replace=False)
        A_edges = [A_candidates[k] for k in A_edges]
        A_weights = w_ex_sorted[:nA_realized]
        for (w, (i, j)) in zip(A_weights, A_edges):
            W[i, j] = w
    else:
        A_weights = np.array([], dtype=float)

    # Assign remaining excitatory weights to BC edges uniformly at random
    remaining_weights = w_ex_sorted[nA_realized:]
    nBC_realized = remaining_weights.size

    if nBC_realized > 0:
        BC_edges_idx = rng.choice(len(BC_candidates), size=nBC_realized, replace=False)
        BC_edges = [BC_candidates[k] for k in BC_edges_idx]
        for (w, (i, j)) in zip(remaining_weights, BC_edges):
            W[i, j] = w

    # Enforce no self-connections explicitly
    np.fill_diagonal(W, 0.0)

    # ---------- Temperature t0 (mean skeleton-internal realized weights) ----------
    if A_weights.size == 0:
        # If ds=0, t0 is undefined in the thesis logic. We forbid it for Goal 2.
        raise ValueError("ds must be > 0 for continuous skeleton temperature t0.")
    #t0 = float(np.mean(A_weights))
    t0 = 5.0 #just checking

    # ---------- Inhibitory outgoing edges (region D) ----------
    # Possible inhibitory -> excitatory targets (skeleton+sea): ni * ne = 2500
    D_candidates = [(i, j) for i in inh for j in exc_sources]

    nD_realized = int(round(inhibitory_density_D * (ni * ne)))

    if wi_mode not in ("-t0", "-1"):
        raise ValueError("wi_mode must be '-t0' or '-1'.")

    wi_value = (-t0) if wi_mode == "-t0" else (-1.0)

    if nD_realized > 0:
        D_edges_idx = rng.choice(len(D_candidates), size=nD_realized, replace=False)
        for k in D_edges_idx:
            i, j = D_candidates[k]
            W[i, j] = wi_value

    # Region E inhibitory->inhibitory is kept at 0 by construction
    # Enforce no self-connections again
    np.fill_diagonal(W, 0.0)

    idx = {"skeleton": sk, "sea": sea, "inhibitory": inh}
    return W, idx, t0
