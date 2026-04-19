"""
Checkerboard visualisation of h-strategies and ∞-strategies.

For two strategies (k1, k2), the joint outcome matrix delta(i,j) = 1 iff
both players simultaneously select a black hat in configuration (i,j).
Rendered as a black-and-white image, black tiles correspond to winning pairs.

Figures 2, 3 and 4 of the paper are produced by this module.
"""

from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt


# ── Hat configurations ─────────────────────────────────────────────────────────

def generate_configs(h):
    """
    All 2^h binary tuples of length h in lexicographic (MSB-first) order.
    This matches the paper's ordering a_1 < ... < a_{2^h} in {0,1}^h, so
    strategies can be used directly with their paper indices (1-based → 0-based).
    Config index i (0-based) corresponds to real x in [i/2^h, (i+1)/2^h).
    """
    return [tuple((i >> (h - 1 - j)) & 1 for j in range(h)) for i in range(1 << h)]


def config_index_map(h):
    return {c: i for i, c in enumerate(generate_configs(h))}


# ── FBH strategy ───────────────────────────────────────────────────────────────

def fbh_strategy(h):
    """
    First-black-hat h-strategy.
    strat[j] = 1-based index of the first black hat in config j.
    Falls back to index 1 for the all-white configuration.
    """
    result = []
    for config in generate_configs(h):
        try:
            result.append(config.index(1) + 1)
        except ValueError:
            result.append(1)
    return result


# ── Joint outcome matrix (finite h) ───────────────────────────────────────────

def outcome_matrix(strat_a, strat_b, h):
    """
    Compute the 2^h x 2^h joint outcome matrix for strategies strat_a, strat_b.
    Entry matrix[j, i] = 1 iff both players win when A has stack i and B has stack j.
    Axes follow the paper convention: column = stack A (left to right),
                                      row    = stack B (bottom to top).
    """
    configs = generate_configs(h)
    cidx    = config_index_map(h)
    n       = 1 << h
    matrix  = np.zeros((n, n), dtype=np.int8)
    for i, stack_a in enumerate(configs):
        for j, stack_b in enumerate(configs):
            ga = strat_a[cidx[stack_b]]   # A observes B's stack -> index into A's stack
            gb = strat_b[cidx[stack_a]]   # B observes A's stack -> index into B's stack
            matrix[j, i] = stack_a[ga - 1] & stack_b[gb - 1]
    return matrix


def winning_probability(matrix):
    return int(matrix.sum()), matrix.size


# ── Bit extraction ─────────────────────────────────────────────────────────────

def _get_bits(vals, bit_pos):
    """
    Extract bit bit_pos (1-based) of all values in vals in [0, 1):
        floor(2^{bit_pos} * val) mod 2.

    Uses np.ldexp to avoid integer overflow for large bit_pos.
    Reliable for bit_pos <= 52 (float64 mantissa limit).
    Returns zeros beyond that, which is correct for any resolution <= 2^52
    since sample points carry no information in those high bits.
    """
    if bit_pos > 52:
        return np.zeros(len(vals), dtype=np.int64)
    return np.ldexp(vals, bit_pos).astype(np.int64) & 1


# ── Non-monochromatic stopping set ─────────────────────────────────────────────

def non_monochromatic_set(h):
    """Config indices that are neither all-0 nor all-1."""
    configs = generate_configs(h)
    return frozenset(
        i for i, c in enumerate(configs)
        if not (all(b == 0 for b in c) or all(b == 1 for b in c))
    )


# ── Exact winning probability for recursive ∞-strategy ────────────────────────

def recursive_winning_probability(strat_a, strat_b, h):
    """
    Exact winning probability of the recursive ∞-strategy built from finite
    h-strategies strat_a and strat_b, as a Fraction.

    The recursive strategy skips monochromatic h-blocks until both players
    simultaneously observe a non-monochromatic block, then applies the finite
    h-strategy.  The exact probability satisfies the fixed-point equation

        P = alpha * P  +  gamma * p_nn  +  beta * (1/4)

    where
        alpha = P(both players' block is mono)     = 4 / 4^h
        beta  = P(exactly one block is mono)       = 4*(2^h-2) / 4^h
        gamma = P(both blocks are non-mono)        = (2^h-2)^2 / 4^h
        p_nn  = conditional win prob when both non-mono  (brute-force)

    Solving gives P = (gamma * p_nn + beta/4) / (1 - alpha).

    The 1/4 coefficient on beta comes from the paper's case analysis:
    when exactly one player's block is mono, the two equally likely
    sub-cases (all-black / all-white) average to a win probability of 1/4.

    Returns
    -------
    Fraction
        Exact rational winning probability.
    """
    configs  = generate_configs(h)
    cidx     = config_index_map(h)
    n        = 1 << h   # 2^h

    # Non-monochromatic configurations (neither all-0 nor all-1)
    non_mono = [c for c in configs
                if not (all(b == 0 for b in c) or all(b == 1 for b in c))]
    n_nm = len(non_mono)   # 2^h - 2

    # ── Brute-force: count wins when BOTH players see a non-mono block ─────────
    # This directly mirrors the approach in the brute-force solver:
    #   guess_a = strat_a[config_idx[stack_b]]
    #   guess_b = strat_b[config_idx[stack_a]]
    #   win iff stack_a[guess_a-1] == 1 and stack_b[guess_b-1] == 1
    wins_nn = 0
    for stack_a in non_mono:
        for stack_b in non_mono:
            ga = strat_a[cidx[stack_b]]
            gb = strat_b[cidx[stack_a]]
            wins_nn += stack_a[ga - 1] & stack_b[gb - 1]

    p_nn = Fraction(wins_nn, n_nm * n_nm)

    # ── Per-block probabilities (over 4^h equally likely stack pairs) ──────────
    n2    = n * n
    alpha = Fraction(4,            n2)   # P(both mono)
    beta  = Fraction(4 * n_nm,     n2)   # P(exactly one mono)
    gamma = Fraction(n_nm * n_nm,  n2)   # P(both non-mono)

    # Closed-form solution of the fixed-point equation
    P = (gamma * p_nn + beta * Fraction(1, 4)) / (1 - alpha)
    return P


# ── Recursive ∞-strategy (vectorised) ─────────────────────────────────────────

def _recursive_strategy_1d(vals, strat, h, stopping, max_blocks=15):
    """
    Vectorised application of the recursive ∞-strategy to vals in [0, 1).

    Scans vals in consecutive blocks of h bits assembled in MSB-first order,
    consistent with lexicographic config ordering. Stops at the first
    non-monochromatic block and returns  m*h + strat[block_index]  as the
    guessed hat index (1-based).

    max_blocks=15 is more than sufficient: for h >= 3 the probability of
    needing block m is <= (2/2^h)^m <= (1/4)^m < 10^{-6} for m >= 10.
    """
    n            = len(vals)
    strat        = np.asarray(strat, dtype=np.int64)
    stopping_arr = np.array(sorted(stopping), dtype=np.int64)
    guess        = np.ones(n, dtype=np.int64)
    found        = np.zeros(n, dtype=bool)

    for m in range(max_blocks):
        if found.all():
            break
        # Assemble block-m config index in MSB-first order:
        # bit (m*h + k + 1) of val contributes 2^(h-1-k) to the block index.
        # This ensures that val in [i/2^h, (i+1)/2^h) gives block_idx = i.
        block_idx = np.zeros(n, dtype=np.int64)
        for k in range(h):
            bit_pos = m * h + k + 1
            bits    = _get_bits(vals, bit_pos)
            block_idx += bits << (h - 1 - k)      # MSB-first: key fix

        in_stop = np.isin(block_idx, stopping_arr)
        update  = in_stop & ~found
        if update.any():
            guess[update] = m * h + strat[block_idx[update]]
            found |= update

    return guess


def recursive_density(strat_a, strat_b, h, resolution=512):
    """
    Sample the joint outcome density of the recursive ∞-strategy on a
    resolution x resolution grid.

    strat_a, strat_b : base h-strategies in paper (lexicographic) order.
                       Pass the same array for symmetric strategies (e.g. S5).

    Returns a (resolution, resolution) int8 array in {0, 1} with:
        axis 1 (columns) = A's stack x, left to right
        axis 0 (rows)    = B's stack y, bottom to top   [use origin='lower']
    """
    stopping = non_monochromatic_set(h)
    y_vals   = (np.arange(resolution) + 0.5) / resolution   # one value per row
    x_vals   = (np.arange(resolution) + 0.5) / resolution   # one value per column

    # Player A observes y -> guess ga(y) = index into A's own x-stack.
    # Player B observes x -> guess gb(x) = index into B's own y-stack.
    # ga depends only on y (rows); gb only on x (columns).
    ga = _recursive_strategy_1d(y_vals, strat_a, h, stopping)   # shape (resolution,)
    gb = _recursive_strategy_1d(x_vals, strat_b, h, stopping)   # shape (resolution,)

    # density[iy, ix] = bit_{ga[iy]}(x[ix])  &  bit_{gb[ix]}(y[iy])
    bit_x = np.zeros((resolution, resolution), dtype=np.int8)
    for ga_val in np.unique(ga):
        rows   = np.where(ga == ga_val)[0]
        x_bits = _get_bits(x_vals, int(ga_val)).astype(np.int8)
        bit_x[np.ix_(rows, np.arange(resolution))] = x_bits[np.newaxis, :]

    bit_y = np.zeros((resolution, resolution), dtype=np.int8)
    for gb_val in np.unique(gb):
        cols   = np.where(gb == gb_val)[0]
        y_bits = _get_bits(y_vals, int(gb_val)).astype(np.int8)
        bit_y[np.ix_(np.arange(resolution), cols)] = y_bits[:, np.newaxis]

    return (bit_x & bit_y).astype(np.int8)


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_matrix(matrix, title="", ax=None, show_prob=True):
    """Render a joint outcome matrix as a black-and-white checkerboard."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="binary", origin="lower",
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if show_prob:
        w, t = winning_probability(matrix)
        title += f"\nwinning prob. {w}/{t}"
    ax.set_title(title, fontsize=9)
    return ax


def plot_recursive_strategy(strat_a, strat_b, h, title="",
                             resolution=512, ax=None, save_path=None):
    """
    Render the recursive ∞-strategy density with exact winning probability.

    The probability displayed in the title is computed exactly via brute-force
    enumeration over non-monochromatic config pairs (see
    recursive_winning_probability), not estimated from the rendered grid.
    """
    density = recursive_density(strat_a, strat_b, h, resolution)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    # Exact winning probability (rational arithmetic, no sampling)
    prob = recursive_winning_probability(strat_a, strat_b, h)

    ax.imshow(density, cmap="binary", origin="lower",
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title}\nwinning prob. = {prob}", fontsize=9)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return ax


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # All strategies are given in paper (lexicographic / MSB-first) order,
    # matching the paper's a_1, ..., a_{2^h} indexing directly.
    S_FBH_3 = fbh_strategy(3)            # computed: [1, 3, 2, 2, 1, 1, 1, 1]

    S3_3_A  = [1, 3, 2, 2, 1, 3, 1, 1]  # k1(a_j), j=1..8, paper §2.4
    S3_3_B  = [1, 3, 2, 3, 1, 1, 2, 1]  # k2(a_j), j=1..8

    S5_5    = [2,3,2,3,5,5,5,5, 4,3,2,3,5,5,5,5,
               1,3,1,3,1,5,1,1, 1,3,1,3,1,4,1,5]   # S5,5(a_j), j=1..32

    # ── Sanity checks ──────────────────────────────────────────────────────────
    print("Verification against paper values:")
    w, t = winning_probability(outcome_matrix(S_FBH_3, S_FBH_3, 3))
    print(f"  FBH   h=3 (finite) : {w}/{t}  (paper: 21/64)")
    w, t = winning_probability(outcome_matrix(S3_3_A, S3_3_B, 3))
    print(f"  S3,3  h=3 (finite) : {w}/{t}  (paper: 22/64)")
    w, t = winning_probability(outcome_matrix(S5_5, S5_5, 5))
    print(f"  S5,5  h=5 (finite) : {w}/{t}  (paper: 358/1024)")

    p = recursive_winning_probability(S_FBH_3, S_FBH_3, 3)
    print(f"  FBH   h=3 (recursive ∞) : {p} = {float(p):.6f}  (paper: 1/3)")
    p = recursive_winning_probability(S3_3_A, S3_3_B, 3)
    print(f"  S3    h=3 (recursive ∞) : {p} = {float(p):.6f}  (paper: 7/20)")
    p = recursive_winning_probability(S5_5, S5_5, 5)
    print(f"  S5    h=5 (recursive ∞) : {p} = {float(p):.6f}  (paper: 7/20)")

    # ── Figure 2: FBH and S_{3,3} for h = 3 ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_matrix(outcome_matrix(S_FBH_3, S_FBH_3, 3),
                title=(r"$\delta^{\mathcal{S}_{\mathrm{FBH},3},"
                        r"\mathcal{S}_{\mathrm{FBH},3}}$"), ax=axes[0])
    plot_matrix(outcome_matrix(S3_3_A, S3_3_B, 3),
                title=r"$\delta^{\mathcal{S}_{3,3},\mathcal{S}_{3,3}}$",
                ax=axes[1])
    fig.suptitle(r"Joint outcome matrices, $h=3$, $n=2$", fontsize=11)
    plt.tight_layout()
    plt.savefig("figure2.pdf", dpi=200, bbox_inches="tight")
    print("Saved figure2.pdf")

    # ── Figure 3: S_{3,3} finite + S_3 recursive ∞-strategy ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_matrix(outcome_matrix(S3_3_A, S3_3_B, 3),
                title=r"3-strategy $\mathcal{S}_{3,3}$", ax=axes[0])
    plot_recursive_strategy(S3_3_A, S3_3_B, h=3,
                            title=r"Strategy $\mathcal{S}_3$",
                            resolution=512, ax=axes[1])
    plt.tight_layout()
    plt.savefig("figure3.pdf", dpi=200, bbox_inches="tight")
    print("Saved figure3.pdf")

    # ── Figure 4: S_{5,5} finite + S_5 recursive ∞-strategy ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_matrix(outcome_matrix(S5_5, S5_5, 5),
                title=r"5-strategy $\mathcal{S}_{5,5}$", ax=axes[0])
    plot_recursive_strategy(S5_5, S5_5, h=5,
                            title=r"Strategy $\mathcal{S}_5$",
                            resolution=512, ax=axes[1])
    plt.tight_layout()
    plt.savefig("figure4.pdf", dpi=200, bbox_inches="tight")
    print("Saved figure4.pdf")