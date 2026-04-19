"""
Interactive analysis of user-defined h-strategies for Levine's hat problem.

For a user-supplied pair of h-strategies (strat_A, strat_B), this script:
  1. Displays the finite joint outcome matrix with exact winning probability.
  2. Displays the recursive ∞-strategy checkerboard with exact winning
     probability (computed via brute-force on non-monochromatic config pairs,
     no sampling).

Run:
    python user_strategy.py

You will be prompted for h, then for each strategy as a space-separated list
of integers in {1, ..., h} of length 2^h, in lexicographic (MSB-first) order:
    a_1=(0,...,0,0), a_2=(0,...,0,1), ..., a_{2^h}=(1,...,1,1).
"""

from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt


# ── Core helpers (identical to checkerboard_viz.py) ───────────────────────────

def generate_configs(h):
    """All 2^h binary tuples of length h in MSB-first lexicographic order."""
    return [tuple((i >> (h - 1 - j)) & 1 for j in range(h)) for i in range(1 << h)]


def config_index_map(h):
    return {c: i for i, c in enumerate(generate_configs(h))}


def outcome_matrix(strat_a, strat_b, h):
    """
    Exact 2^h × 2^h joint outcome matrix.
    Entry [j, i] = 1 iff both players win when A has stack i and B has stack j.
    """
    configs = generate_configs(h)
    cidx    = config_index_map(h)
    n       = 1 << h
    matrix  = np.zeros((n, n), dtype=np.int8)
    for i, stack_a in enumerate(configs):
        for j, stack_b in enumerate(configs):
            ga = strat_a[cidx[stack_b]]
            gb = strat_b[cidx[stack_a]]
            matrix[j, i] = stack_a[ga - 1] & stack_b[gb - 1]
    return matrix


def recursive_winning_probability(strat_a, strat_b, h):
    """
    Exact winning probability of the recursive ∞-strategy as a Fraction.

    Brute-forces wins over all non-monochromatic config pairs, then solves
    the fixed-point equation from the paper:

        P = alpha·P + gamma·p_nn + beta·(1/4)
        => P = (gamma·p_nn + beta/4) / (1 - alpha)

    where alpha / beta / gamma are the rational probabilities that both,
    exactly one, or neither player's h-block is monochromatic.
    """
    configs  = generate_configs(h)
    cidx     = config_index_map(h)
    n        = 1 << h

    non_mono = [c for c in configs
                if not (all(b == 0 for b in c) or all(b == 1 for b in c))]
    n_nm = len(non_mono)  # 2^h - 2

    if n_nm == 0:
        # h = 1: every config is monochromatic, strategy never terminates usefully
        return Fraction(0)

    wins_nn = 0
    for stack_a in non_mono:
        for stack_b in non_mono:
            ga = strat_a[cidx[stack_b]]
            gb = strat_b[cidx[stack_a]]
            wins_nn += stack_a[ga - 1] & stack_b[gb - 1]

    p_nn  = Fraction(wins_nn, n_nm * n_nm)

    n2    = n * n
    alpha = Fraction(4,           n2)
    beta  = Fraction(4 * n_nm,    n2)
    gamma = Fraction(n_nm * n_nm, n2)

    return (gamma * p_nn + beta * Fraction(1, 4)) / (1 - alpha)


def _get_bits(vals, bit_pos):
    if bit_pos > 52:
        return np.zeros(len(vals), dtype=np.int64)
    return np.ldexp(vals, bit_pos).astype(np.int64) & 1


def non_monochromatic_set(h):
    configs = generate_configs(h)
    return frozenset(
        i for i, c in enumerate(configs)
        if not (all(b == 0 for b in c) or all(b == 1 for b in c))
    )


def _recursive_strategy_1d(vals, strat, h, stopping, max_blocks=15):
    n            = len(vals)
    strat        = np.asarray(strat, dtype=np.int64)
    stopping_arr = np.array(sorted(stopping), dtype=np.int64)
    guess        = np.ones(n, dtype=np.int64)
    found        = np.zeros(n, dtype=bool)
    for m in range(max_blocks):
        if found.all():
            break
        block_idx = np.zeros(n, dtype=np.int64)
        for k in range(h):
            bits = _get_bits(vals, m * h + k + 1)
            block_idx += bits << (h - 1 - k)
        in_stop = np.isin(block_idx, stopping_arr)
        update  = in_stop & ~found
        if update.any():
            guess[update] = m * h + strat[block_idx[update]]
            found |= update
    return guess


def recursive_density(strat_a, strat_b, h, resolution=512):
    """512×512 pixel approximation of the ∞-strategy checkerboard (for display only)."""
    stopping = non_monochromatic_set(h)
    y_vals   = (np.arange(resolution) + 0.5) / resolution
    x_vals   = (np.arange(resolution) + 0.5) / resolution

    ga = _recursive_strategy_1d(y_vals, strat_a, h, stopping)
    gb = _recursive_strategy_1d(x_vals, strat_b, h, stopping)

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


# ── Input helpers ──────────────────────────────────────────────────────────────

def _prompt_h():
    """Ask the user for block size h (integer >= 2)."""
    while True:
        raw = input("  Block size h (integer >= 2): ").strip()
        try:
            h = int(raw)
            if h >= 2:
                return h
            print("  h must be at least 2.")
        except ValueError:
            print("  Please enter a valid integer.")


def _prompt_strategy(label, h):
    """
    Ask the user for a strategy as 2^h space-separated integers in {1,...,h}.
    Configs are listed in MSB-first lexicographic order:
        a_1 = (0,...,0), a_2 = (0,...,0,1), ..., a_{2^h} = (1,...,1).
    """
    n      = 1 << h
    configs = generate_configs(h)

    print(f"\n  Strategy {label}: enter {n} integers in {{1,...,{h}}},")
    print(f"  one per config in MSB-first lex order.")
    print(f"  Configs: {' | '.join(''.join(map(str, c)) for c in configs)}")

    while True:
        raw = input(f"  {label}: ").strip()
        try:
            vals = list(map(int, raw.split()))
        except ValueError:
            print("  Could not parse integers — please try again.")
            continue

        if len(vals) != n:
            print(f"  Expected {n} values, got {len(vals)} — please try again.")
            continue

        if not all(1 <= v <= h for v in vals):
            print(f"  All values must be in {{1,...,{h}}} — please try again.")
            continue

        return vals


def _print_config_table(h):
    """Print a numbered table of all configs so the user can verify ordering."""
    configs = generate_configs(h)
    print(f"\n  Config table for h={h}  (MSB-first, 0=white, 1=black):")
    print(f"  {'index':>6}  {'bits':>{h+2}}  real interval")
    n = 1 << h
    for i, c in enumerate(configs):
        bits     = ''.join(map(str, c))
        lo, hi   = i / n, (i + 1) / n
        print(f"  a_{i+1:<4}  {bits:>{h}}    [{lo:.4f}, {hi:.4f})")


# ── Analysis & plotting ────────────────────────────────────────────────────────

def analyse_and_plot(strat_a, strat_b, h, resolution=512):
    """
    Run the full analysis for the given strategy pair and display results.

    Left panel  — finite h-strategy joint outcome matrix (exact, brute-force).
    Right panel — recursive ∞-strategy checkerboard (display: 512×512 grid;
                  probability: exact rational, brute-force on non-mono pairs).
    """
    # ── Finite matrix ──────────────────────────────────────────────────────────
    matrix     = outcome_matrix(strat_a, strat_b, h)
    wins_fin   = int(matrix.sum())
    total_fin  = matrix.size               # 4^h

    print(f"\n  Finite h={h} strategy:")
    print(f"    Winning configs : {wins_fin} / {total_fin}")
    print(f"    Win probability : {Fraction(wins_fin, total_fin)} "
          f"≈ {wins_fin / total_fin:.6f}")

    # ── Recursive ∞-strategy ──────────────────────────────────────────────────
    prob_inf = recursive_winning_probability(strat_a, strat_b, h)

    print(f"\n  Recursive ∞-strategy (block size {h}):")
    print(f"    Win probability : {prob_inf} ≈ {float(prob_inf):.6f}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    density = recursive_density(strat_a, strat_b, h, resolution)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # Left: finite matrix
    axes[0].imshow(matrix, cmap="binary", origin="lower",
                   vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel("A's stack  →", fontsize=8)
    axes[0].set_ylabel("B's stack  →", fontsize=8)
    axes[0].set_title(
        f"Finite {h}-strategy\n"
        f"winning prob. = {wins_fin}/{total_fin}",
        fontsize=9,
    )

    # Right: recursive ∞-strategy
    axes[1].imshow(density, cmap="binary", origin="lower",
                   vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel("A's stack  →", fontsize=8)
    axes[1].set_ylabel("B's stack  →", fontsize=8)
    axes[1].set_title(
        f"Recursive ∞-strategy (order {h})\n"
        f"winning prob. = {prob_inf}",
        fontsize=9,
    )

    fig.suptitle(
        f"Levine hat problem  —  user strategy  (h={h}, n=2)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("user_strategy_output.pdf", dpi=200, bbox_inches="tight")
    print("\n  Plot saved to user_strategy_output.pdf")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Levine hat problem — user strategy analyser")
    print("=" * 60)

    h = _prompt_h()
    _print_config_table(h)

    print()
    strat_a = _prompt_strategy("A", h)
    strat_b = _prompt_strategy("B", h)

    print("\n  Running analysis...")
    analyse_and_plot(strat_a, strat_b, h)


if __name__ == "__main__":
    main()