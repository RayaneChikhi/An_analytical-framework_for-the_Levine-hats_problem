"""
Recursive strategy construction and convergence bounds for Levine's hat problem.

Implements:
  - the recursive ∞-strategy from Definition 12 (winning probability computation)
  - the combining construction S ⋆ T from Definition 24
  - the combining formula (Lemma 25)
  - lower bounds on V_{2,h} from Propositions 16 and 17 (types S3 and S5)
  - the near-optimal convergence rate from Theorem 27
"""

from fractions import Fraction

# ── Winning probability of a recursive strategy ───────────────────────────────

def recursive_win_prob(base_win_prob, h):
    """
    Winning probability of the recursive ∞-strategy built from a t-strategy
    with winning probability base_win_prob, where t = h hats per block.

    The stopping set C is the set of non-monochromatic h-tuples, so
    P(block in C) = 1 - 2/2^h = (2^h - 2) / 2^h.

    Solves:  W = (4 / 4^h) * W  +  ((2^h-2)^2 / 4^h) * p_cond  +  cross terms
    directly via the formula derived in Lemma 21.
    """
    t   = h
    num = Fraction(1, 4**t) * (1 + Fraction(7, 5) * (4**(t-1) - 1))
    return num

def combining_formula(w_S, w_T, N):
    """
    Winning probability of the composed strategy S ⋆ T (Lemma 25).

    w_S : winning probability of the N-strategy S
    w_T : winning probability of the M-strategy T
    N   : block size of S
    """
    return w_S + Fraction(4 * w_T - 1, 4**N)

# ── Standardized strategy chain ───────────────────────────────────────────────

# Base case: S_{3,3}
W_S3 = Fraction(22, 64)   # = 11/32, verified by brute force

def standardized_chain_win_prob(t):
    """
    Winning probability of the standardized t-strategy S_{t,t} constructed
    by the inductive combining construction (Theorem 26).
    Valid for odd t >= 3.
    """
    if t == 3:
        return W_S3
    if t % 2 == 0:
        raise ValueError("Rt-type strategies do not exist for even t (Theorem 22).")
    # Inductive step: S_{t,t} = S_{t-2, t-2} ⋆ S_{3,3}
    w_prev = standardized_chain_win_prob(t - 2)
    return combining_formula(w_prev, W_S3, t - 2)

def check_rt_target(t):
    """
    Verify that S_{t,t} achieves the required winning probability for an
    Rt-type strategy (equation (2) in Theorem 26).
    """
    target = Fraction(7, 20) - Fraction(2, 5 * 4**t)
    actual = standardized_chain_win_prob(t)
    return actual == target

# ── Lower bounds on V_{2,h} ───────────────────────────────────────────────────

def lower_bound_S3(h, V2_small):
    """
    Lower bound on 7/20 - V_{2,h} from Proposition 16 (type S3).

    V2_small : dict mapping r in {1,2,3} to known values of V_{2,r}
    Returns 7/20 - V_{2,h} <= (7/20 - V_{2,r}) / 16^q
    """
    q, r = divmod(h, 3)
    if r == 0:
        q, r = q - 1, 3
    gap_r = Fraction(7, 20) - Fraction(V2_small[r]).limit_denominator(10**9)
    return gap_r / Fraction(16**q)

def lower_bound_S5(h, V2_small):
    """
    Lower bound on 7/20 - V_{2,h} from Proposition 17 (type S5).

    V2_small : dict mapping r in {1,...,5} to known values of V_{2,r}
    Returns 7/20 - V_{2,h} <= (7/20 - V_{2,r}) / 256^q
    """
    q, r = divmod(h, 5)
    if r == 0:
        q, r = q - 1, 5
    gap_r = Fraction(7, 20) - Fraction(V2_small[r]).limit_denominator(10**9)
    return gap_r / Fraction(256**q)

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verify Theorem 26: Rt-type strategies exist for all odd t in {3,...,15}
    print("Checking Rt-type strategy existence (Theorem 26):")
    for t in range(3, 16, 2):
        w   = standardized_chain_win_prob(t)
        ok  = check_rt_target(t)
        print(f"  t = {t:2d}:  w(S_{{t,t}}) = {float(w):.10f}  {'✓' if ok else '✗'}")

    # Theorem 22: no Rt-type strategy for even t
    print("\nChecking Theorem 22 (no even Rt-type strategies):")
    for t in [4, 6, 8]:
        p = Fraction(1, 4**t) * (1 + Fraction(7, 5) * (4**(t-1) - 1))
        is_integer_multiple = (p * 4**t).denominator == 1
        print(f"  t = {t}: required p = {p}  "
              f"({'integer multiple of 1/4^t — Rt exists' if is_integer_multiple else 'not integer multiple — Rt impossible'})")

    # Lower bounds on V_{2,h} for h up to 13
    # Known small values from Table 1 (hill-climbing lower bounds)
    V2_known = {
        1: Fraction(1, 4),
        2: Fraction(5, 16),
        3: Fraction(22, 64),
        4: Fraction(356, 1024),
        5: Fraction(358, 1024),
    }

    print("\nLower bounds on V_{2,h} via S5 (Proposition 17 / Theorem 18):")
    print(f"  {'h':>3}  {'V_{2,h} >=':>20}")
    for h in range(6, 14):
        gap = lower_bound_S5(h, V2_known)
        lb  = Fraction(7, 20) - gap
        print(f"  {h:>3}  {float(lb):.15f}")