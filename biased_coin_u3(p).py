"""
Biased coin variant of Levine's hat problem.

Each hat is black independently with probability p in (0,1).
This module computes the lower bounds U1, U2 (Buhler et al., Theorem 3) and
U3 (Theorem 29, this paper) for the optimal winning probability V2(p),
and reproduces Figure 7 (dominance regions plot).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Lower bounds ───────────────────────────────────────────────────────────────

def U1(p):
    """Lower bound from S3, valid for p <= 1/2 (Buhler et al.)."""
    num = p * (1 + p + p**2 + 3*p**3 - 3*p**4 + p**5)
    den = (1 + p) * (2 - p) * (1 + p**2)
    return num / den

def U2(p):
    """Lower bound from S3 (symmetric), valid for p >= 1/2 (Buhler et al.)."""
    num = p * (1 + 5*p - 10*p**2 + 10*p**3 - 5*p**4 + p**5)
    den = (2 - 2*p + p**2) * (1 + p) * (2 - p)
    return num / den

def U3(p):
    """
    Lower bound from S5 (Theorem 29), strictly better than max(U1, U2)
    for 0 < p < 0.312.
    """
    num = (5*p - 20*p**2 + 51*p**3 - 82*p**4 + 85*p**5
           - 52*p**6 + 10*p**7 + 10*p**8 - 7*p**9)
    den = (10 - 45*p + 120*p**2 - 210*p**3 + 250*p**4
           - 200*p**5 + 100*p**6 - 25*p**7)
    return num / den

def best_lower_bound(p):
    """Pointwise maximum of all known lower bounds."""
    return max(U1(p), U2(p), U3(p))

# ── Dominance regions ─────────────────────────────────────────────────────────

def dominance_regions(ps):
    """
    For each p in ps, return which bound dominates: 'U1', 'U2', or 'U3'.
    """
    regions = []
    for p in ps:
        u1, u2, u3 = U1(p), U2(p), U3(p)
        best = max(u1, u2, u3)
        if u3 == best:
            regions.append("U3")
        elif u1 == best:
            regions.append("U1")
        else:
            regions.append("U2")
    return regions

# ── Figure 7 ───────────────────────────────────────────────────────────────────

def plot_dominance(save_path=None):
    """Reproduce Figure 7: dominance regions of U1, U2, U3."""
    ps      = np.linspace(0.001, 0.999, 2000)
    regions = dominance_regions(ps)

    colors  = {"U1": "blue", "U2": "green", "U3": "red"}
    labels  = {"U1": r"$U_1$ dominating",
               "U2": r"$U_2$ dominating",
               "U3": r"$U_3$ dominating"}

    fig, ax = plt.subplots(figsize=(8, 5))

    # plot each bound coloured by dominance
    for bound, fn in [("U1", U1), ("U2", U2), ("U3", U3)]:
        vals = [fn(p) for p in ps]
        mask = [r == bound for r in regions]
        ax.plot(
            [p for p, m in zip(ps, mask) if m],
            [v for v, m in zip(vals, mask) if m],
            color=colors[bound], linewidth=1.5,
        )

    # mark the conjectured optimum
    ax.plot(0.5, 0.35, "ko", markersize=5, zorder=5)
    ax.annotate(r"$(0.5,\ 0.35)$", xy=(0.5, 0.35),
                xytext=(0.52, 0.30), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))

    legend_elements = [
        Line2D([0], [0], color=colors[b], lw=1.5, label=labels[b])
        for b in ["U1", "U2", "U3"]
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")
    ax.set_xlabel("Probability $p$", fontsize=10)
    ax.set_ylabel("Lower bound on $V_2(p)$", fontsize=10)
    ax.set_title("Dominance Regions", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verify U3 improves on U1 and U2 for p < 0.312
    threshold = 0.312
    ps_test   = np.linspace(0.001, threshold - 0.001, 500)
    assert all(U3(p) > max(U1(p), U2(p)) for p in ps_test), \
        "U3 does not dominate on the expected interval."
    print(f"U3 strictly dominates max(U1, U2) for all tested p in (0, {threshold}).")
    print(f"U1(0.5) = {U1(0.5):.6f},  U2(0.5) = {U2(0.5):.6f}  (both should equal 7/20 = 0.35)")

    plot_dominance(save_path="figure7.pdf")