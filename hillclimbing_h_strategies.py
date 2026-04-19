"""
Hill-climbing search for efficient h-strategies in Levine's hat problem (n=2 players).

Each strategy is a map {0,1}^h -> {1,...,h}, encoded as a list of length 2^h.
Hat configurations are indexed lexicographically via their binary representation.
The winning probability is computed exactly over all 4^h configuration pairs.
"""

from random import randint
import numpy as np
import random

# ── Parameters ────────────────────────────────────────────────────────────────

h = 5
n_configs = 2 ** h           # number of hat configurations per player
n_pairs   = 4 ** h           # total configuration pairs
mutable   = list(range(1, n_configs - 1))  # indices open to perturbation

step                = 0
perturbation_radius = 1

# ── Hat configurations ─────────────────────────────────────────────────────────

def generate_configs(h):
    return [tuple((i >> j) & 1 for j in range(h)) for i in range(1 << h)]

configs    = generate_configs(h)
config_idx = {c: i for i, c in enumerate(configs)}

# ── Core game logic ────────────────────────────────────────────────────────────

def play(strat_a, strat_b, stack_a, stack_b):
    guess_a = strat_a[config_idx[stack_b]]
    guess_b = strat_b[config_idx[stack_a]]
    return min(stack_a[guess_a - 1], stack_b[guess_b - 1])

def winning_probability(strat_a, strat_b):
    total = sum(
        play(strat_a, strat_b, a, b)
        for a in configs for b in configs
    )
    return total / n_pairs

# ── Neighbour generation ───────────────────────────────────────────────────────

def get_neighbour(strat):
    idxs = np.random.choice(mutable, perturbation_radius, replace=False)
    perturbation = {}
    previous     = {}
    for idx in idxs:
        previous[idx]     = strat[idx]
        perturbation[idx] = randint(1, h)
    return perturbation, previous

# ── Hill-climbing ──────────────────────────────────────────────────────────────

def climb(strat_a, strat_b, value):
    global step
    best = 0
    while True:
        if step % 2 == 0:
            max_value = value
            max_strat = strat_a
            perturb, prev = get_neighbour(strat_a)
            for idx in perturb:
                strat_a[idx] = perturb[idx]
            t = winning_probability(strat_a, strat_b)
            if t >= max_value:
                max_value = t
                max_strat = strat_a
            else:
                for idx in prev:
                    strat_a[idx] = prev[idx]
            value   = max_value
            strat_a = max_strat
            if value > best:
                best = value
                print(max_value, max_strat, strat_b)
        else:
            max_value = value
            max_strat = strat_b
            perturb, prev = get_neighbour(strat_b)
            for idx in perturb:
                strat_b[idx] = perturb[idx]
            t = winning_probability(strat_a, strat_b)
            if t >= max_value:
                max_value = t
                max_strat = strat_b
            else:
                for idx in prev:
                    strat_b[idx] = prev[idx]
            value   = max_value
            strat_b = max_strat
            if value > best:
                best = value
                print(max_value, strat_a, max_strat)
        step += 1


def hillclimb():
    strat_a = [1] * n_configs
    strat_b = [1] * n_configs
    v = winning_probability(strat_a, strat_b)
    print("Starting at", v, "with strategy", strat_a, strat_b)
    climb(strat_a, strat_b, v)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hillclimb()