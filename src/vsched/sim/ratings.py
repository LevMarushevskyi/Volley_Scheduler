"""Rating gap -> win probability, and single-match simulation."""
from __future__ import annotations

import random

ELO_SCALE = 400.0


def p_set(ra: float, rb: float, scale: float = ELO_SCALE) -> float:
    """P(A wins a set) = 1 / (1 + 10^(-(rA - rB)/scale))."""
    return 1.0 / (1.0 + 10.0 ** (-(ra - rb) / scale))


def play_match(ra: float, rb: float, rng: random.Random, match_format: str = "best_of_3",
               scale: float = ELO_SCALE) -> tuple[int, int]:
    """Return (sets won by A, sets won by B).

    best_of_3: play sets until one side has 2.
    two_sets:  play exactly 2 sets (1-1 splits are possible).
    """
    p = p_set(ra, rb, scale)
    a = b = 0
    if match_format == "two_sets":
        for _ in range(2):
            if rng.random() < p:
                a += 1
            else:
                b += 1
        return a, b
    while a < 2 and b < 2:
        if rng.random() < p:
            a += 1
        else:
            b += 1
    return a, b


def p_match_best_of_3(ra: float, rb: float, scale: float = ELO_SCALE) -> float:
    p = p_set(ra, rb, scale)
    return p * p + 2 * p * p * (1 - p)
