"""Monte Carlo fairness metrics (W5 probability, W7 sampling).

simulate()         fixed ratings (the locked teammate API).
simulate_spread()  redraws ratings from Normal(1500, sd) every run, so the
                   result describes a kind of field rather than one draw.
Rates come with a 95% normal interval p +- 1.96 sqrt(p(1-p)/N).
"""
from __future__ import annotations

import math
import random

import numpy as np

from .tournament import Format, run_once


def kendall_tau_b(true_rank: np.ndarray, placement: np.ndarray) -> float:
    """tau-b between true rank (no ties, lower = better) and placement (ties allowed)."""
    x = np.sign(true_rank[:, None] - true_rank[None, :])
    y = np.sign(placement[:, None] - placement[None, :])
    iu = np.triu_indices(len(true_rank), 1)
    xs, ys = x[iu], y[iu]
    s = float(np.sum(xs * ys))
    n0 = len(xs)
    n1 = float(np.sum(xs == 0))
    n2 = float(np.sum(ys == 0))
    den = math.sqrt((n0 - n1) * (n0 - n2))
    return s / den if den else 0.0


def ci95(p: float, n: int) -> tuple[float, float]:
    half = 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n) if n else 0.0
    return max(0.0, p - half), min(1.0, p + half)


def _run(fmt: Format, n_runs: int, seed: int, draw) -> dict:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    n = fmt.n_teams
    wins = 0
    champ_counts = [0] * n
    taus = np.empty(n_runs)
    played_sum = 0
    for i in range(n_runs):
        ratings = draw(nrng)
        order = np.argsort(-np.asarray(ratings), kind="stable")  # input indices, strongest first
        # team id = true rank (0 = best); ratings tied with the best do not count as best
        teams = [{"id": k, "rating": float(ratings[j])} for k, j in enumerate(order)]
        res = run_once(fmt, teams, rng)
        champ = res["champion"]
        if champ is not None:
            champ_counts[int(order[champ["id"]])] += 1
            if champ["id"] == 0:
                wins += 1
        place = np.array([res["placement"][k] for k in range(n)], dtype=float)
        taus[i] = kendall_tau_b(np.arange(n, dtype=float), place)
        played_sum += sum(res["played"].values())
    p = wins / n_runs
    lo, hi = ci95(p, n_runs)
    return {"p_best_wins_gold": p, "ci_low": lo, "ci_high": hi,
            "kendall_tau_mean": float(taus.mean()), "kendall_tau_sd": float(taus.std()),
            "matches_per_team_mean": played_sum / (n_runs * n), "n_runs": n_runs,
            "champion_counts": champ_counts}


def simulate(fmt: Format, ratings: list[float], n_runs: int, seed: int) -> dict:
    """Fixed ratings; every set result is a random draw.
    Returns p_best_wins_gold, ci_low, ci_high, kendall_tau_mean, matches_per_team_mean
    (plus kendall_tau_sd, n_runs and champion_counts indexed like `ratings`)."""
    if len(ratings) != fmt.n_teams:
        raise ValueError("len(ratings) must equal fmt.n_teams")
    arr = np.asarray(ratings, dtype=float)
    return _run(fmt, n_runs, seed, lambda _rng: arr)


def simulate_spread(fmt: Format, sd: float, n_runs: int, seed: int, mean: float = 1500.0) -> dict:
    """Ratings redrawn from Normal(mean, sd) every run."""
    return _run(fmt, n_runs, seed, lambda rng: rng.normal(mean, sd, fmt.n_teams))
