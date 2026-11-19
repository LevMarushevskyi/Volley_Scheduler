"""Match-overrun replay (Phase 8, stretch).

Match length ~ lognormal. A match starts at its planned time, or later if its
court, either team or its ref team is still busy. Replays report when the day
ends and the longest delay any match suffers.
"""
from __future__ import annotations

import math

import numpy as np

from ..model import Instance, Schedule


def lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """(mu, sigma) of a lognormal with the given mean and coefficient of variation."""
    sigma2 = math.log(1 + cv * cv)
    return math.log(mean) - sigma2 / 2, math.sqrt(sigma2)


def fit_lognormal(durations_min: list[float]) -> tuple[float, float]:
    x = np.log(np.asarray(durations_min, dtype=float))
    return float(x.mean()), float(x.std(ddof=1))


def replay(inst: Instance, sched: Schedule, durations: np.ndarray) -> dict:
    """durations: minutes for each assignment in planned (slot, court) order."""
    L = inst.match_minutes
    order = sorted(sched.assignments, key=lambda a: (a.slot, a.court))
    court_free: dict[int, float] = {}
    team_free: dict[str, float] = {}
    max_delay = 0.0
    end = 0.0
    for a, d in zip(order, durations):
        planned = a.slot * L
        start = max(planned, court_free.get(a.court, 0.0),
                    team_free.get(a.team1, 0.0), team_free.get(a.team2, 0.0), team_free.get(a.ref, 0.0))
        fin = start + float(d)
        court_free[a.court] = fin
        for t in (a.team1, a.team2, a.ref):
            team_free[t] = fin
        max_delay = max(max_delay, start - planned)
        end = max(end, fin)
    planned_end = (max((a.slot for a in sched.assignments), default=-1) + 1) * L
    return {"day_end_min": end, "planned_end_min": planned_end, "overrun_min": end - planned_end,
            "max_delay_min": max_delay}


def simulate_delays(inst: Instance, sched: Schedule, *, n_reps: int = 1000, seed: int = 0,
                    mean: float | None = None, cv: float = 0.15,
                    mu_sigma: tuple[float, float] | None = None) -> dict[str, np.ndarray]:
    """1,000 replays by default. Without fitted (mu, sigma) the mean match length is
    assumed to equal the planned slot length with a 15% coefficient of variation."""
    rng = np.random.default_rng(seed)
    mu, sigma = mu_sigma or lognormal_params(mean or inst.match_minutes, cv)
    n = len(sched.assignments)
    out = {"day_end_min": np.empty(n_reps), "overrun_min": np.empty(n_reps), "max_delay_min": np.empty(n_reps)}
    for i in range(n_reps):
        r = replay(inst, sched, rng.lognormal(mu, sigma, n))
        for k in out:
            out[k][i] = r[k]
    return out
