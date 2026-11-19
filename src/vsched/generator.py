"""Seeded random instances in three tiers (tiny / medium / large)."""
from __future__ import annotations

import math

import numpy as np

from .model import Instance, Team, minutes_to_hhmm, hhmm_to_minutes
from .roundrobin import snake_pools

TIERS = {
    "tiny":   {"pools": (1, 2),  "sizes": (3, 4),    "court_frac": (1.0, 1.0)},
    "medium": {"pools": (3, 6),  "sizes": (3, 4, 5), "court_frac": (0.67, 1.0)},
    "large":  {"pools": (8, 12), "sizes": (3, 4),    "court_frac": (0.5, 1.0)},
}


def make_instance(n_teams: int, pool_size: int, courts: int, *, seed: int, instance_id: str | None = None,
                  start: str = "08:00", end: str | None = None, match_minutes: int = 60,
                  slack_slots: int = 2, coach_frac: float = 0.15, rating_sd: float = 150.0) -> Instance:
    """Build one instance. If `end` is None, the day is just long enough for the
    busiest court under the default court rule, plus `slack_slots`."""
    rng = np.random.default_rng(seed)
    ratings = np.sort(rng.normal(1500, rating_sd, n_teams))[::-1]  # seed 1 = highest rating
    teams = [Team(id=f"T{i + 1:02d}", seed=i + 1, rating=round(float(ratings[i]), 1)) for i in range(n_teams)]
    pools = snake_pools(teams, pool_size)
    names = list(pools)
    n_pairs = int(round(coach_frac * n_teams / 2))
    if len(names) >= 2 and n_pairs == 0 and coach_frac > 0:
        n_pairs = 1
    coach_of: dict[str, str] = {}
    pool_of = {t.id: p for p, mem in pools.items() for t in mem}
    available = [t.id for t in teams]
    for k in range(n_pairs if len(names) >= 2 else 0):
        rng.shuffle(available)
        a = available[0]
        partner = next((b for b in available[1:] if pool_of[b] != pool_of[a]), None)
        if partner is None:
            break
        coach_of[a] = coach_of[partner] = f"K{k + 1}"
        available = [t for t in available if t not in (a, partner)]
    teams = [Team(t.id, t.seed, t.rating, coach_of.get(t.id)) for t in teams]
    if end is None:
        per_court = [0] * courts
        for i, mem in enumerate(pools.values()):
            per_court[i % courts] += len(mem) * (len(mem) - 1) // 2
        slots = max(per_court) + slack_slots
        end = minutes_to_hhmm(hhmm_to_minutes(start) + slots * match_minutes)
    return Instance(instance_id=instance_id or f"gen-{seed}", teams=tuple(teams), pool_size=pool_size,
                    courts=courts, start=start, end=end, match_minutes=match_minutes, seed=seed)


def generate(tier: str, seed: int) -> Instance:
    cfg = TIERS[tier]
    rng = np.random.default_rng(seed)
    n_pools = int(rng.integers(cfg["pools"][0], cfg["pools"][1] + 1))
    k = int(rng.choice(cfg["sizes"]))
    if tier == "tiny" and n_pools == 2:
        k = 3
    lo, hi = cfg["court_frac"]
    courts = max(1, min(n_pools, int(math.ceil(n_pools * rng.uniform(lo, hi)))))
    while courts < n_pools and math.ceil(n_pools / courts) * k * (k - 1) // 2 > 12:
        courts += 1
    slack = int(rng.integers(1, 4))
    return make_instance(n_pools * k, k, courts, seed=seed, instance_id=f"{tier}-{seed:03d}", slack_slots=slack)


def generate_many(tier: str, n: int, base_seed: int = 0) -> list[Instance]:
    return [generate(tier, base_seed + i) for i in range(n)]
