"""Snake seeding into pools, round-robin pairings, and the default court rule."""
from __future__ import annotations

import math
import string

from .model import Instance, Match, Team


def pool_name(i: int) -> str:
    letters = string.ascii_uppercase
    return letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]


def n_pools(n_teams: int, pool_size: int) -> int:
    return math.ceil(n_teams / pool_size)


def snake_pools(teams: list[Team] | tuple[Team, ...], pool_size: int) -> dict[str, list[Team]]:
    """Snake seeding. 12 teams, k=4: A=1,6,7,12  B=2,5,8,11  C=3,4,9,10.

    The number of pools is ceil(n / k); when n is not a multiple of k some
    pools get k-1 teams. Every pool must end up with at least 3 teams.
    """
    ordered = sorted(teams, key=lambda t: t.seed)
    p = n_pools(len(ordered), pool_size)
    pools: list[list[Team]] = [[] for _ in range(p)]
    for i, t in enumerate(ordered):
        row, col = divmod(i, p)
        idx = col if row % 2 == 0 else p - 1 - col
        pools[idx].append(t)
    if any(len(pl) < 3 for pl in pools):
        raise ValueError(f"{len(ordered)} teams with pool size {pool_size} leaves a pool smaller than 3")
    return {pool_name(i): pl for i, pl in enumerate(pools)}


def round_robin(pool: str, members: list[Team]) -> list[Match]:
    """All k(k-1)/2 pairings; team1 is the better-seeded team."""
    ms = sorted(members, key=lambda t: t.seed)
    out = []
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = ms[i], ms[j]
            out.append(Match(match_id=f"{pool}-{a.id}-{b.id}", pool=pool, team1=a.id, team2=b.id))
    return out


def instance_pools(inst: Instance) -> dict[str, list[Team]]:
    return snake_pools(inst.teams, inst.pool_size)


def instance_matches(inst: Instance) -> list[Match]:
    out: list[Match] = []
    for p, members in instance_pools(inst).items():
        out.extend(round_robin(p, members))
    return out


def default_pool_courts(inst: Instance) -> dict[str, int]:
    """Default court rule: each pool stays on one court (1-based).

    Pools share a court only when there are more pools than courts; pool i
    then goes on court (i mod C) + 1, i.e. pools are dealt round-robin.
    """
    pools = list(instance_pools(inst))
    return {p: (i % inst.courts) + 1 for i, p in enumerate(pools)}
