"""One simulated tournament: pool play, then single-elimination brackets.

Rules (documented because they affect fairness):
  * Pools: snake seeding by true rating (seed 1 = best), or random pools.
  * Pool matches: best of 3 sets, or exactly 2 sets ('two_sets').
  * Pool standings: match points (win = 1; a 1-1 split in two_sets = 0.5
    each), then set ratio (sets won / sets played), then head-to-head (only
    when exactly two teams are tied), then a coin flip.
  * Brackets are cut by pool finishing PLACE, as VolleyLens does: e.g.
    Gold = 1st and 2nd of every pool, Silver = the rest.
  * Bracket seeding: by place (pool winners first), then by pool record
    (match points, set ratio), then coin flip. Standard bracket order
    (1 v N, ...), byes to the top seeds when the size is not a power of 2.
    Bracket matches are always best of 3.
  * Final placement: brackets in listed order; inside a bracket by the round
    a team went out (champion 1st, finalist 2nd, semifinal losers tied 3rd,
    ...); teams in no bracket come last, by pool place.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .ratings import play_match


@dataclass
class Format:
    n_teams: int
    pool_size: int
    brackets: list[tuple[str, list[int]]] = field(default_factory=lambda: [("gold", [1, 2])])
    match_format: str = "best_of_3"         # or "two_sets"
    seeding: str = "snake"                  # or "random"

    @property
    def label(self) -> str:
        cut = "+".join(f"{name}:{''.join(map(str, places))}" for name, places in self.brackets)
        return f"pools of {self.pool_size} | {cut} | {self.seeding} | {self.match_format}"


def make_pools(n_teams: int, pool_size: int, seeding: str, rng: random.Random) -> list[list[int]]:
    """Teams are indices 0..n-1 in true-rating order (0 = best)."""
    p = math.ceil(n_teams / pool_size)
    order = list(range(n_teams))
    if seeding == "random":
        rng.shuffle(order)
    pools: list[list[int]] = [[] for _ in range(p)]
    for i, t in enumerate(order):
        row, col = divmod(i, p)
        pools[col if row % 2 == 0 else p - 1 - col].append(t)
    return pools


def bracket_order(size: int) -> list[int]:
    """Seed numbers (1-based) in bracket line order, e.g. 8 -> 1,8,4,5,2,7,3,6."""
    order = [1]
    while len(order) < size:
        n = len(order) * 2
        order = [x for s in order for x in (s, n + 1 - s)]
    return order


def play_bracket(entrants: list[int], ratings, rng: random.Random, played: list[int]) -> list[int]:
    """Single elimination. entrants are in seed order. Returns the round each
    entrant went out in (0 = champion, 1 = lost final, 2 = lost semi, ...)."""
    n = len(entrants)
    if n == 0:
        return []
    if n == 1:
        return [0]
    size = 1 << (n - 1).bit_length()
    rounds = int(math.log2(size))
    line = [entrants[s - 1] if s <= n else None for s in bracket_order(size)]
    out = {}
    r = rounds
    while len(line) > 1:
        nxt = []
        for i in range(0, len(line), 2):
            a, b = line[i], line[i + 1]
            if a is None or b is None:
                nxt.append(a if b is None else b)
                continue
            sa, sb = play_match(ratings[a], ratings[b], rng, "best_of_3")
            played[a] += 1
            played[b] += 1
            win, lose = (a, b) if sa > sb else (b, a)
            out[lose] = r
            nxt.append(win)
        line = nxt
        r -= 1
    out[line[0]] = 0
    return [out[t] for t in entrants]


def run_once(fmt: Format, ratings, rng: random.Random) -> dict:
    """ratings indexed by team in true order (0 = best). Returns champion of the
    first bracket, a placement value per team (lower = better), matches played."""
    n = fmt.n_teams
    pools = make_pools(n, fmt.pool_size, fmt.seeding, rng)
    played = [0] * n
    pts = [0.0] * n
    sw = [0] * n
    sp = [0] * n
    h2h = {}
    for pool in pools:
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                a, b = pool[i], pool[j]
                sa, sb = play_match(ratings[a], ratings[b], rng, fmt.match_format)
                played[a] += 1
                played[b] += 1
                sw[a] += sa
                sw[b] += sb
                sp[a] += sa + sb
                sp[b] += sa + sb
                if sa > sb:
                    pts[a] += 1
                    h2h[(a, b)] = a
                elif sb > sa:
                    pts[b] += 1
                    h2h[(a, b)] = b
                else:
                    pts[a] += 0.5
                    pts[b] += 0.5
    coin = [rng.random() for _ in range(n)]
    ratio = [sw[t] / sp[t] if sp[t] else 0.0 for t in range(n)]
    place = [0] * n
    for pool in pools:
        ranked = sorted(pool, key=lambda t: (-pts[t], -ratio[t], coin[t]))
        # head-to-head for exactly-two-way ties on points and set ratio
        i = 0
        while i < len(ranked) - 1:
            a, b = ranked[i], ranked[i + 1]
            tied = (pts[a], ratio[a]) == (pts[b], ratio[b])
            three = i + 2 < len(ranked) and (pts[ranked[i + 2]], ratio[ranked[i + 2]]) == (pts[a], ratio[a])
            prev = i > 0 and (pts[ranked[i - 1]], ratio[ranked[i - 1]]) == (pts[a], ratio[a])
            if tied and not three and not prev:
                w = h2h.get((min(a, b), max(a, b)))
                if w == b:
                    ranked[i], ranked[i + 1] = b, a
                i += 2
            else:
                i += 1
        for k, t in enumerate(ranked):
            place[t] = k + 1

    placement = [0.0] * n
    in_bracket = [False] * n
    base = 0
    champion = None
    for bi, (_name, places) in enumerate(fmt.brackets):
        ents = [t for t in range(n) if place[t] in places]
        ents.sort(key=lambda t: (place[t], -pts[t], -ratio[t], coin[t]))
        outs = play_bracket(ents, ratings, rng, played)
        if bi == 0 and ents:
            champion = ents[outs.index(0)]
        for t, r in zip(ents, outs):
            placement[t] = base + (0 if r == 0 else 2 ** (r - 1))  # 0, 1, 2, 4, 8 ...
            in_bracket[t] = True
        base += len(ents) + 1
    for t in range(n):
        if not in_bracket[t]:
            placement[t] = base + place[t]
    return {"champion": champion, "placement": placement, "played": played}
