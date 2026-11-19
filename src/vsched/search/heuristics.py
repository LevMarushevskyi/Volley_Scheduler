"""Admissible, consistent heuristics for the cell-by-cell best-first search.

Every term is a lower bound on cost not yet in g, and every term only grows
along a path by no more than g grows. That makes h consistent, so A* with a
closed set stays optimal (checked by the A* == UCS test).

  makespan  (the Build Guide's starting heuristic)
      w_makespan * slots still to open beyond the current one. The search
      charges makespan slot by slot in g. Under the court rule every pool sits
      on one court, so a court with R remaining matches needs R more cells;
      the current slot's cell counts if it is still open. (Teams' remaining
      matches never give a later bound than their court, because a team's
      matches are a subset of its court's matches.)

  full  = makespan
      + w_idle * idle slots already certain but not yet charged: a team that
        has started and still has a match to play pays, when it next plays,
        for every slot since its last activity, including the current slot if
        its court's cell in this slot is already decided.
      + w_imbalance * a lower bound on each pool's final work imbalance: at
        least 1 when the pool's match count is not divisible by its size, and
        at least (most refs so far) - min over teams of (refs so far + pool
        matches left that the team is not playing in).
"""
from __future__ import annotations

HEURISTICS = ("makespan", "full")


def lower_bound_end(slot: int, court: int, rem_court) -> int:
    """Latest slot index any court must still reach (-1 if nothing remains)."""
    end = -1
    for q, r in enumerate(rem_court):
        if r:
            e = slot + r - (1 if q >= court else 0)
            if e > end:
                end = e
    return end


def pending_idle(slot: int, court: int, teams, mask: int, P, team_court) -> int:
    n = 0
    for t, (last, _p, _r, _refs) in enumerate(teams):
        if last < 0 or last == slot:
            continue
        if all(mask >> mi & 1 for mi in P.team_matches[t]):
            continue
        gap = slot - last - 1
        if team_court[t] < court:
            gap += 1
        n += gap
    return n


def imbalance_lb(mask: int, teams, P, pool_match_list) -> int:
    total = 0
    for p, mem in enumerate(P.pool_members):
        k = len(mem)
        left = [mi for mi in pool_match_list[p] if not mask >> mi & 1]
        parity = 1 if len(pool_match_list[p]) % k else 0
        refs = [teams[t][3] for t in mem]
        if not left:
            total += max(refs) - min(refs)
            continue
        hi = max(refs)
        lo = min(refs[i] + sum(1 for mi in left if t not in P.matches[mi][:2]) for i, t in enumerate(mem))
        total += max(parity, hi - lo)
    return total
