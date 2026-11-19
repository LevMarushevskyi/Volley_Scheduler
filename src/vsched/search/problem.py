"""Compiled, index-based view of an instance shared by search and annealing."""
from __future__ import annotations

from dataclasses import dataclass

from ..model import Assignment, Instance, Schedule
from ..roundrobin import default_pool_courts, instance_matches, instance_pools


@dataclass
class Problem:
    inst: Instance
    team_ids: list[str]
    pools: list[str]
    pool_members: list[list[int]]       # pool idx -> team idxs
    team_pool: list[int]                # team idx -> pool idx
    matches: list[tuple[int, int, int]]  # (t1, t2, pool idx)
    match_ids: list[str]
    pool_court: list[int]               # pool idx -> 0-based court
    court_matches: list[list[int]]
    team_matches: list[list[int]]
    coach_mates: list[list[int]]
    n_slots: int
    n_courts: int

    @property
    def n_teams(self) -> int:
        return len(self.team_ids)

    def to_schedule(self, cells: dict[int, tuple[int, int, int]], method: str) -> Schedule:
        """cells: match idx -> (court0, slot, ref idx)."""
        out = []
        for mi, (c, s, r) in sorted(cells.items(), key=lambda kv: (kv[1][1], kv[1][0])):
            t1, t2, p = self.matches[mi]
            out.append(Assignment(match_id=self.match_ids[mi], pool=self.pools[p],
                                  team1=self.team_ids[t1], team2=self.team_ids[t2],
                                  court=c + 1, slot=s, ref=self.team_ids[r]))
        return Schedule(instance_id=self.inst.instance_id, method=method, assignments=out)

    def from_schedule(self, sched: Schedule) -> dict[int, tuple[int, int, int]]:
        mi = {m: i for i, m in enumerate(self.match_ids)}
        ti = {t: i for i, t in enumerate(self.team_ids)}
        return {mi[a.match_id]: (a.court - 1, a.slot, ti[a.ref]) for a in sched.assignments}


def compile_problem(inst: Instance, pool_courts: dict[str, int] | None = None) -> Problem:
    pools_d = instance_pools(inst)
    pools = list(pools_d)
    team_ids = [t.id for p in pools for t in pools_d[p]]
    ti = {t: i for i, t in enumerate(team_ids)}
    pool_members = [[ti[t.id] for t in pools_d[p]] for p in pools]
    team_pool = [0] * len(team_ids)
    for pi, mem in enumerate(pool_members):
        for t in mem:
            team_pool[t] = pi
    pidx = {p: i for i, p in enumerate(pools)}
    ms = instance_matches(inst)
    matches = [(ti[m.team1], ti[m.team2], pidx[m.pool]) for m in ms]
    pc = pool_courts or default_pool_courts(inst)
    pool_court = [pc[p] - 1 for p in pools]
    court_matches = [[] for _ in range(inst.courts)]
    team_matches = [[] for _ in team_ids]
    for i, (a, b, p) in enumerate(matches):
        court_matches[pool_court[p]].append(i)
        team_matches[a].append(i)
        team_matches[b].append(i)
    coach = {t.id: t.coach for t in inst.teams}
    coach_mates = [[ti[u] for u in team_ids if u != t and coach[t] and coach[u] == coach[t]] for t in team_ids]
    return Problem(inst=inst, team_ids=team_ids, pools=pools, pool_members=pool_members,
                   team_pool=team_pool, matches=matches, match_ids=[m.match_id for m in ms],
                   pool_court=pool_court, court_matches=court_matches, team_matches=team_matches,
                   coach_mates=coach_mates, n_slots=inst.n_slots, n_courts=inst.courts)
