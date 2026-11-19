"""Hard-constraint checker and soft cost function.

Every method, baseline and experiment is judged by this file.

Component definitions (all counted per team unless noted):
  idle            empty slots between a team's first and last activity (play or ref)
  three_straight  plays in a 3rd (or later) consecutive slot: a run of L plays counts L-2
  double_work     refs in a 2nd (or later) consecutive slot: a run of L refs counts L-1
  work_imbalance  per pool, max refs worked - min refs worked (summed over pools)
  coach_conflict  per slot, pairs of same-coach teams that are both busy
  makespan        slots used through the last match = last match slot + 1
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import yaml

from .model import Instance, Schedule
from .roundrobin import instance_matches, instance_pools

COMPONENTS = ("idle", "three_straight", "double_work", "work_imbalance", "coach_conflict", "makespan")
DEFAULT_WEIGHTS = {"idle": 1, "three_straight": 5, "double_work": 2,
                   "work_imbalance": 3, "coach_conflict": 10, "makespan": 2}

IDLE, PLAY, REF = 0, 1, 2

_REPO = Path(__file__).resolve().parents[2]


def load_weights(path: str | Path | None = None) -> dict[str, float]:
    p = Path(path) if path else _REPO / "configs" / "weights.yaml"
    w = dict(DEFAULT_WEIGHTS)
    if p.exists():
        with open(p) as f:
            w.update(yaml.safe_load(f) or {})
    return {k: float(w[k]) for k in COMPONENTS}


def check_feasible(inst: Instance, sched: Schedule) -> list[str]:
    """Return a list of hard-constraint violations; empty means legal."""
    v: list[str] = []
    matches = {m.match_id: m for m in instance_matches(inst)}
    pool_of = {t.id: p for p, members in instance_pools(inst).items() for t in members}

    counts = Counter(a.match_id for a in sched.assignments)
    for mid in matches:
        if counts[mid] == 0:
            v.append(f"missing match {mid}")
        elif counts[mid] > 1:
            v.append(f"match {mid} scheduled {counts[mid]} times")
    for a in sched.assignments:
        m = matches.get(a.match_id)
        if m is None:
            v.append(f"unknown match {a.match_id}")
        elif {a.team1, a.team2} != {m.team1, m.team2} or a.pool != m.pool:
            v.append(f"match {a.match_id} has wrong teams or pool")

    for a in sched.assignments:
        if not (0 <= a.slot < inst.n_slots):
            v.append(f"match {a.match_id} slot {a.slot} outside day (0..{inst.n_slots - 1})")
        if not (1 <= a.court <= inst.courts):
            v.append(f"match {a.match_id} court {a.court} does not exist")

    cells = Counter((a.court, a.slot) for a in sched.assignments)
    for (c, s), n in sorted(cells.items()):
        if n > 1:
            v.append(f"court {c} slot {s} has {n} matches")

    playing = defaultdict(set)
    for a in sched.assignments:
        playing[a.slot].update((a.team1, a.team2))
    for a in sched.assignments:
        if pool_of.get(a.ref) != a.pool:
            v.append(f"match {a.match_id} ref {a.ref} is not in pool {a.pool}")
        if a.ref in playing[a.slot]:
            v.append(f"match {a.match_id} ref {a.ref} is playing in slot {a.slot}")

    busy = Counter()
    for a in sched.assignments:
        for t in (a.team1, a.team2, a.ref):
            busy[(t, a.slot)] += 1
    for (t, s), n in sorted(busy.items()):
        if n > 1:
            v.append(f"team {t} has {n} duties in slot {s}")
    return v


def activity_grid(inst: Instance, sched: Schedule) -> dict[str, list[int]]:
    """team id -> per-slot activity (IDLE/PLAY/REF). Length covers the day and any overflow."""
    T = max([inst.n_slots] + [a.slot + 1 for a in sched.assignments])
    grid = {t.id: [IDLE] * T for t in inst.teams}
    for a in sched.assignments:
        grid[a.team1][a.slot] = PLAY
        grid[a.team2][a.slot] = PLAY
        grid[a.ref][a.slot] = REF
    return grid


def team_components(row: list[int]) -> tuple[int, int, int, int]:
    """(idle, three_straight, double_work, refs) for one team's activity row."""
    busy = [i for i, x in enumerate(row) if x != IDLE]
    if not busy:
        return 0, 0, 0, 0
    idle = (busy[-1] - busy[0] + 1) - len(busy)
    three = double = refs = 0
    pr = rr = 0
    for x in row:
        if x == PLAY:
            pr += 1
            rr = 0
            if pr >= 3:
                three += 1
        elif x == REF:
            rr += 1
            pr = 0
            refs += 1
            if rr >= 2:
                double += 1
        else:
            pr = rr = 0
    return idle, three, double, refs


def cost(inst: Instance, sched: Schedule, weights: dict[str, float] | None = None) -> dict[str, float]:
    """Each component plus the weighted 'total'."""
    w = weights or load_weights()
    grid = activity_grid(inst, sched)
    comp = dict.fromkeys(COMPONENTS, 0.0)
    refs = {}
    for tid, row in grid.items():
        i, th, d, r = team_components(row)
        comp["idle"] += i
        comp["three_straight"] += th
        comp["double_work"] += d
        refs[tid] = r
    for members in instance_pools(inst).values():
        rs = [refs[t.id] for t in members]
        comp["work_imbalance"] += max(rs) - min(rs)
    by_coach = defaultdict(list)
    for t in inst.teams:
        if t.coach:
            by_coach[t.coach].append(t.id)
    for ids in by_coach.values():
        for a, b in combinations(ids, 2):
            comp["coach_conflict"] += sum(1 for x, y in zip(grid[a], grid[b]) if x != IDLE and y != IDLE)
    comp["makespan"] = float(max((a.slot for a in sched.assignments), default=-1) + 1)
    comp["total"] = sum(w[k] * comp[k] for k in COMPONENTS)
    return comp
