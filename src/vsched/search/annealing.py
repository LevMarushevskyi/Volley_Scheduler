"""Simulated annealing over complete schedules.

(Not in the course slides; local search that repairs a full schedule a
little at a time, for instances where A* runs out of time.)

1. Start from the greedy schedule (or the template, or any given schedule).
2. Each step tries one random move: swap two matches' cells on a court, move
   a match to an empty cell on its court, change one match's ref, or swap two
   pools' courts.
3. Moves that break a hard constraint are skipped. Only the teams a move
   touches are re-checked and re-scored.
4. Better schedules are always accepted; worse ones with probability
   exp(-delta / T).
5. T is multiplied by 0.995 every 100 moves.
6. Repeat with several seeds and keep the best.
"""
from __future__ import annotations

import math
import random
import time
from collections import defaultdict

from ..cost import COMPONENTS, IDLE, PLAY, REF, cost as score, check_feasible, team_components
from ..model import Instance, SearchResult, Schedule
from .problem import Problem, compile_problem


class _State:
    """Mutable schedule with incremental cost."""

    def __init__(self, P: Problem, cells: dict[int, tuple[int, int, int]], weights: dict[str, float]):
        self.P, self.w = P, weights
        T = P.n_slots
        self.T = T
        self.cells = dict(cells)  # match -> (court, slot, ref)
        self.occ = [[-1] * T for _ in range(P.n_courts)]
        self.act = [[IDLE] * T for _ in range(P.n_teams)]
        self.slot_count = [0] * T
        for mi, (c, s, r) in self.cells.items():
            self._put(mi, c, s, r)
        self.team_part = [team_components(row) for row in self.act]
        self.coach_pairs = sorted({(min(t, u), max(t, u)) for t in range(P.n_teams) for u in P.coach_mates[t]})
        self.pairs_of = defaultdict(list)
        for pr in self.coach_pairs:
            self.pairs_of[pr[0]].append(pr)
            self.pairs_of[pr[1]].append(pr)
        self.pair_val = {pr: self._pair(pr) for pr in self.coach_pairs}
        self.pool_val = [self._pool(p) for p in range(len(P.pools))]
        self.total = self._total()

    # grid primitives
    def _put(self, mi, c, s, r):
        a, b, _ = self.P.matches[mi]
        self.occ[c][s] = mi
        self.act[a][s] = PLAY
        self.act[b][s] = PLAY
        self.act[r][s] = REF
        self.slot_count[s] += 1

    def _take(self, mi, c, s, r):
        a, b, _ = self.P.matches[mi]
        self.occ[c][s] = -1
        self.act[a][s] = IDLE
        self.act[b][s] = IDLE
        self.act[r][s] = IDLE
        self.slot_count[s] -= 1

    def _ok(self, mi, c, s, r) -> bool:
        a, b, p = self.P.matches[mi]
        return (0 <= s < self.T and self.occ[c][s] == -1 and self.act[a][s] == IDLE
                and self.act[b][s] == IDLE and self.act[r][s] == IDLE and r != a and r != b
                and self.P.team_pool[r] == p)

    # cost pieces
    def _pair(self, pr):
        x, y = self.act[pr[0]], self.act[pr[1]]
        return sum(1 for i in range(self.T) if x[i] != IDLE and y[i] != IDLE)

    def _pool(self, p):
        rs = [self.team_part[t][3] for t in self.P.pool_members[p]]
        return max(rs) - min(rs)

    def _makespan(self):
        for s in range(self.T - 1, -1, -1):
            if self.slot_count[s]:
                return s + 1
        return 0

    def components(self) -> dict[str, float]:
        c = dict.fromkeys(COMPONENTS, 0.0)
        for i, th, d, _ in self.team_part:
            c["idle"] += i
            c["three_straight"] += th
            c["double_work"] += d
        c["work_imbalance"] = float(sum(self.pool_val))
        c["coach_conflict"] = float(sum(self.pair_val.values()))
        c["makespan"] = float(self._makespan())
        return c

    def _total(self) -> float:
        c = self.components()
        return sum(self.w[k] * c[k] for k in COMPONENTS)

    def try_move(self, changes: list[tuple[int, int, int, int]]):
        """changes: [(match, court, slot, ref)]. Applies if legal; returns undo info or None."""
        old = [(mi, *self.cells[mi]) for mi, *_ in changes]
        for mi, c, s, r in old:
            self._take(mi, c, s, r)
        placed = []
        for mi, c, s, r in changes:
            if not self._ok(mi, c, s, r):
                for pm in placed:
                    self._take(*pm)
                for o in old:
                    self._put(*o)
                return None
            self._put(mi, c, s, r)
            placed.append((mi, c, s, r))
        for mi, c, s, r in changes:
            self.cells[mi] = (c, s, r)
        return old

    def rescore(self, matches_touched):
        """Recompute the touched teams' pieces; return (new_total, saved state for undo)."""
        P = self.P
        teams = set()
        for mi in matches_touched:
            a, b, _ = P.matches[mi]
            teams.update((a, b))
        # refs could have changed: include every team of the touched pools
        pools = {P.matches[mi][2] for mi in matches_touched}
        for p in pools:
            teams.update(P.pool_members[p])
        saved = ({t: self.team_part[t] for t in teams}, {p: self.pool_val[p] for p in pools}, {}, self.total)
        for t in teams:
            self.team_part[t] = team_components(self.act[t])
        for p in pools:
            self.pool_val[p] = self._pool(p)
        for t in teams:
            for pr in self.pairs_of.get(t, ()):
                if pr not in saved[2]:
                    saved[2][pr] = self.pair_val[pr]
                    self.pair_val[pr] = self._pair(pr)
        self.total = self._total_fast()
        return saved

    def _total_fast(self) -> float:
        w = self.w
        idle = three = dw = 0
        for i, th, d, _ in self.team_part:
            idle += i
            three += th
            dw += d
        return (w["idle"] * idle + w["three_straight"] * three + w["double_work"] * dw
                + w["work_imbalance"] * sum(self.pool_val) + w["coach_conflict"] * sum(self.pair_val.values())
                + w["makespan"] * self._makespan())

    def undo(self, old, saved):
        # revert grid
        cur = [(mi, *self.cells[mi]) for mi, *_ in old]
        for c in cur:
            self._take(*c)
        for o in old:
            self._put(*o)
            self.cells[o[0]] = (o[1], o[2], o[3])
        tp, pv, pr, tot = saved
        for t, v in tp.items():
            self.team_part[t] = v
        for p, v in pv.items():
            self.pool_val[p] = v
        for k, v in pr.items():
            self.pair_val[k] = v
        self.total = tot


def _propose(st: _State, rng: random.Random, court_of_pool: list[int]):
    P = st.P
    kind = rng.random()
    mi = rng.randrange(len(P.matches))
    c, s, r = st.cells[mi]
    a, b, p = P.matches[mi]
    if kind < 0.35:  # swap two matches' cells on the same court
        same = [m for m in range(len(P.matches)) if m != mi and st.cells[m][0] == c]
        if not same:
            return None
        m2 = rng.choice(same)
        c2, s2, r2 = st.cells[m2]
        return [(mi, c, s2, r), (m2, c2, s, r2)]
    if kind < 0.70:  # move to an empty cell on its court
        empties = [x for x in range(st.T) if st.occ[c][x] == -1]
        if not empties:
            return None
        return [(mi, c, rng.choice(empties), r)]
    if kind < 0.95:  # change the ref
        options = [t for t in P.pool_members[p] if t not in (a, b, r)]
        if not options:
            return None
        return [(mi, c, s, rng.choice(options))]
    # swap two pools' courts
    if len(P.pools) < 2:
        return None
    p1, p2 = rng.sample(range(len(P.pools)), 2)
    c1, c2 = court_of_pool[p1], court_of_pool[p2]
    if c1 == c2:
        return None
    ch = []
    for m, (mc, ms, mr) in st.cells.items():
        mp = P.matches[m][2]
        if mp == p1:
            ch.append((m, c2, ms, mr))
        elif mp == p2:
            ch.append((m, c1, ms, mr))
    return ch


def anneal(inst: Instance, weights: dict[str, float], start: Schedule, *, seed: int = 0,
           n_moves: int = 100_000, restarts: int = 5, t0: float | None = None,
           cool: float = 0.995, cool_every: int = 100, time_limit_s: float = 60,
           method: str = "annealing") -> SearchResult:
    tstart = time.perf_counter()
    P = compile_problem(inst)
    start_cells = P.from_schedule(start)
    best_cells, best_total = dict(start_cells), None
    hit = False
    per_restart = time_limit_s / max(1, restarts)
    for rs in range(restarts):
        rng = random.Random(seed * 1000 + rs)
        st = _State(P, start_cells, weights)
        court_of_pool = list(P.pool_court)
        for mi, (c, _, _) in st.cells.items():
            court_of_pool[P.matches[mi][2]] = c
        if best_total is None:
            best_total = st.total
        temp = t0 if t0 is not None else max(1.0, max(weights.values()))
        r_best, r_best_cells = st.total, dict(st.cells)
        r_t0 = time.perf_counter()
        for k in range(n_moves):
            if k and k % cool_every == 0:
                temp *= cool
                if time.perf_counter() - r_t0 > per_restart:
                    hit = True
                    break
            ch = _propose(st, rng, court_of_pool)
            if not ch:
                continue
            before = st.total
            old = st.try_move(ch)
            if old is None:
                continue
            saved = st.rescore([m for m, *_ in ch])
            delta = st.total - before
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                # keep court_of_pool in sync after a pool-court swap
                for m, c, _, _ in ch:
                    court_of_pool[P.matches[m][2]] = c
                if st.total < r_best - 1e-9:
                    r_best, r_best_cells = st.total, dict(st.cells)
            else:
                st.undo(old, saved)
        if r_best < best_total - 1e-9:
            best_total, best_cells = r_best, r_best_cells
    sched = P.to_schedule(best_cells, method)
    return SearchResult(schedule=sched, cost=score(inst, sched, weights), feasible=not check_feasible(inst, sched),
                        nodes_expanded=0, max_frontier=0, runtime_s=time.perf_counter() - tstart,
                        hit_limit=hit, optimal_proven=False)


def solve_annealing(instance: Instance, weights: dict[str, float], *, seed: int = 0,
                    node_limit: int = 2_000_000, time_limit_s: float = 60,
                    n_moves: int = 50_000, restarts: int = 5, start: str = "greedy") -> SearchResult:
    """Start from greedy (default) or the template, then anneal."""
    t = time.perf_counter()
    if start == "template":
        from ..baselines import solve_template
        init = solve_template(instance, weights)
        if not init.feasible:
            start = "greedy"
    if start == "greedy":
        from .bestfirst import solve_greedy
        init = solve_greedy(instance, weights, node_limit=node_limit, time_limit_s=time_limit_s / 4)
    if not init.feasible:
        from ..baselines import solve_random
        init = solve_random(instance, weights, seed=seed)
    if not init.feasible:
        init.method = "annealing"
        return init
    remaining = max(1.0, time_limit_s - (time.perf_counter() - t))
    res = anneal(instance, weights, init.schedule, seed=seed, n_moves=n_moves, restarts=restarts,
                 time_limit_s=remaining)
    res.runtime_s = time.perf_counter() - t
    return res
