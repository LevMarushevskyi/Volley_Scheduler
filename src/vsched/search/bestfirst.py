"""One best-first loop for UCS, A*, weighted A* and greedy best-first.

State space (Build Guide Phase 4):
  * The grid is filled one cell at a time in time order: slot 0 court 1,
    slot 0 court 2, ..., then slot 1, and so on.
  * State = (next cell index, placed matches, per-team summary). Placed
    matches are a bitmask (equivalent to a frozenset). The team summary is
    (last busy slot, play streak, ref streak, refs worked): all the future
    cost depends on.
  * Actions: put one remaining match of a pool that sits on this cell's court
    (both teams free this slot) plus a free ref from the same pool into the
    cell, or leave the cell empty.
  * Step cost g: penalties that become certain at placement (idle gap closed,
    3rd straight match, 2nd straight ref, coach conflict). Makespan is also
    charged as soon as it is certain: w_makespan each time the search opens a
    new slot while matches remain, so a last match in slot s pays s+1 slots,
    exactly the makespan component. Work imbalance is added when the last
    match is placed (the goal). Charging makespan early instead of at the goal
    leaves every schedule's cost unchanged but gives UCS a useful g; without
    it UCS sees g = 0 everywhere and enumerates the whole space.
  * Repeated states: a closed set keyed on a canonical state hash. Teams in a
    pool that have no coach are interchangeable (the cost never looks at
    their names), so two states that differ only by relabelling such teams
    get the same key. This is exact: merged states have identical futures.
The methods differ only in priority: UCS g, A* g+h, weighted A* g+w*h, greedy h.
"""
from __future__ import annotations

import heapq
import time
from itertools import count, permutations, product

from ..cost import COMPONENTS, cost as score
from ..model import Instance, SearchResult, Schedule
from .heuristics import imbalance_lb, lower_bound_end, pending_idle
from .problem import Problem, compile_problem


def _empty_result(inst: Instance, method: str, nodes: int, frontier: int, t0: float, hit: bool) -> SearchResult:
    c = dict.fromkeys(COMPONENTS, 0.0)
    c["total"] = float("inf")
    return SearchResult(schedule=Schedule(inst.instance_id, method, []), cost=c, feasible=False,
                        nodes_expanded=nodes, max_frontier=frontier,
                        runtime_s=time.perf_counter() - t0, hit_limit=hit, optimal_proven=False)


class _Canon:
    """Canonical per-pool state keys under relabelling of coach-free teams."""

    def __init__(self, P: Problem, has_coach: list[bool]):
        self.P = P
        self.members = P.pool_members
        self.local_pairs = []
        for p, mem in enumerate(P.pool_members):
            pos = {t: i for i, t in enumerate(mem)}
            self.local_pairs.append([(mi, pos[a], pos[b]) for mi, (a, b, mp) in enumerate(P.matches) if mp == p])
        self.fixed = [[i for i, t in enumerate(mem) if has_coach[t]] for mem in P.pool_members]
        self.cache: dict = {}

    def key(self, p: int, mask: int, teams) -> tuple:
        mem = self.members[p]
        placed = tuple((i, j) for mi, i, j in self.local_pairs[p] if mask >> mi & 1)
        sums = tuple(teams[t] for t in mem)
        ck = (p, placed, sums)
        hit = self.cache.get(ck)
        if hit is not None:
            return hit
        fixed = self.fixed[p]
        free = sorted((i for i in range(len(mem)) if i not in fixed), key=lambda i: sums[i])
        groups, cur = [], [free[0]] if free else []
        for i in free[1:]:
            if sums[i] == sums[cur[-1]]:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        if cur:
            groups.append(cur)
        best = None
        n_pl = len(placed)
        if n_pl == 0 or n_pl == len(self.local_pairs[p]) or all(len(g) == 1 for g in groups):
            perm_iter = [tuple(groups)]
        else:
            perm_iter = product(*(permutations(g) for g in groups))
        for gp in perm_iter:
            order = list(fixed)
            for g in gp:
                order.extend(g)
            newpos = [0] * len(mem)
            for k, i in enumerate(order):
                newpos[i] = k
            rel = tuple(sorted((min(newpos[i], newpos[j]), max(newpos[i], newpos[j])) for i, j in placed))
            if best is None or rel < best:
                best = rel
        order0 = list(fixed) + [i for g in groups for i in g]
        out = (tuple(sums[i] for i in order0), best)
        if len(self.cache) < 2_000_000:
            self.cache[ck] = out
        return out


def best_first(inst: Instance, weights: dict[str, float], *, mode: str, w: float = 1.0,
               heuristic: str = "full", node_limit: int = 2_000_000,
               time_limit_s: float = 60, method_name: str | None = None,
               pool_courts: dict[str, int] | None = None, symmetry: bool = True) -> SearchResult:
    """mode: 'ucs' | 'astar' | 'wastar' | 'greedy'; heuristic: 'makespan' | 'full'."""
    t0 = time.perf_counter()
    method = method_name or mode
    P: Problem = compile_problem(inst, pool_courts)
    C, T, nT = P.n_courts, P.n_slots, P.n_teams
    W_idle, W_three, W_dw = weights["idle"], weights["three_straight"], weights["double_work"]
    W_imb, W_cc, W_mk = weights["work_imbalance"], weights["coach_conflict"], weights["makespan"]
    full = (1 << len(P.matches)) - 1
    team_court = [P.pool_court[P.team_pool[t]] for t in range(nT)]
    court_pools = [[p for p in range(len(P.pools)) if P.pool_court[p] == q] for q in range(C)]
    pool_match_list = [[mi for mi, m in enumerate(P.matches) if m[2] == p] for p in range(len(P.pools))]
    use_h = mode != "ucs"
    use_full = heuristic == "full"
    has_coach = [bool(P.coach_mates[t]) for t in range(nT)]
    canon = _Canon(P, has_coach) if symmetry else None

    def h_of(idx, mask, teams, rem_court) -> float | None:
        s, q = divmod(idx, C)
        end = lower_bound_end(s, q, rem_court)
        if end >= T:
            return None
        if not use_h:
            return 0.0
        h = W_mk * (end - s)
        if use_full:
            h += W_idle * pending_idle(s, q, teams, mask, P, team_court)
            h += W_imb * imbalance_lb(mask, teams, P, pool_match_list)
        return h

    def terminal(teams) -> float:
        imb = 0
        for mem in P.pool_members:
            rs = [teams[t][3] for t in mem]
            imb += max(rs) - min(rs)
        return W_imb * imb

    def prio(g, h, depth):
        if mode == "ucs":
            return (g, -depth)
        if mode == "astar":
            return (g + h, -depth)
        if mode == "wastar":
            return (g + w * h, -depth)
        return (h, -depth, g)  # greedy

    def pool_keys(mask, teams):
        if canon is None:
            return (mask, teams)
        return tuple(canon.key(p, mask, teams) for p in range(len(P.pools)))

    teams0 = tuple((-1, 0, 0, 0) for _ in range(nT))
    rem_court0 = tuple(len(m) for m in P.court_matches)
    h0 = h_of(0, 0, teams0, rem_court0)
    if h0 is None:
        return _empty_result(inst, method, 0, 0, t0, False)
    tie = count()
    g0 = W_mk if P.matches else 0.0
    pk0 = pool_keys(0, teams0)
    root = (0, 0, teams0, rem_court0, g0, None, None, pk0)
    frontier = [(prio(g0, h0, 0), next(tie), root)]
    best_g: dict = {(0, pk0): g0}
    closed: set = set()
    nodes = 0
    max_frontier = 1
    hit = False

    while frontier:
        _, _, node = heapq.heappop(frontier)
        idx, mask, teams, rem_court, g, parent, action, pk = node
        if mask == full:
            cells = {}
            n = node
            while n[5] is not None:
                if n[6] is not None:
                    mi, cq, cs, cr = n[6]
                    cells[mi] = (cq, cs, cr)
                n = n[5]
            sched = P.to_schedule(cells, method)
            c = score(inst, sched, weights)
            return SearchResult(schedule=sched, cost=c, feasible=True, nodes_expanded=nodes,
                                max_frontier=max_frontier, runtime_s=time.perf_counter() - t0,
                                hit_limit=False, optimal_proven=mode in ("ucs", "astar"))
        key = (idx, pk)
        if key in closed:
            continue
        closed.add(key)
        nodes += 1
        if nodes >= node_limit or (nodes % 2000 == 0 and time.perf_counter() - t0 > time_limit_s):
            hit = True
            break

        s, q = divmod(idx, C)
        nidx = idx + 1
        children = [(mask, teams, rem_court, g, None, pk)]  # leave the cell empty
        for p in court_pools[q]:
            members = P.pool_members[p]
            free = [t for t in members if teams[t][0] != s]
            if len(free) < 3:
                continue
            for mi in pool_match_list[p]:
                if mask >> mi & 1:
                    continue
                a, b, _ = P.matches[mi]
                if teams[a][0] == s or teams[b][0] == s:
                    continue
                for r in free:
                    if r == a or r == b:
                        continue
                    step = 0.0
                    nt = list(teams)
                    for t, kind in ((a, 1), (b, 1), (r, 2)):
                        last, ps, rs, refs = nt[t]
                        cont = last == s - 1
                        if last >= 0 and not cont:
                            step += W_idle * (s - last - 1)
                        if kind == 1:
                            ps = ps + 1 if cont else 1
                            rs = 0
                            if ps >= 3:
                                step += W_three
                        else:
                            rs = rs + 1 if cont else 1
                            ps = 0
                            refs += 1
                            if rs >= 2:
                                step += W_dw
                        nt[t] = (s, ps, rs, refs)
                    if W_cc:
                        trio = (a, b, r)
                        for i, t in enumerate(trio):
                            for u in P.coach_mates[t]:
                                if teams[u][0] == s or u in trio[:i]:
                                    step += W_cc
                    rc = list(rem_court)
                    rc[q] -= 1
                    nmask = mask | (1 << mi)
                    nteams = tuple(nt)
                    if canon is not None:
                        npk = list(pk)
                        npk[p] = canon.key(p, nmask, nteams)
                        npk = tuple(npk)
                    else:
                        npk = (nmask, nteams)
                    children.append((nmask, nteams, tuple(rc), g + step, (mi, q, s, r), npk))
        for cmask, cteams, crc, cg, act, cpk in children:
            if cmask == full:
                cg = cg + terminal(cteams)
                h = 0.0
            else:
                if nidx >= C * T:
                    continue
                if nidx % C == 0:
                    cg = cg + W_mk
                h = h_of(nidx, cmask, cteams, crc)
                if h is None:
                    continue
            ckey = (nidx, cpk)
            if ckey in closed:
                continue
            bg = best_g.get(ckey)
            if bg is not None and bg <= cg:
                continue
            best_g[ckey] = cg
            child = (nidx, cmask, cteams, crc, cg, node, act, cpk)
            heapq.heappush(frontier, (prio(cg, h, nidx), next(tie), child))
        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

    return _empty_result(inst, method, nodes, max_frontier, t0, hit)


def _solver(mode: str, w: float = 1.0, name: str | None = None, heuristic: str = "full"):
    def solve(instance: Instance, weights: dict[str, float], *, seed: int = 0,
              node_limit: int = 2_000_000, time_limit_s: float = 60) -> SearchResult:
        return best_first(instance, weights, mode=mode, w=w, heuristic=heuristic,
                          node_limit=node_limit, time_limit_s=time_limit_s, method_name=name or mode)
    solve.__name__ = f"solve_{(name or mode).replace('.', '')}"
    return solve


solve_ucs = _solver("ucs", name="ucs")
solve_astar = _solver("astar", name="astar")
solve_astar_basic = _solver("astar", name="astar_basic", heuristic="makespan")
solve_wastar15 = _solver("wastar", 1.5, name="wastar1.5")
solve_wastar2 = _solver("wastar", 2.0, name="wastar2")
solve_greedy = _solver("greedy", name="greedy", heuristic="makespan")
