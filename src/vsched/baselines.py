"""Baselines: random placement and the standard pool-sheet template."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from .cost import cost as score, check_feasible
from .model import Instance, SearchResult, Schedule
from .search.problem import compile_problem

_REPO = Path(__file__).resolve().parents[2]
TEMPLATES_PATH = _REPO / "data" / "templates.json"


def _result(inst, sched, weights, t0) -> SearchResult:
    feas = not check_feasible(inst, sched)
    return SearchResult(schedule=sched, cost=score(inst, sched, weights), feasible=feas,
                        nodes_expanded=0, max_frontier=0, runtime_s=time.perf_counter() - t0,
                        hit_limit=False, optimal_proven=False)


# ------------------------------------------------------------------ random
def solve_random(instance: Instance, weights: dict[str, float], *, seed: int = 0,
                 node_limit: int = 2_000_000, time_limit_s: float = 60, max_tries: int = 500) -> SearchResult:
    """Shuffle the matches, put each in the first legal cell on its pool's court,
    pick a random legal ref. Retry with a new shuffle if stuck."""
    t0 = time.perf_counter()
    P = compile_problem(instance)
    rng = random.Random(seed)
    for _ in range(max_tries):
        order = list(range(len(P.matches)))
        rng.shuffle(order)
        busy = [set() for _ in range(P.n_slots)]       # teams busy per slot
        used = set()                                   # (court, slot)
        cells = {}
        ok = True
        for mi in order:
            a, b, p = P.matches[mi]
            q = P.pool_court[p]
            placed = False
            for s in range(P.n_slots):
                if (q, s) in used or a in busy[s] or b in busy[s]:
                    continue
                refs = [r for r in P.pool_members[p] if r not in (a, b) and r not in busy[s]]
                if not refs:
                    continue
                r = rng.choice(refs)
                used.add((q, s))
                busy[s].update((a, b, r))
                cells[mi] = (q, s, r)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return _result(instance, P.to_schedule(cells, "random"), weights, t0)
    return _result(instance, Schedule(instance.instance_id, "random", []), weights, t0)


# ---------------------------------------------------------------- template
def load_templates(path: str | Path | None = None) -> dict[int, list[tuple[int, int, int, int]]]:
    """templates.json -> {pool size: [(pos_a, pos_b, ref_pos, slot), ...]}.

    File format (Lev's pool sheets), keyed by pool size:
        {"4": [{"slot": 1, "teams": [1, 3], "ref": 2}, ...], ...}
    Positions are 1-based places in the pool's seed order (1 = best seed in
    the pool); slots are 1-based and relative to the pool's start on its
    court. Keys starting with "_" are comments.
    """
    p = Path(path) if path else TEMPLATES_PATH
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: the template baseline needs the pool sheets")
    with open(p) as f:
        data = json.load(f)
    out = {}
    for k, entries in data.items():
        if k.startswith("_"):
            continue
        seq = []
        for i, e in enumerate(entries):
            a, b = e["teams"]
            seq.append((int(a), int(b), int(e["ref"]), int(e.get("slot", i + 1))))
        out[int(k)] = sorted(seq, key=lambda x: x[3])
    return out


def solve_template(instance: Instance, weights: dict[str, float], *, seed: int = 0,
                   node_limit: int = 2_000_000, time_limit_s: float = 60,
                   templates: dict[int, list[tuple[int, int, int, int]]] | None = None) -> SearchResult:
    """Lay each pool's sheet onto its court at the sheet's slots.
    Pools sharing a court run back to back in pool order."""
    t0 = time.perf_counter()
    tpls = templates if templates is not None else load_templates()
    P = compile_problem(instance)
    mindex = {(min(a, b), max(a, b)): i for i, (a, b, _) in enumerate(P.matches)}
    next_slot = [0] * P.n_courts
    cells = {}
    for p, members in enumerate(P.pool_members):  # members are in seed order
        k = len(members)
        if k not in tpls:
            raise KeyError(f"no template for pool size {k} in templates.json")
        q = P.pool_court[p]
        start = next_slot[q]
        for (a, b, r, slot) in tpls[k]:
            ta, tb, tr = members[a - 1], members[b - 1], members[r - 1]
            mi = mindex[(min(ta, tb), max(ta, tb))]
            cells[mi] = (q, start + slot - 1, tr)
            next_slot[q] = max(next_slot[q], start + slot)
    return _result(instance, P.to_schedule(cells, "template"), weights, t0)
