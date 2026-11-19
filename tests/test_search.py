import itertools

import pytest

from vsched.baselines import solve_random, solve_template, load_templates
from vsched.cost import check_feasible, cost
from vsched.generator import generate
from vsched.methods import METHODS, solve
from vsched.model import Assignment, Schedule
from vsched.roundrobin import default_pool_courts, instance_matches, instance_pools
from vsched.search.annealing import solve_annealing
from vsched.search.bestfirst import best_first, solve_astar, solve_astar_basic, solve_greedy, solve_ucs

from conftest import make

TINY = [generate("tiny", s) for s in range(20)]
FIXTURE_TPL = load_templates()  # data/templates.json


@pytest.mark.parametrize("inst", TINY, ids=lambda i: i.instance_id)
def test_astar_equals_ucs(inst, W):
    u = solve_ucs(inst, W)
    a = solve_astar(inst, W)
    b = solve_astar_basic(inst, W)
    assert u.feasible and a.feasible and b.feasible and u.optimal_proven
    assert a.cost["total"] == u.cost["total"] == b.cost["total"]
    assert a.nodes_expanded <= u.nodes_expanded
    for r in (u, a, b):
        assert check_feasible(inst, r.schedule) == []


def _brute_force(inst, W):
    """Exhaustive optimum under the court rule (only for very small instances)."""
    pools = instance_pools(inst)
    pc = default_pool_courts(inst)
    matches = instance_matches(inst)
    per_pool = {}
    for p, mem in pools.items():
        ms = [m for m in matches if m.pool == p]
        opts = []
        for slots in itertools.permutations(range(inst.n_slots), len(ms)):
            for refs in itertools.product(*[[t.id for t in mem if t.id not in (m.team1, m.team2)] for m in ms]):
                busy = {}
                ok = True
                for m, s, r in zip(ms, slots, refs):
                    for t in (m.team1, m.team2, r):
                        if (t, s) in busy:
                            ok = False
                        busy[(t, s)] = 1
                if ok:
                    opts.append([Assignment(m.match_id, p, m.team1, m.team2, pc[p], s, r) for m, s, r in zip(ms, slots, refs)])
        per_pool[p] = opts
    best = float("inf")
    for combo in itertools.product(*per_pool.values()):
        sch = Schedule(inst.instance_id, "bf", [a for part in combo for a in part])
        best = min(best, cost(inst, sch, W)["total"])
    return best


@pytest.mark.parametrize("n,k,courts,slots,coaches", [
    (3, 3, 1, 5, None),
    (6, 3, 2, 4, {"T01": "K1", "T02": "K1"}),
    (6, 3, 1, 6, {"T04": "K1", "T03": "K1"}),
])
def test_ucs_matches_brute_force(n, k, courts, slots, coaches, W):
    inst = make(n, k, courts, slots, coaches)
    assert solve_ucs(inst, W).cost["total"] == _brute_force(inst, W)


def test_weighted_astar_within_bound(W):
    for inst in TINY[:8]:
        opt = solve_ucs(inst, W).cost["total"]
        for w in (1.5, 2.0):
            r = best_first(inst, W, mode="wastar", w=w)
            assert opt <= r.cost["total"] <= w * opt + 1e-9


@pytest.mark.parametrize("seed", range(100))
def test_baselines_and_greedy_feasible_on_generated(seed, W):
    tier = ("tiny", "medium", "large")[seed % 3]
    inst = generate(tier, seed)
    for r in (solve_random(inst, W, seed=seed), solve_greedy(inst, W, time_limit_s=10),
              solve_template(inst, W, templates=FIXTURE_TPL)):
        if r is None:
            continue
        if r.schedule.method == "template" and not r.feasible:
            continue
        assert r.feasible, (r.schedule.method, check_feasible(inst, r.schedule))
        assert check_feasible(inst, r.schedule) == []


@pytest.mark.parametrize("seed", range(10))
def test_annealing_feasible_and_no_worse_than_start(seed, W):
    inst = generate(("tiny", "medium")[seed % 2], seed)
    g = solve_greedy(inst, W)
    a = solve_annealing(inst, W, seed=seed, n_moves=3000, restarts=2)
    assert a.feasible and check_feasible(inst, a.schedule) == []
    assert a.cost["total"] <= g.cost["total"]


def test_annealing_reaches_optimum_on_most_tiny(W):
    hits = 0
    for inst in TINY[:10]:
        opt = solve_ucs(inst, W).cost["total"]
        hits += solve_annealing(inst, W, seed=1, n_moves=10000, restarts=3).cost["total"] == opt
    assert hits >= 8


def test_registry_signature_and_fallback(W):
    inst = generate("medium", 0)
    for name in ("greedy", "random"):
        assert METHODS[name](inst, W, seed=0, node_limit=1000, time_limit_s=5).feasible
    r = solve("astar", inst, W, node_limit=50, time_limit_s=5, fallback=True)
    assert r.feasible and r.hit_limit and "fallback" in r.schedule.method


def test_templates_file_is_valid_round_robin():
    tpl = load_templates()
    assert set(tpl) >= {3, 4, 5, 6}
    for k, seq in tpl.items():
        pairs = {frozenset(e[:2]) for e in seq}
        assert len(seq) == len(pairs) == k * (k - 1) // 2
        slots = [e[3] for e in seq]
        assert len(set(slots)) == len(slots)
        for a, b, r, _ in seq:
            assert r not in (a, b) and {a, b, r} <= set(range(1, k + 1))


def test_template_matches_lev_sheet(W):
    inst = make(4, 4, 1, 8)
    r = solve_template(inst, W)
    got = [(a.slot, a.team1, a.team2, a.ref) for a in sorted(r.schedule.assignments, key=lambda a: a.slot)]
    assert got == [(0, "T01", "T03", "T02"), (1, "T02", "T04", "T01"), (2, "T01", "T04", "T03"),
                   (3, "T02", "T03", "T04"), (4, "T03", "T04", "T02"), (5, "T01", "T02", "T03")]
