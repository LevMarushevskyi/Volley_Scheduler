"""Registry: every method has the same signature
solve(instance, weights, *, seed, node_limit=2_000_000, time_limit_s=60) -> SearchResult
"""
from __future__ import annotations

from .baselines import solve_random, solve_template
from .model import Instance, SearchResult
from .search.annealing import solve_annealing
from .search.bestfirst import (solve_astar, solve_astar_basic, solve_greedy, solve_ucs,
                               solve_wastar15, solve_wastar2)

METHODS = {
    "ucs": solve_ucs,
    "astar": solve_astar,
    "astar_basic": solve_astar_basic,
    "wastar1.5": solve_wastar15,
    "wastar2": solve_wastar2,
    "greedy": solve_greedy,
    "annealing": solve_annealing,
    "random": solve_random,
    "template": solve_template,
}

EXACT = ("ucs", "astar", "astar_basic")


def solve(method: str, instance: Instance, weights: dict[str, float], *, seed: int = 0,
          node_limit: int = 2_000_000, time_limit_s: float = 60, fallback: bool = False) -> SearchResult:
    """Run one method. With fallback=True a search that hits its limit returns
    the greedy schedule instead (result.schedule.method says 'greedy-fallback')."""
    res = METHODS[method](instance, weights, seed=seed, node_limit=node_limit, time_limit_s=time_limit_s)
    if fallback and not res.feasible and method not in ("greedy", "random", "template"):
        g = solve_greedy(instance, weights, seed=seed, time_limit_s=time_limit_s)
        g.schedule.method = f"{method}->greedy-fallback"
        g.hit_limit = True
        g.runtime_s += res.runtime_s
        g.nodes_expanded += res.nodes_expanded
        return g
    return res
