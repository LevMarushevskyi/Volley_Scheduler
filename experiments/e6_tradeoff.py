"""E6 (Phase 7): convenience vs fairness, one point per format.

For each pool size the 24-team pool-play day is built with the best method
that finishes (A* if it proves optimality within the node limit, otherwise
annealing from greedy) and scored with the cost function (x). The same format
is simulated for fairness (y). The Gold cut and seeding do not change the
pool-play schedule, so formats that share a pool size share x.
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from vsched.generator import make_instance
from vsched.search.annealing import solve_annealing
from vsched.search.bestfirst import solve_astar
from vsched.sim.fairness import simulate_spread
from vsched.sim.tournament import Format

from .common import RESULTS, SERIES, config, save, style, weights

BIG = 10_000


def build(cfg, k, W):
    inst = make_instance(cfg["n_teams"], k, cfg["courts"], seed=cfg["instance_seed"], start=cfg["start"],
                         end=cfg["end"], match_minutes=cfg["match_minutes"], instance_id=f"e6-pools{k}")
    r = solve_astar(inst, W, node_limit=cfg["node_limit"], time_limit_s=BIG)
    method = "astar"
    if not r.feasible:
        r = solve_annealing(inst, W, seed=cfg["instance_seed"], time_limit_s=BIG, **cfg["annealing"])
        method = "annealing"
    return inst, r, method


def main():
    cfg = config()["e6"]
    W = weights()
    rows = []
    for k in (3, 4, 6):
        inst, r, method = build(cfg, k, W)
        print(f"pools of {k}: {method} cost {r.cost['total']} feasible={r.feasible}", flush=True)
        for cut in ([1], [1, 2]):
            rest = [p for p in range(1, k + 1) if p not in cut]
            fmt = Format(cfg["n_teams"], k, [("gold", cut), ("silver", rest)])
            f = simulate_spread(fmt, cfg["spread"], cfg["n_runs"], seed=cfg["seed"] + 10 * k + len(cut))
            rows.append({"pool_size": k, "gold_cut": len(cut), "label": f"pools of {k}, top {len(cut)}",
                         "schedule_method": method, "convenience_cost": r.cost["total"],
                         "makespan_slots": r.cost["makespan"], "idle": r.cost["idle"],
                         "coach_conflict": r.cost["coach_conflict"],
                         **{k2: round(v, 6) for k2, v in f.items() if k2 != "champion_counts"}})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e6_tradeoff.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for j, cut in enumerate((1, 2)):
        d = df[df.gold_cut == cut]
        ax.errorbar(d.convenience_cost, d.p_best_wins_gold,
                    yerr=[d.p_best_wins_gold - d.ci_low, d.ci_high - d.p_best_wins_gold],
                    fmt="o" if cut == 1 else "s", ms=8, color=SERIES[j], capsize=3, lw=1.5,
                    label=f"Gold = top {cut} per pool")
        for _, row in d.iterrows():
            ax.annotate(f"pools of {row.pool_size}", (row.convenience_cost, row.p_best_wins_gold),
                        textcoords="offset points", xytext=(8, 6 if cut == 2 else -12), fontsize=8, color="#52514e")
    style(ax, "Convenience vs fairness (24 teams)", "schedule cost (lower = more convenient)",
          f"P(best team wins Gold), rating SD {cfg['spread']}")
    ax.legend(frameon=False, fontsize=8, loc="best")
    save(fig, "e6_tradeoff.png")


if __name__ == "__main__":
    main()
