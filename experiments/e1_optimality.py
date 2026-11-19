"""E1: how close does each method get to the true best on tiny instances?
Also: nodes expanded vs instance size for UCS / A* (basic h) / A* (full h)."""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from vsched.baselines import solve_random, solve_template
from vsched.generator import generate, make_instance
from vsched.search.annealing import solve_annealing
from vsched.search.bestfirst import (solve_astar, solve_astar_basic, solve_greedy, solve_ucs,
                                     solve_wastar15, solve_wastar2)

from .common import RESULTS, SERIES, BASELINE, config, has_templates, save, style, weights

BIG = 10_000


def main():
    cfg = config()["e1"]
    W = weights()
    rows, timing = [], []
    for i in range(cfg["n_tiny"]):
        seed = cfg["base_seed"] + i
        inst = generate("tiny", seed)
        ucs = solve_ucs(inst, W, node_limit=cfg["ucs_node_limit"], time_limit_s=BIG)
        opt = ucs.cost["total"]
        runs = {"ucs": ucs,
                "astar": solve_astar(inst, W, time_limit_s=BIG),
                "astar_basic": solve_astar_basic(inst, W, time_limit_s=BIG),
                "wastar1.5": solve_wastar15(inst, W, time_limit_s=BIG),
                "wastar2": solve_wastar2(inst, W, time_limit_s=BIG),
                "greedy": solve_greedy(inst, W, time_limit_s=BIG),
                "annealing": solve_annealing(inst, W, seed=seed, time_limit_s=BIG, **cfg["annealing"]),
                "random": solve_random(inst, W, seed=seed)}
        if has_templates():
            runs["template"] = solve_template(inst, W)
        for m, r in runs.items():
            rows.append({"instance": inst.instance_id, "teams": len(inst.teams), "pool_size": inst.pool_size,
                         "courts": inst.courts, "method": m, "cost": r.cost["total"], "optimal": opt,
                         "gap": r.cost["total"] - opt, "is_optimal": r.cost["total"] == opt,
                         "feasible": r.feasible, "nodes_expanded": r.nodes_expanded,
                         "max_frontier": r.max_frontier})
            timing.append({"instance": inst.instance_id, "method": m, "runtime_s": r.runtime_s})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e1_tiny.csv", index=False)
    pd.DataFrame(timing).to_csv(RESULTS / "e1_tiny_timing.csv", index=False)
    summ = df.groupby("method").agg(mean_gap=("gap", "mean"), max_gap=("gap", "max"),
                                    share_optimal=("is_optimal", "mean"),
                                    mean_nodes=("nodes_expanded", "mean")).round(3)
    summ.to_csv(RESULTS / "e1_gap_table.csv")
    print(summ.to_string())
    bad = df[(df.method == "astar") & (df.cost != df.optimal)]
    assert bad.empty, "A* disagrees with UCS: heuristic overestimates"

    # nodes vs size
    scan = []
    for n, k, c in cfg["size_scan"]:
        inst = make_instance(n, k, c, seed=cfg["base_seed"], coach_frac=0.0, slack_slots=1,
                             instance_id=f"scan-{n}x{k}")
        n_matches = (n // k) * k * (k - 1) // 2
        for m, f in (("ucs", solve_ucs), ("astar_basic", solve_astar_basic), ("astar", solve_astar)):
            r = f(inst, W, node_limit=cfg["size_scan_node_limit"], time_limit_s=BIG)
            scan.append({"instance": inst.instance_id, "teams": n, "pools": n // k, "pool_size": k,
                         "matches": n_matches, "method": m, "nodes_expanded": r.nodes_expanded,
                         "finished": not r.hit_limit, "cost": r.cost["total"]})
    sd = pd.DataFrame(scan)
    sd.to_csv(RESULTS / "e1_nodes_vs_size.csv", index=False)
    print(sd.to_string(index=False))

    plot_nodes(sd, cfg["size_scan_node_limit"])


def plot_nodes(sd: pd.DataFrame, limit: int):
    import numpy as np
    sd = sd.copy()
    sd["config"] = sd.pools.astype(str) + " pool" + np.where(sd.pools > 1, "s", "") + " of " + sd.pool_size.astype(str)
    order = sd.drop_duplicates("config").sort_values(["matches", "pools"]).config.tolist()
    labels = {"ucs": "UCS", "astar_basic": "A* (makespan h)", "astar": "A* (full h)"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    xs = np.arange(len(order))
    for j, (m, col) in enumerate(zip(labels, SERIES)):
        d = sd[sd.method == m].set_index("config").loc[order]
        x = xs + (j - 1) * 0.27
        ax.bar(x, d.nodes_expanded, width=0.25, color=np.where(d.finished, col, "white"),
               edgecolor=col, linewidth=1.5, label=labels[m])
    ax.axhline(limit, color=BASELINE, ls="--", lw=1)
    ax.text(-0.45, limit * 1.25, "node limit; hollow bar = did not finish", color=BASELINE, fontsize=8)
    ax.set_yscale("log")
    mcount = sd.drop_duplicates("config").set_index("config").loc[order].matches
    ax.set_xticks(xs, [f"{c}\n({n} matches)" for c, n in zip(order, mcount)], fontsize=8)
    style(ax, "Nodes expanded to prove the optimum", None, "nodes expanded (log)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))
    save(fig, "e1_nodes_vs_size.png")


if __name__ == "__main__":
    main()
