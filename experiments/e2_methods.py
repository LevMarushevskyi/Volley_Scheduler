"""E2: which method wins on medium and large instances, and how fast?
Figure: schedule cost by method relative to the template (or to random when
data/templates.json is not present yet)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vsched.cost import COMPONENTS
from vsched.generator import generate
from vsched.methods import METHODS

from .common import BASELINE, RESULTS, SERIES, config, has_templates, save, style, weights

BIG = 10_000
LABEL = {"template": "Template", "random": "Random", "greedy": "Greedy", "wastar2": "Weighted A* (w=2)",
         "astar": "A*", "annealing": "Annealing"}


def run_method(m, inst, W, seed, cfg):
    if m == "annealing":
        return METHODS[m](inst, W, seed=seed, time_limit_s=BIG, node_limit=cfg["search_node_limit"],
                          **cfg["annealing"])
    return METHODS[m](inst, W, seed=seed, node_limit=cfg["search_node_limit"], time_limit_s=BIG)


def main():
    cfg = config()["e2"]
    W = weights()
    methods = [m for m in cfg["methods"] if m != "template" or has_templates()]
    rows, timing = [], []
    for tier in ("medium", "large"):
        for i in range(cfg[tier]["n"]):
            seed = cfg[tier]["base_seed"] + i
            inst = generate(tier, seed)
            for m in methods:
                r = run_method(m, inst, W, seed, cfg)
                row = {"tier": tier, "instance": inst.instance_id, "teams": len(inst.teams),
                       "pool_size": inst.pool_size, "courts": inst.courts, "slots": inst.n_slots,
                       "method": m, "feasible": r.feasible, "hit_limit": r.hit_limit,
                       "nodes_expanded": r.nodes_expanded}
                row.update({k: (r.cost[k] if r.feasible else np.nan) for k in COMPONENTS + ("total",)})
                rows.append(row)
                timing.append({"tier": tier, "instance": inst.instance_id, "method": m, "runtime_s": r.runtime_s})
                print(tier, inst.instance_id, m, row["total"], f"{r.runtime_s:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e2_methods.csv", index=False)
    tdf = pd.DataFrame(timing)
    tdf.to_csv(RESULTS / "e2_methods_timing.csv", index=False)

    ref = "template" if has_templates() else "random"
    base = df[df.method == ref].set_index("instance")["total"]
    df["rel_cost"] = df.total / df.instance.map(base)
    summ = df.groupby(["tier", "method"]).agg(solved=("feasible", "mean"), mean_total=("total", "mean"),
                                              rel_to_ref=("rel_cost", "mean"),
                                              **{f"mean_{k}": (k, "mean") for k in COMPONENTS}).round(3)
    summ = summ.join(tdf.groupby(["tier", "method"]).runtime_s.median().rename("median_runtime_s").round(2))
    summ.to_csv(RESULTS / "e2_summary.csv")
    print(summ.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, tier in zip(axes, ("medium", "large")):
        d = df[df.tier == tier]
        xs = np.arange(len(methods))
        for x, m in zip(xs, methods):
            v = d[d.method == m].rel_cost
            solved = v.notna().mean()
            col = BASELINE if m in ("template", "random") else SERIES[0] if m != "annealing" else SERIES[1]
            if v.notna().any():
                ax.bar(x, v.mean(), width=0.6, color=col)
                ax.scatter(np.full(v.notna().sum(), x), v.dropna(), s=10, color="#0b0b0b", alpha=0.5, zorder=3)
            ax.text(x, 0.02, f"{solved:.0%}\nsolved", ha="center", va="bottom", fontsize=7,
                    color="white" if v.notna().any() and v.mean() > 0.15 else "#52514e")
        ax.axhline(1, color=BASELINE, ls="--", lw=1)
        ax.set_xticks(xs, [LABEL[m] for m in methods], rotation=30, ha="right", fontsize=8)
        style(ax, f"{tier.capitalize()} instances", None, f"cost / {ref} cost" if tier == "medium" else None)
    fig.suptitle(f"Schedule cost by method (lower is better; dashed = {ref})", x=0.02, ha="left", fontsize=11)
    save(fig, "e2_cost_by_method.png")


if __name__ == "__main__":
    main()
