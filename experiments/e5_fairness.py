"""E5: which formats are fairest? Win rate of the best team (95% CI) by format and spread."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vsched.sim.fairness import simulate_spread
from vsched.sim.tournament import Format

from .common import BASELINE, RESULTS, SERIES, config, save, style


def formats(cfg):
    for k, cut, seeding in itertools.product(cfg["pool_sizes"], cfg["gold_cuts"], cfg["seedings"]):
        rest = [p for p in range(1, k + 1) if p not in cut]
        yield Format(cfg["n_teams"], k, [("gold", list(cut)), ("silver", rest)], "best_of_3", seeding)


def main():
    cfg = config()["e5"]
    rows = []
    for sd in cfg["spreads"]:
        for i, fmt in enumerate(formats(cfg)):
            r = simulate_spread(fmt, sd, cfg["n_runs"], seed=cfg["seed"] + 100 * sd + i)
            rows.append({"spread": sd, "pool_size": fmt.pool_size, "gold_cut": len(fmt.brackets[0][1]),
                         "seeding": fmt.seeding, "format": fmt.label,
                         **{k: round(v, 6) for k, v in r.items() if k != "champion_counts"}})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e5_fairness.csv", index=False)
    print(df.drop(columns="format").to_string(index=False))

    fig, axes = plt.subplots(1, len(cfg["spreads"]), figsize=(9, 3.4), sharey=False)
    for ax, sd in zip(axes, cfg["spreads"]):
        d = df[(df.spread == sd) & (df.seeding == "snake")]
        for j, cut in enumerate(sorted(d.gold_cut.unique())):
            e = d[d.gold_cut == cut].sort_values("pool_size")
            x = np.arange(len(e)) + (j - 0.5) * 0.3
            ax.errorbar(x, e.p_best_wins_gold, yerr=[e.p_best_wins_gold - e.ci_low, e.ci_high - e.p_best_wins_gold],
                        fmt="o", ms=6, color=SERIES[j], capsize=3, lw=1.5,
                        label=f"Gold = top {cut} per pool")
        ax.set_xticks(range(len(cfg["pool_sizes"])), [f"pools of {k}" for k in cfg["pool_sizes"]], fontsize=8)
        style(ax, f"Rating spread SD = {sd}", None, "P(best team wins Gold)" if sd == cfg["spreads"][0] else None)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fairness by format (snake seeding, 24 teams, 95% CI)", x=0.02, ha="left", fontsize=11)
    save(fig, "e5_fairness.png")


if __name__ == "__main__":
    main()
