"""E7 (stretch): which schedules still finish on time when matches run long?"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from vsched.generator import generate
from vsched.methods import METHODS
from vsched.sim.durations import simulate_delays

from .common import BASELINE, RESULTS, SERIES, config, has_templates, save, style, weights

BIG = 10_000


def main():
    cfg = config()["e7"]
    W = weights()
    methods = (["template"] if has_templates() else ["random"]) + ["greedy", "annealing"]
    rows = []
    for i in range(cfg["n"]):
        seed = cfg["base_seed"] + i
        inst = generate("medium", seed)
        for m in methods:
            kw = {"n_moves": 20000, "restarts": 3} if m == "annealing" else {}
            r = METHODS[m](inst, W, seed=seed, time_limit_s=BIG, **kw)
            if not r.feasible:
                continue
            d = simulate_delays(inst, r.schedule, n_reps=cfg["n_reps"], seed=cfg["seed"] + i, cv=cfg["cv"])
            for k in range(cfg["n_reps"]):
                rows.append({"instance": inst.instance_id, "method": m, "rep": k,
                             "overrun_min": round(float(d["overrun_min"][k]), 4),
                             "day_end_min": round(float(d["day_end_min"][k]), 4),
                             "max_delay_min": round(float(d["max_delay_min"][k]), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e7_delays.csv", index=False)
    print(df.groupby("method")[["day_end_min", "overrun_min", "max_delay_min"]].describe().round(1).T.to_string())
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    data = [df[df.method == m].overrun_min for m in methods]
    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, showfliers=False, widths=0.5)
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(BASELINE if m in ("template", "random") else SERIES[0] if m == "greedy" else SERIES[1])
    style(ax, "Day overrun under lognormal match lengths", None, "minutes past planned end")
    save(fig, "e7_delays.png")


if __name__ == "__main__":
    main()
