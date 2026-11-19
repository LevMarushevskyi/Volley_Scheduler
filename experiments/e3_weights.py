"""E3: does the method ranking hold when the cost weights change?"""
from __future__ import annotations

import pandas as pd

from vsched.generator import generate
from vsched.methods import METHODS

from .common import RESULTS, config, has_templates, weights

BIG = 10_000


def main():
    cfg = config()["e3"]
    methods = [m for m in cfg["methods"] if m != "template" or has_templates()]
    rows = []
    for name, over in cfg["settings"].items():
        W = weights(over)
        for i in range(cfg["n"]):
            seed = cfg["base_seed"] + i
            inst = generate("medium", seed)
            for m in methods:
                kw = cfg["annealing"] if m == "annealing" else {}
                r = METHODS[m](inst, W, seed=seed, time_limit_s=BIG, **kw)
                rows.append({"setting": name, "instance": inst.instance_id, "method": m,
                             "total": r.cost["total"] if r.feasible else float("nan")})
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby(["setting", "instance"]).total.rank(method="min")
    df.to_csv(RESULTS / "e3_weights.csv", index=False)
    table = df.groupby(["setting", "method"]).agg(mean_total=("total", "mean"), mean_rank=("rank", "mean")).round(2)
    table.to_csv(RESULTS / "e3_rank_table.csv")
    print(table.unstack(0).to_string())


if __name__ == "__main__":
    main()
