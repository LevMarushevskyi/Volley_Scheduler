"""E4: our methods vs VolleyLens on real (anonymized) divisions.

Reads pairs data/real/<name>.instance.json + data/real/<name>.schedule.json.
If none are present, it says so and exits cleanly; every other result runs
from generated instances.
"""
from __future__ import annotations

import pandas as pd

from vsched.cost import check_feasible, cost
from vsched.methods import METHODS
from vsched.model import load_instance, load_schedule

from .common import ROOT, RESULTS, config, has_templates, weights

BIG = 10_000


def main():
    real = ROOT / "data" / "real"
    pairs = sorted(p for p in real.glob("*.instance.json"))
    if not pairs:
        print("E4: no real divisions in data/real/ - skipped (see data/real/README.md)")
        return
    W = weights()
    methods = ["greedy", "annealing"] + (["template"] if has_templates() else [])
    cfg = config()["e2"]
    rows = []
    for ip in pairs:
        inst = load_instance(ip)
        sp = ip.with_name(ip.name.replace(".instance.json", ".schedule.json"))
        if sp.exists():
            vl = load_schedule(sp)
            v = check_feasible(inst, vl)
            rows.append({"division": inst.instance_id, "teams": len(inst.teams), "method": "volleylens",
                         "feasible": not v, "violations": len(v), **cost(inst, vl, W)})
        for m in methods:
            kw = cfg["annealing"] if m == "annealing" else {}
            r = METHODS[m](inst, W, seed=0, time_limit_s=BIG, **kw)
            rows.append({"division": inst.instance_id, "teams": len(inst.teams), "method": m,
                         "feasible": r.feasible, "violations": 0, **r.cost})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e4_real.csv", index=False)
    print(df.pivot_table(index="division", columns="method", values="total").round(1).to_string())


if __name__ == "__main__":
    main()
