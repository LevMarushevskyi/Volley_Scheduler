"""Anonymize a real division before it goes into data/real/.

Input: one JSON file {"instance": {...}, "schedule": {...}} in the repo's
formats, but with real team names in `id`/`team1`/`team2`/`ref` and real
club/coach names in `coach`. Output: <out>.instance.json and
<out>.schedule.json with teams -> T01, T02, ... (by seed) and clubs/coaches
-> C01, C02, ...; every other field (names, cities, venues) is dropped.

    python -m vsched.anonymize export.json data/real/div01
"""
from __future__ import annotations

import json
import sys

from .model import Assignment, Instance, Schedule, Team, save_instance, save_schedule


def anonymize(inst_d: dict, sched_d: dict | None, division_id: str) -> tuple[Instance, Schedule | None]:
    teams = sorted(inst_d["teams"], key=lambda t: int(t["seed"]))
    tmap = {str(t["id"]): f"T{i + 1:02d}" for i, t in enumerate(teams)}
    cmap: dict[str, str] = {}
    for t in teams:
        c = t.get("coach")
        if c and c not in cmap:
            cmap[c] = f"C{len(cmap) + 1:02d}"
    inst = Instance(
        instance_id=division_id,
        teams=tuple(Team(tmap[str(t["id"])], int(t["seed"]), float(t.get("rating", 1500.0)),
                         cmap.get(t.get("coach")) if t.get("coach") else None) for t in teams),
        pool_size=int(inst_d["pool_size"]), courts=int(inst_d["courts"]), start=inst_d["start"],
        end=inst_d["end"], match_minutes=int(inst_d["match_minutes"]), seed=0)
    sched = None
    if sched_d:
        out = []
        for a in sched_d["assignments"]:
            t1, t2 = sorted((tmap[str(a["team1"])], tmap[str(a["team2"])]))
            out.append(Assignment(f"{a['pool']}-{t1}-{t2}", str(a["pool"]), t1, t2, int(a["court"]),
                                  int(a["slot"]), tmap[str(a["ref"])]))
        sched = Schedule(division_id, "volleylens", out)
    return inst, sched


def main(argv: list[str]) -> None:
    src, out = argv
    with open(src) as f:
        d = json.load(f)
    name = out.rstrip("/").split("/")[-1]
    inst, sched = anonymize(d["instance"], d.get("schedule"), name)
    save_instance(inst, f"{out}.instance.json")
    if sched:
        save_schedule(sched, f"{out}.schedule.json")
    print(f"wrote {out}.instance.json" + (f" and {out}.schedule.json" if sched else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
