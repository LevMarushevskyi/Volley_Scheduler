"""Tables for showing a schedule (used by the app; no scheduling logic here)."""
from __future__ import annotations

import pandas as pd

from .cost import IDLE, PLAY, REF, activity_grid
from .model import Instance, Schedule, Team
from .roundrobin import instance_pools

POOL_COLORS = ["#d6e6f8", "#fbdccd", "#cdeee1", "#fbe9bf", "#f8dbe6", "#cfe7cf",
               "#dcd8f6", "#f8d4d3"]


def team_labels(inst: Instance) -> dict[str, str]:
    """team id -> pool position label like 'A1' (1 = best seed in the pool)."""
    out = {}
    for p, members in instance_pools(inst).items():
        for i, t in enumerate(sorted(members, key=lambda t: t.seed), start=1):
            out[t.id] = f"{p}{i}"
    return out


def pool_color(pool: str, pools: list[str]) -> str:
    return POOL_COLORS[pools.index(pool) % len(POOL_COLORS)]


def grid_frame(inst: Instance, sched: Schedule) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Court x time grid. Returns (text frame, background-color frame)."""
    lab = team_labels(inst)
    pools = list(instance_pools(inst))
    T = max([inst.n_slots] + [a.slot + 1 for a in sched.assignments])
    idx = [inst.slot_time(s) for s in range(T)]
    cols = [f"Court {c}" for c in range(1, inst.courts + 1)]
    text = pd.DataFrame("", index=idx, columns=cols)
    color = pd.DataFrame("", index=idx, columns=cols)
    for a in sched.assignments:
        text.iat[a.slot, a.court - 1] = f"{lab[a.team1]} vs {lab[a.team2]} · ref {lab[a.ref]}"
        color.iat[a.slot, a.court - 1] = f"background-color: {pool_color(a.pool, pools)}; color: #0b0b0b"
    return text, color


def timeline_frame(inst: Instance, sched: Schedule) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team x time: Play / Ref / idle-gap (·) / blank outside a team's day."""
    lab = team_labels(inst)
    grid = activity_grid(inst, sched)
    T = len(next(iter(grid.values()))) if grid else 0
    cols = [inst.slot_time(s) for s in range(T)]
    rows, colors, names = [], [], []
    for t in sorted(inst.teams, key=lambda t: lab[t.id]):
        row = grid[t.id]
        busy = [i for i, x in enumerate(row) if x != IDLE]
        r, c = [], []
        for i, x in enumerate(row):
            if x == PLAY:
                r.append("Play"); c.append("background-color: #2a78d6; color: white")
            elif x == REF:
                r.append("Ref"); c.append("background-color: #eb6834; color: white")
            elif busy and busy[0] < i < busy[-1]:
                r.append("·"); c.append("background-color: #f1d0cf; color: #52514e")
            else:
                r.append(""); c.append("")
        rows.append(r); colors.append(c)
        names.append(f"{lab[t.id]} {t.id}" + (f" ({t.coach})" if t.coach else ""))
    return pd.DataFrame(rows, index=names, columns=cols), pd.DataFrame(colors, index=names, columns=cols)


def schedule_csv(inst: Instance, sched: Schedule) -> str:
    lab = team_labels(inst)
    rows = [{"time": inst.slot_time(a.slot), "slot": a.slot, "court": a.court, "pool": a.pool,
             "team1": a.team1, "team2": a.team2, "ref": a.ref,
             "label": f"{lab[a.team1]} vs {lab[a.team2]} ref {lab[a.ref]}"}
            for a in sorted(sched.assignments, key=lambda a: (a.slot, a.court))]
    return pd.DataFrame(rows).to_csv(index=False)


def parse_team_list(text: str) -> list[Team]:
    """Lines of 'name, seed[, coach]' (seed optional: line order). Ratings default to 1500."""
    teams = []
    for i, line in enumerate(l for l in text.splitlines() if l.strip()):
        parts = [p.strip() for p in line.split(",")]
        name = parts[0]
        seed = int(parts[1]) if len(parts) > 1 and parts[1] else i + 1
        coach = parts[2] if len(parts) > 2 and parts[2] else None
        teams.append(Team(id=name, seed=seed, rating=1500.0, coach=coach))
    ids = [t.id for t in teams]
    if len(set(ids)) != len(ids):
        raise ValueError("team names must be unique")
    return teams
