"""Problem model: teams, instances, schedules, and the shared method result.

These dataclasses and their JSON formats are the locked interface every
method, baseline, experiment and app tab builds against.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def hhmm_to_minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def minutes_to_hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


@dataclass(frozen=True)
class Team:
    id: str
    seed: int
    rating: float = 1500.0  # used only by the fairness simulator
    coach: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "seed": self.seed, "rating": self.rating}
        if self.coach is not None:
            d["coach"] = self.coach
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Team":
        return Team(id=str(d["id"]), seed=int(d["seed"]),
                    rating=float(d.get("rating", 1500.0)), coach=d.get("coach"))


@dataclass(frozen=True)
class Instance:
    instance_id: str
    teams: tuple[Team, ...]
    pool_size: int
    courts: int
    start: str = "08:00"
    end: str = "16:00"
    match_minutes: int = 60
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "teams", tuple(self.teams))

    @property
    def n_slots(self) -> int:
        """T = floor((end - start) / match_minutes)."""
        return (hhmm_to_minutes(self.end) - hhmm_to_minutes(self.start)) // self.match_minutes

    def slot_time(self, slot: int) -> str:
        return minutes_to_hhmm(hhmm_to_minutes(self.start) + slot * self.match_minutes)

    def team(self, team_id: str) -> Team:
        for t in self.teams:
            if t.id == team_id:
                return t
        raise KeyError(team_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "teams": [t.to_dict() for t in self.teams],
            "pool_size": self.pool_size,
            "courts": self.courts,
            "start": self.start,
            "end": self.end,
            "match_minutes": self.match_minutes,
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Instance":
        return Instance(
            instance_id=str(d.get("instance_id", "instance")),
            teams=tuple(Team.from_dict(t) for t in d["teams"]),
            pool_size=int(d["pool_size"]),
            courts=int(d["courts"]),
            start=str(d.get("start", "08:00")),
            end=str(d.get("end", "16:00")),
            match_minutes=int(d.get("match_minutes", 60)),
            seed=int(d.get("seed", 0)),
        )


@dataclass(frozen=True)
class Match:
    match_id: str
    pool: str
    team1: str
    team2: str


@dataclass(frozen=True)
class Assignment:
    match_id: str
    pool: str
    team1: str
    team2: str
    court: int  # 1-based
    slot: int   # 0-based
    ref: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Assignment":
        return Assignment(match_id=str(d["match_id"]), pool=str(d["pool"]),
                          team1=str(d["team1"]), team2=str(d["team2"]),
                          court=int(d["court"]), slot=int(d["slot"]), ref=str(d["ref"]))


@dataclass
class Schedule:
    instance_id: str
    method: str
    assignments: list[Assignment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "method": self.method,
            "assignments": [a.to_dict() for a in sorted(self.assignments, key=lambda a: (a.slot, a.court))],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Schedule":
        return Schedule(instance_id=str(d["instance_id"]), method=str(d.get("method", "unknown")),
                        assignments=[Assignment.from_dict(a) for a in d["assignments"]])


@dataclass
class SearchResult:
    schedule: Schedule
    cost: dict[str, float]      # each component + "total"
    feasible: bool
    nodes_expanded: int         # 0 for baselines / annealing
    max_frontier: int
    runtime_s: float
    hit_limit: bool             # node or time limit reached
    optimal_proven: bool        # True only for UCS/A* that finished


# ---------------------------------------------------------------- JSON io
def _read(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _write(obj: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def load_instance(path: str | Path) -> Instance:
    return Instance.from_dict(_read(path))


def save_instance(inst: Instance, path: str | Path) -> None:
    _write(inst.to_dict(), path)


def load_schedule(path: str | Path) -> Schedule:
    return Schedule.from_dict(_read(path))


def save_schedule(sched: Schedule, path: str | Path) -> None:
    _write(sched.to_dict(), path)
