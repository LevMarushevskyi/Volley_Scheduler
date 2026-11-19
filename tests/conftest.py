import pytest

from vsched.cost import DEFAULT_WEIGHTS
from vsched.model import Assignment, Instance, Schedule, Team


@pytest.fixture
def W():
    return {k: float(v) for k, v in DEFAULT_WEIGHTS.items()}


def make(n, k, courts, slots, coaches=None, match_minutes=60):
    coaches = coaches or {}
    teams = tuple(Team(f"T{i:02d}", i, 1500.0, coaches.get(f"T{i:02d}")) for i in range(1, n + 1))
    end_min = 8 * 60 + slots * match_minutes
    return Instance("test", teams, k, courts, "08:00", f"{end_min // 60:02d}:{end_min % 60:02d}", match_minutes, 0)


def sched(pool, rows):
    """rows: (slot, court, t1, t2, ref) with team numbers."""
    out = []
    for s, c, a, b, r in rows:
        a, b = min(a, b), max(a, b)
        out.append(Assignment(f"{pool}-T{a:02d}-T{b:02d}", pool, f"T{a:02d}", f"T{b:02d}", c, s, f"T{r:02d}"))
    return Schedule("test", "hand", out)
