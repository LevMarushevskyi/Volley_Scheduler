import pytest

from vsched.generator import generate
from vsched.model import Instance, Schedule, load_instance, load_schedule, save_instance, save_schedule
from vsched.roundrobin import default_pool_courts, round_robin, snake_pools
from vsched.search.bestfirst import solve_greedy

from conftest import make


def test_snake_12_teams_k4():
    inst = make(12, 4, 3, 8)
    pools = snake_pools(inst.teams, 4)
    seeds = {p: [t.seed for t in m] for p, m in pools.items()}
    assert seeds == {"A": [1, 6, 7, 12], "B": [2, 5, 8, 11], "C": [3, 4, 9, 10]}


@pytest.mark.parametrize("k,expected", [(3, 3), (4, 6), (5, 10), (6, 15)])
def test_round_robin_counts(k, expected):
    inst = make(k, k, 1, 20)
    ms = round_robin("A", list(inst.teams))
    assert len(ms) == expected
    assert len({frozenset((m.team1, m.team2)) for m in ms}) == expected


def test_uneven_pools_and_too_small():
    inst = make(10, 4, 3, 8)
    assert sorted(len(m) for m in snake_pools(inst.teams, 4).values()) == [3, 3, 4]
    with pytest.raises(ValueError):
        snake_pools(make(5, 4, 1, 8).teams, 4)


def test_n_slots_and_court_rule():
    inst = make(16, 4, 3, 7, match_minutes=45)
    assert inst.n_slots == 7
    assert default_pool_courts(inst) == {"A": 1, "B": 2, "C": 3, "D": 1}


def test_json_roundtrip(W, tmp_path):
    inst = generate("medium", 4)
    save_instance(inst, tmp_path / "i.json")
    assert load_instance(tmp_path / "i.json") == inst
    s = solve_greedy(inst, W).schedule
    save_schedule(s, tmp_path / "s.json")
    s2 = load_schedule(tmp_path / "s.json")
    assert s2.to_dict() == s.to_dict()
    assert Instance.from_dict(inst.to_dict()) == inst
    assert Schedule.from_dict(s.to_dict()).to_dict() == s.to_dict()
