import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from vsched.anonymize import anonymize
from vsched.cost import check_feasible
from vsched.present import parse_team_list


def test_app_runs_without_exceptions():
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    assert any(m.label == "Total cost" for m in at.metric)


def test_parse_team_list():
    ts = parse_team_list("Eagles, 1, K1\nSharks\n\nBears, 3")
    assert [(t.id, t.seed, t.coach) for t in ts] == [("Eagles", 1, "K1"), ("Sharks", 2, None), ("Bears", 3, None)]


def test_anonymize_strips_names():
    inst = {"teams": [{"id": "Real Club 14s", "seed": 2, "coach": "Coach Smith", "city": "X"},
                      {"id": "Other 14s", "seed": 1, "coach": "Coach Smith"},
                      {"id": "Third", "seed": 3}],
            "pool_size": 3, "courts": 1, "start": "08:00", "end": "12:00", "match_minutes": 60, "venue": "Gym"}
    sched = {"assignments": [
        {"pool": "A", "team1": "Other 14s", "team2": "Third", "court": 1, "slot": 0, "ref": "Real Club 14s"},
        {"pool": "A", "team1": "Real Club 14s", "team2": "Third", "court": 1, "slot": 1, "ref": "Other 14s"},
        {"pool": "A", "team1": "Other 14s", "team2": "Real Club 14s", "court": 1, "slot": 2, "ref": "Third"}]}
    i, s = anonymize(inst, sched, "div01")
    blob = json.dumps(i.to_dict()) + json.dumps(s.to_dict())
    for word in ("Real", "Other", "Third", "Smith", "Gym", "city"):
        assert word not in blob
    assert [t.id for t in i.teams] == ["T01", "T02", "T03"] and i.teams[0].coach == "C01"
    assert check_feasible(i, s) == []
