from vsched.cost import check_feasible, cost, team_components, PLAY, REF, IDLE

from conftest import make, sched

# One pool of 4 (T01..T04), one court. Scored by hand in the comments.
S1 = [(0, 1, 1, 2, 3), (1, 1, 3, 4, 1), (2, 1, 1, 3, 2), (3, 1, 2, 4, 3), (4, 1, 1, 4, 2), (5, 1, 2, 3, 4)]
# T01 P R P . P   idle1 | T02 P . R P R P idle1 refs2 | T03 R P P R . P idle1 refs2 | T04 . P . P P R idle1
S4 = [(0, 1, 1, 2, 4), (1, 1, 1, 3, 4), (2, 1, 1, 4, 2), (3, 1, 2, 3, 1), (4, 1, 2, 4, 3), (5, 1, 3, 4, 2)]
# T01 P P P R (three 1) | T02 P . R P P R idle1 | T03 . P . P R P idle1 | T04 R R P . P P idle1 double1


def test_hand_scored_pool_of_4(W):
    inst = make(4, 4, 1, 8)
    s = sched("A", S1)
    assert check_feasible(inst, s) == []
    c = cost(inst, s, W)
    assert (c["idle"], c["three_straight"], c["double_work"], c["work_imbalance"], c["coach_conflict"], c["makespan"]) == (4, 0, 0, 1, 0, 6)
    assert c["total"] == 4 + 3 + 12 == 19


def test_hand_scored_three_straight_double_work(W):
    inst = make(4, 4, 1, 8)
    s = sched("A", S4)
    assert check_feasible(inst, s) == []
    c = cost(inst, s, W)
    assert (c["idle"], c["three_straight"], c["double_work"], c["work_imbalance"], c["makespan"]) == (3, 1, 1, 1, 6)
    assert c["total"] == 3 + 5 + 2 + 3 + 12 == 25


def test_hand_scored_coach_conflict(W):
    # 6 teams, k=3: A = seeds 1,4,5 on court 1; B = 2,3,6 on court 2; T01 and T02 share coach K1.
    inst = make(6, 3, 2, 4, coaches={"T01": "K1", "T02": "K1"})
    s = sched("A", [(0, 1, 1, 4, 5), (1, 1, 1, 5, 4), (2, 1, 4, 5, 1)])
    s.assignments += sched("B", [(0, 2, 2, 3, 6), (1, 2, 3, 6, 2), (2, 2, 2, 6, 3)]).assignments
    assert check_feasible(inst, s) == []
    c = cost(inst, s, W)
    assert (c["idle"], c["three_straight"], c["double_work"], c["work_imbalance"], c["coach_conflict"], c["makespan"]) == (0, 0, 0, 0, 3, 3)
    assert c["total"] == 30 + 6


def test_team_components_runs():
    assert team_components([PLAY] * 4) == (0, 2, 0, 0)          # run of 4 plays counts 2
    assert team_components([REF, REF, REF, IDLE, PLAY]) == (1, 0, 2, 3)
    assert team_components([IDLE, PLAY, IDLE, IDLE, REF, IDLE]) == (2, 0, 0, 1)
    assert team_components([IDLE] * 3) == (0, 0, 0, 0)


# ------------------------------------------------- checker: one test per type
def test_violation_missing_and_duplicate():
    inst = make(4, 4, 1, 8)
    v = check_feasible(inst, sched("A", S1[:-1]))
    assert any("missing match" in x for x in v)
    v = check_feasible(inst, sched("A", S1 + [(6, 1, 2, 3, 1)]))
    assert any("scheduled 2 times" in x for x in v)


def test_violation_court_double_booked():
    inst = make(8, 4, 2, 8)  # A = 1,4,5,8  B = 2,3,6,7
    rows = [(0, 1, 1, 4, 5), (1, 1, 1, 5, 4), (2, 1, 1, 8, 4), (3, 1, 4, 5, 1), (4, 1, 4, 8, 5), (5, 1, 5, 8, 1)]
    s = sched("A", rows)
    s.assignments += sched("B", [(0, 1, 2, 3, 6), (1, 2, 2, 6, 3), (2, 2, 2, 7, 6), (3, 2, 3, 6, 2), (4, 2, 3, 7, 2), (5, 2, 6, 7, 3)]).assignments
    v = check_feasible(inst, s)
    assert any("court 1 slot 0 has 2 matches" in x for x in v)


def test_violation_team_two_duties():
    inst = make(4, 4, 2, 8)
    rows = [(0, 1, 1, 2, 3), (0, 2, 1, 4, 2)] + [r for r in S1 if r[2:4] not in ((1, 2), (1, 4))]
    v = check_feasible(inst, sched("A", rows))
    assert any("team T01 has 2 duties in slot 0" in x for x in v)


def test_violation_ref_wrong_pool_or_playing():
    inst = make(8, 4, 2, 8)  # A = 1,4,5,8  B = 2,3,6,7
    s = sched("A", [(0, 1, 1, 4, 2)])  # T02 is in pool B
    assert any("is not in pool A" in x for x in check_feasible(inst, s))
    inst4 = make(4, 4, 1, 8)
    s = sched("A", [(0, 1, 1, 2, 2)] + S1[1:])
    assert any("ref T02 is playing in slot 0" in x for x in check_feasible(inst4, s))


def test_violation_outside_day():
    inst = make(4, 4, 1, 6)
    rows = S1[:-1] + [(6, 1, 2, 3, 4)]
    v = check_feasible(inst, sched("A", rows))
    assert any("outside day" in x for x in v)
