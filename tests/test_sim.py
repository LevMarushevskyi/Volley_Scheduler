import numpy as np

from vsched.generator import generate
from vsched.search.bestfirst import solve_greedy
from vsched.sim.durations import replay, simulate_delays
from vsched.sim.fairness import kendall_tau_b, simulate, simulate_spread
from vsched.sim.ratings import p_match_best_of_3, p_set
from vsched.sim.tournament import Format, best_team_win_rate, bracket_order, play_pool
from vsched.sim.ratings import make_teams


def test_set_probability():
    assert p_set(1500, 1500) == 0.5
    assert abs(p_set(1900, 1500) - 1 / 1.1) < 1e-12
    assert p_match_best_of_3(1600, 1500) > p_set(1600, 1500)


def test_bracket_order():
    assert bracket_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]


def test_equal_ratings_uniform_champion():
    fmt = Format(24, 6, [("gold", [1, 2])])
    r = simulate(fmt, [1500.0] * 24, 12000, seed=3)
    share = np.array(r["champion_counts"]) / 12000
    assert abs(share.mean() - 1 / 24) < 1e-9
    assert np.all(np.abs(share - 1 / 24) < 0.015)
    assert abs(r["kendall_tau_mean"]) < 0.02


def test_huge_spread_best_nearly_always_wins():
    fmt = Format(24, 4, [("gold", [1, 2]), ("silver", [3, 4])])
    ratings = [3000 - 1000 * i for i in range(24)]
    r = simulate(fmt, ratings, 2000, seed=1)
    assert r["p_best_wins_gold"] > 0.99 and r["ci_low"] <= r["p_best_wins_gold"] <= r["ci_high"]
    assert r["kendall_tau_mean"] > 0.85


def test_matches_per_team_and_determinism():
    fmt = Format(24, 4, [("gold", [1, 2]), ("silver", [3, 4])])
    a = simulate_spread(fmt, 150, 500, seed=9)
    b = simulate_spread(fmt, 150, 500, seed=9)
    assert a == b
    assert a["matches_per_team_mean"] > 3  # 3 pool matches + at least one bracket match for most


def test_kendall():
    x = np.arange(5, dtype=float)
    assert kendall_tau_b(x, x) == 1.0
    assert kendall_tau_b(x, x[::-1].copy()) == -1.0


def test_delay_replay(W):
    inst = generate("medium", 2)
    s = solve_greedy(inst, W).schedule
    on_time = replay(inst, s, np.full(len(s.assignments), float(inst.match_minutes)))
    assert on_time["overrun_min"] == 0 and on_time["max_delay_min"] == 0
    d = simulate_delays(inst, s, n_reps=50, seed=1, cv=0.3)
    assert d["day_end_min"].shape == (50,) and d["max_delay_min"].min() >= 0


def test_pr1_api_still_works():
    """The fairness-sim owner's original entry points keep their behaviour."""
    teams = make_teams(16, spread=1000)
    best, p, margin = best_team_win_rate(teams, n_pools=4, n_runs=300)
    assert best["rating"] == max(t["rating"] for t in teams) and p > 0.9 and margin > 0


def test_head_to_head_breaks_two_way_tie(monkeypatch):
    import random
    import vsched.sim.tournament as tour
    # A and B finish 2-1 with identical set ratio 5/9; B beat A, so B is 1st.
    results = {("A", "B"): (1, 2), ("A", "C"): (2, 1), ("A", "D"): (2, 1),
               ("B", "C"): (1, 2), ("B", "D"): (2, 1), ("C", "D"): (0, 2)}
    ratings = {1500 + i: n for i, n in enumerate("ABCD")}
    monkeypatch.setattr(tour, "play_match", lambda ra, rb, rng, **kw: results[(ratings[ra], ratings[rb])])
    pool = [{"id": n, "rating": 1500 + i} for i, n in enumerate("ABCD")]
    standings, rec = tour.play_pool(pool, random.Random(0))
    assert [t["id"] for t in standings] == ["B", "A", "D", "C"]
