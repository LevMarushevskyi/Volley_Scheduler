import random
from itertools import combinations
from ratings import make_teams, play_match


def snake_pools(teams, n_pools):
    """Split teams into balanced pools using snake seeding."""
    ranked = sorted(teams, key=lambda t: t["rating"], reverse=True)  # best first
    pools = [[] for _ in range(n_pools)]   # e.g. 4 empty pools: [[], [], [], []]
    for i, team in enumerate(ranked):      # i = 0, 1, 2, ... (seed number - 1)
        row = i // n_pools                 # which "row" of the snake we're on
        col = i % n_pools                  # position within that row
        if row % 2 == 1:                   # odd rows go backwards
            col = n_pools - 1 - col
        pools[col].append(team)
    return pools

def play_pool(pool, rng):
    """Round robin: every team plays every other team once.
    Returns the pool sorted 1st to last."""
    record = {t["id"]: {"wins": 0, "sets_won": 0, "sets_lost": 0}
              for t in pool}               # a blank scoreboard for each team

    for a, b in combinations(pool, 2):     # every possible pair, once
        a_sets, b_sets = play_match(a["rating"], b["rating"], rng)
        record[a["id"]]["sets_won"] += a_sets
        record[a["id"]]["sets_lost"] += b_sets
        record[b["id"]]["sets_won"] += b_sets
        record[b["id"]]["sets_lost"] += a_sets
        winner = a if a_sets > b_sets else b
        record[winner["id"]]["wins"] += 1

    def rank_key(t):
        r = record[t["id"]]
        set_ratio = r["sets_won"] / (r["sets_won"] + r["sets_lost"])
        return (r["wins"], set_ratio, rng.random())  # wins, then sets, then coin flip

    return sorted(pool, key=rank_key, reverse=True), record

def seed_order(teams, records):
    """Sort teams by pool results: wins, then set ratio."""
    def key(t):
        r = records[t["id"]]
        return (r["wins"], r["sets_won"] / (r["sets_won"] + r["sets_lost"]))
    return sorted(teams, key=key, reverse=True)


def play_bracket(seeded, rng, verbose=False):
    """Single elimination for 8 seeded teams. Returns the champion."""
    order = [0, 7, 3, 4, 1, 6, 2, 5]       # 1v8, 4v5, 2v7, 3v6 (counting from 0)
    alive = [seeded[i] for i in order]     # line teams up in bracket order

    while len(alive) > 1:                  # until one team is left
        winners = []
        for i in range(0, len(alive), 2):  # take teams two at a time
            a, b = alive[i], alive[i + 1]
            a_sets, b_sets = play_match(a["rating"], b["rating"], rng)
            w = a if a_sets > b_sets else b
            winners.append(w)
            if verbose:
                print(f"  {a['id']} vs {b['id']}  {a_sets}-{b_sets}  -> {w['id']}")
        if verbose:
            print("  ---")
        alive = winners                    # winners move to the next round

    return alive[0]                        # the champion

def run_tournament(teams, n_pools, rng):
    """One full tournament: pools -> Gold bracket -> champion."""
    pools = snake_pools(teams, n_pools)

    pool_winners, runners_up, records = [], [], {}
    for pool in pools:
        standings, record = play_pool(pool, rng)
        records.update(record)              # merge this pool's scoreboard in
        pool_winners.append(standings[0])   # 1st place
        runners_up.append(standings[1])     # 2nd place

    seeded = seed_order(pool_winners, records) + seed_order(runners_up, records)
    return play_bracket(seeded, rng)        # no printing, just the champion

def best_team_win_rate(teams, n_pools, n_runs=1000, seed=0):
    """Run many tournaments. How often does the highest-rated team win?"""
    rng = random.Random(seed)
    best = max(teams, key=lambda t: t["rating"])   # the true best team

    wins = 0
    for _ in range(n_runs):
        champ = run_tournament(teams, n_pools, rng)
        if champ["id"] == best["id"]:
            wins += 1

    p = wins / n_runs                               # win rate, e.g. 0.42
    margin = 1.96 * (p * (1 - p) / n_runs) ** 0.5   # 95% margin of error
    return best, p, margin

##TEST 
if __name__ == "__main__":
    print("Sanity checks + main result (16 teams, 4 pools of 4, 1000 runs each)\n")

    for spread, label in [(0, "Equal field (all 1500)"),
                          (150, "Normal field (main result)"),
                          (1000, "Huge gap")]:
        teams = make_teams(16, spread=spread)
        best, p, margin = best_team_win_rate(teams, n_pools=4, n_runs=1000)
        print(f"{label:28} best team wins {p:.1%} (± {margin:.1%})")

    print(f"\nPure luck would be {1/16:.1%}")

