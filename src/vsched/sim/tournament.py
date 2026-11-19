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

##TEST 
if __name__ == "__main__":
    rng = random.Random(7)
    teams = make_teams(16)
    pools = snake_pools(teams, 4)

    pool_winners, runners_up, records = [], [], {}
    for pool in pools:
        standings, record = play_pool(pool, rng)
        records.update(record)             # merge this pool's scoreboard in
        pool_winners.append(standings[0])  # 1st place
        runners_up.append(standings[1])    # 2nd place

    seeded = seed_order(pool_winners, records) + seed_order(runners_up, records)
    print("Gold seeds:", [t["id"] for t in seeded])

    champ = play_bracket(seeded, rng, verbose=True)
    print(f"Champion: {champ['id']} ({champ['rating']})")

