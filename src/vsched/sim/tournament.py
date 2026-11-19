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

##TEST 
if __name__ == "__main__":
    rng = random.Random(7)
    teams = make_teams(16)
    pools = snake_pools(teams, 4)

    for letter, pool in zip("ABCD", pools):
        standings, record = play_pool(pool, rng)
        print(f"\nPool {letter}:")
        for place, t in enumerate(standings, start=1):
            r = record[t["id"]]
            print(f"  {place}. {t['id']} ({t['rating']})  "
                  f"wins {r['wins']}  sets {r['sets_won']}-{r['sets_lost']}")

