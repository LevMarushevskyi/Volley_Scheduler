"""Teams, Elo set win %, and match simulation (fairness sim building blocks).

Original demo by the fairness-sim owner (PR #1); extended so the general
simulator can use the same functions: a configurable Elo scale and a
'two_sets' match format alongside best of 3.
"""
import random  # Python's built-in dice roller

ELO_SCALE = 400


def make_teams(n_teams, mean=1500, spread=150, seed=42):
    """Create n_teams fake teams, each with a skill rating."""
    rng = random.Random(seed)              # our own dice, locked to a seed
    teams = []                             # empty list to fill up
    for i in range(n_teams):               # repeat once per team
        rating = rng.gauss(mean, spread)   # random number from a bell curve
        teams.append({
            "id": f"T{i+1:02d}",           # name: T01, T02, ... T16
            "rating": round(rating),       # whole number, like 1534
        })
    return teams


def p_win_set(rating_a, rating_b, scale=ELO_SCALE):
    """Chance (0 to 1) that team A beats team B in one set."""
    gap = rating_a - rating_b              # how much better A is
    return 1 / (1 + 10 ** (-gap / scale))  # Elo formula


p_set = p_win_set  # short alias used by the general simulator


def play_set(rating_a, rating_b, rng, scale=ELO_SCALE):
    """Play one set. True = A wins, False = B wins."""
    roll = rng.random()                    # random number between 0 and 1
    return roll < p_win_set(rating_a, rating_b, scale)


def play_match(rating_a, rating_b, rng, best_of=3, match_format="best_of_3", scale=ELO_SCALE):
    """Play a match. Returns (sets A won, sets B won).

    best_of_3 (default): keep playing until someone has 2 sets.
    two_sets: play exactly 2 sets, so a 1-1 split is possible.
    """
    a_sets = 0                             # scoreboard starts 0-0
    b_sets = 0
    if match_format == "two_sets":
        for _ in range(2):
            if play_set(rating_a, rating_b, rng, scale):
                a_sets += 1
            else:
                b_sets += 1
        return a_sets, b_sets
    need = best_of // 2 + 1                # sets needed to win: 2
    while a_sets < need and b_sets < need: # keep going until someone hits 2
        if play_set(rating_a, rating_b, rng, scale):
            a_sets += 1                    # A took this set
        else:
            b_sets += 1                    # B took this set
    return a_sets, b_sets                  # e.g. (2, 1)


def p_match_best_of_3(rating_a, rating_b, scale=ELO_SCALE):
    """Exact chance A wins a best-of-3: win 2-0, or 2-1 (two orders)."""
    p = p_win_set(rating_a, rating_b, scale)
    return p * p + 2 * p * p * (1 - p)


##TEST

if __name__ == "__main__":
    for gap in [0, 100, 200, 400]:
        print(f"A is +{gap} ELO: {p_win_set(1500 + gap, 1500):.2f} chance to win a set")

    rng = random.Random(1)

    print("\n5 sample matches, 1700 vs 1500:")
    for _ in range(5):
        print(" ", play_match(1700, 1500, rng))

    wins = 0
    for _ in range(1000):
        a, b = play_match(1700, 1500, rng)
        if a > b:
            wins += 1
    print(f"1700 won {wins} of 1000 matches")
