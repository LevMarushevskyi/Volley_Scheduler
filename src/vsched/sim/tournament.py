"""One simulated tournament: snake-seeded pools, round robin, then brackets.

Built on the fairness-sim owner's PR #1 (snake_pools, play_pool, seed_order,
play_bracket, run_tournament, best_team_win_rate keep their behaviour and
signatures). Generalised for the experiments:
  * Format: any pool size, any brackets cut by pool PLACE (as VolleyLens
    does: Gold = 1st and 2nd of every pool), snake or random pools,
    best of 3 or two sets in pools.
  * play_pool: adds the head-to-head tiebreak (the notes' next step).
    Standings: match wins (a 1-1 split in two_sets = 0.5 each), then set
    ratio, then head-to-head when exactly two teams are tied, then a coin flip.
  * play_bracket: any size; byes to the top seeds when the size is not a
    power of 2; 8 teams still pair 1v8, 4v5, 2v7, 3v6. Bracket matches are
    always best of 3. Bracket seeding: pool winners first, then runners-up,
    ...; within a place by pool results (wins, then set ratio).
  * run_once: one tournament for a Format, with a final placement per team
    (for Kendall tau) and matches played per team.
"""
import math
import random
from dataclasses import dataclass, field
from itertools import combinations

try:  # package import (vsched.sim.tournament)
    from .ratings import make_teams, play_match
except ImportError:  # `python src/vsched/sim/tournament.py`, as in PR #1
    from ratings import make_teams, play_match


@dataclass
class Format:
    n_teams: int
    pool_size: int
    brackets: list = field(default_factory=lambda: [("gold", [1, 2])])  # [(name, pool places)]
    match_format: str = "best_of_3"         # or "two_sets"
    seeding: str = "snake"                  # or "random"

    @property
    def n_pools(self):
        return math.ceil(self.n_teams / self.pool_size)

    @property
    def label(self):
        cut = "+".join(f"{name}:{''.join(map(str, places))}" for name, places in self.brackets)
        return f"pools of {self.pool_size} | {cut} | {self.seeding} | {self.match_format}"


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


def random_pools(teams, n_pools, rng):
    """Same pool sizes as snake seeding, but teams dealt in random order."""
    shuffled = list(teams)
    rng.shuffle(shuffled)
    pools = [[] for _ in range(n_pools)]
    for i, team in enumerate(shuffled):
        row, col = divmod(i, n_pools)
        pools[col if row % 2 == 0 else n_pools - 1 - col].append(team)
    return pools


def play_pool(pool, rng, match_format="best_of_3", played=None):
    """Round robin: every team plays every other team once.
    Returns the pool sorted 1st to last, and each team's record."""
    record = {t["id"]: {"wins": 0, "sets_won": 0, "sets_lost": 0}
              for t in pool}               # a blank scoreboard for each team
    beat = {}                              # head-to-head: (a, b) -> winner id

    for a, b in combinations(pool, 2):     # every possible pair, once
        a_sets, b_sets = play_match(a["rating"], b["rating"], rng, match_format=match_format)
        record[a["id"]]["sets_won"] += a_sets
        record[a["id"]]["sets_lost"] += b_sets
        record[b["id"]]["sets_won"] += b_sets
        record[b["id"]]["sets_lost"] += a_sets
        if played is not None:
            played[a["id"]] = played.get(a["id"], 0) + 1
            played[b["id"]] = played.get(b["id"], 0) + 1
        if a_sets == b_sets:               # 1-1 split (two_sets format)
            record[a["id"]]["wins"] += 0.5
            record[b["id"]]["wins"] += 0.5
            continue
        winner = a if a_sets > b_sets else b
        record[winner["id"]]["wins"] += 1
        beat[frozenset((a["id"], b["id"]))] = winner["id"]

    def main_key(t):
        r = record[t["id"]]
        played_sets = r["sets_won"] + r["sets_lost"]
        return (r["wins"], r["sets_won"] / played_sets if played_sets else 0.0)

    coin = {t["id"]: rng.random() for t in pool}
    standings = sorted(pool, key=lambda t: (main_key(t), coin[t["id"]]), reverse=True)

    # head-to-head, only when exactly two teams are tied on wins and set ratio
    i = 0
    while i < len(standings) - 1:
        group = [standings[i]]
        while i + len(group) < len(standings) and main_key(standings[i + len(group)]) == main_key(standings[i]):
            group.append(standings[i + len(group)])
        if len(group) == 2:
            a, b = group
            if beat.get(frozenset((a["id"], b["id"]))) == b["id"]:
                standings[i], standings[i + 1] = b, a
        i += len(group)
    return standings, record


def seed_order(teams, records):
    """Sort teams by pool results: wins, then set ratio."""
    def key(t):
        r = records[t["id"]]
        return (r["wins"], r["sets_won"] / (r["sets_won"] + r["sets_lost"]))
    return sorted(teams, key=key, reverse=True)


def bracket_order(size):
    """Seed numbers (1-based) in bracket line order: 8 -> 1,8,4,5,2,7,3,6."""
    order = [1]
    while len(order) < size:
        n = len(order) * 2
        order = [x for s in order for x in (s, n + 1 - s)]
    return order


def play_bracket(seeded, rng, verbose=False, out_round=None, played=None):
    """Single elimination for any number of seeded teams. Returns the champion.

    out_round (optional dict) gets, per team id, the round it went out in:
    0 = champion, 1 = lost the final, 2 = lost a semifinal, ...
    """
    n = len(seeded)
    if n == 0:
        return None
    size = 1 << (n - 1).bit_length()       # next power of 2: byes fill the rest
    alive = [seeded[s - 1] if s <= n else None for s in bracket_order(size)]
    rnd = int(math.log2(size))

    while len(alive) > 1:                  # until one team is left
        winners = []
        for i in range(0, len(alive), 2):  # take teams two at a time
            a, b = alive[i], alive[i + 1]
            if a is None or b is None:     # a bye: the other team walks through
                winners.append(a if b is None else b)
                continue
            a_sets, b_sets = play_match(a["rating"], b["rating"], rng)
            w, lose = (a, b) if a_sets > b_sets else (b, a)
            winners.append(w)
            if out_round is not None:
                out_round[lose["id"]] = rnd
            if played is not None:
                played[a["id"]] = played.get(a["id"], 0) + 1
                played[b["id"]] = played.get(b["id"], 0) + 1
            if verbose:
                print(f"  {a['id']} vs {b['id']}  {a_sets}-{b_sets}  -> {w['id']}")
        if verbose:
            print("  ---")
        alive = winners                    # winners move to the next round
        rnd -= 1

    if out_round is not None:
        out_round[alive[0]["id"]] = 0
    return alive[0]                        # the champion


def run_once(fmt, teams, rng):
    """One tournament for a Format. teams: [{"id", "rating"}].
    Returns the first bracket's champion, a placement per team id (lower =
    better; bracket teams by bracket order then round reached, others after,
    by pool place) and matches played per team id."""
    if fmt.seeding == "random":
        pools = random_pools(teams, fmt.n_pools, rng)
    else:
        pools = snake_pools(teams, fmt.n_pools)
    played, records, by_place = {}, {}, {}
    for pool in pools:
        standings, record = play_pool(pool, rng, fmt.match_format, played)
        records.update(record)
        for place, t in enumerate(standings, start=1):
            by_place.setdefault(place, []).append(t)

    placement, base, champion = {}, 0, None
    for bi, (_name, places) in enumerate(fmt.brackets):
        seeded = []
        for place in places:               # pool winners first, then runners-up, ...
            seeded += seed_order(by_place.get(place, []), records)
        out = {}
        champ = play_bracket(seeded, rng, out_round=out, played=played)
        if bi == 0:
            champion = champ
        for t in seeded:
            r = out[t["id"]]
            placement[t["id"]] = base + (0 if r == 0 else 2 ** (r - 1))  # 0, 1, 2, 4, ...
        base += len(seeded) + 1
    for place in sorted(by_place):
        for t in by_place[place]:
            placement.setdefault(t["id"], base + place)
    return {"champion": champion, "placement": placement, "played": played}


def run_tournament(teams, n_pools, rng):
    """One full tournament: pools -> Gold bracket (top 2 per pool) -> champion."""
    fmt = Format(len(teams), math.ceil(len(teams) / n_pools), [("gold", [1, 2])])
    return run_once(fmt, teams, rng)["champion"]


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
