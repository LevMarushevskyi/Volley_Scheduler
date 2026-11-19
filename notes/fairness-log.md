# Fairness Simulator: Working Notes

## Purpose
Measures how often the genuinely best team wins a tournament under a given
format. Real tournaments can't answer this, since each happens once and true
team strength is unknown. In the sim, strength is set in advance, so the best
team is always known, and the same tournament can be replayed thousands of times.

## How one tournament works
- **Teams:** 16 teams, ratings drawn from a bell curve (mean 1500, spread 150).
  Fixed seed (42) gives the same field every run.
- **Sets:** win chance from the Elo formula (equal = 50%, +200 = 76%,
  +400 = 91%), decided by a weighted coin flip.
- **Matches:** best of 3. A 1700 team beat a 1500 team in ~74% of sets but ~85%
  of matches, so more games means less luck.
- **Pools:** snake-seeded into 4 pools of 4; full round robin in each.
- **Gold bracket:** top 2 per pool, 8-team single elimination.

## First full tournament (seed 7)

**Pool play:** Best team T12 (1675) went 3-0 in Pool A without dropping a set
(6-0) and entered Gold as the 1 seed.

**Gold bracket path:**
- Quarterfinal: T12 beat T15 (1389) 2-0
- Semifinal: T12 beat T07 (1550) 2-1
- Final: T12 lost to T04 (1605) 0-2

**Why this counts as the wrong outcome:** the sim defines "fair" as the
highest-rated team winning, since rating is the true strength the sim assigned.
T12 was rated 70 points above T04. By the Elo formula, that gives T12 ~60% per
set and ~65% per best-of-3 match, so T12 was the favorite, but T04 still had
roughly a 1-in-3 shot. The champion (T04) was the 2nd-best team, not the best.
An upset, but not an unlikely one.

**Other notes:**
- Weak team T15 (1389) finished 2nd in Pool C and made Gold, beating out
  T10 (1517) and T02 (1474) on set ratio.
- Pool B came down to a coin flip after a tie on both wins and sets.

**Takeaway:** one tournament can crown the wrong team even when the best team
dominates pools. Many runs are needed to tell whether that's rare or common.

## Design decisions (state in paper)
- Bracket seeded by pool results (wins, then set ratio), not true ratings,
  since directors only see results.
- Pool winners seeded 1-4, runners-up 5-8; matchups 1v8, 4v5, 2v7, 3v6.
- Pool tiebreak: wins, then set ratio, then coin flip. Head-to-head not added yet.
- Strength is one number: no matchup or style effects.
- Ratings fixed during a tournament: no fatigue or form swings.

## Monte Carlo result: 1,000 tournaments (16 teams, 4 pools of 4, top 2 -> Gold)
- Best team T12 (1675) won Gold 39.8% of the time (± 3.0%, 95% margin).
- Pure luck would be 6.25% (1 in 16), so the format rewards skill ~6x over chance.
- But the best team still loses ~60% of tournaments; the seed-7 upset was the
  typical outcome, not a fluke.

  ## Sanity checks (1,000 runs each)
- Equal field (all 1500): "best" team won 5.4% (± 1.4%); expected 6.25%. Pass.
- Huge gap (spread 1000): best team won 98.9% (± 0.6%). Pass.
- Normal field (spread 150): 39.8% (± 3.0%), the main result.

## Next steps
- Best team's win rate over 1,000 runs, with margin of error.
- Sanity checks: equal ratings, each team ~1/16; huge spread, best team ~always.
- Compare formats: pool sizes 3/4/6, top-1 vs top-2 advancing, snake vs random pools.
- Repeat at several rating spreads.
- Add head-to-head tiebreak.