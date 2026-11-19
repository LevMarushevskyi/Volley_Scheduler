# Convenient and Fair: Volleyball Tournament Scheduler

COMP 560 final project. Big question:

> Can a volleyball tournament be both convenient for teams and fair to the
> strongest team, and what does each format trade to get there?

- **Convenience (search).** Build single-day pool-play schedules on a court ×
  time grid with uniform-cost search, A\*, weighted A\*, greedy best-first and
  simulated annealing, score them with an explicit cost function, and compare
  against a random baseline and the standard pool-sheet template.
- **Fairness (simulation).** Simulate each tournament format thousands of times
  with Elo-style set probabilities and measure how often the truly best team
  wins Gold.

Research code, MIT license. Real VolleyLens data is included only if cleared
and anonymized (see `data/real/README.md`); every other result runs from
generated instances with fixed seeds.

## Setup (Python 3.11)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # pinned deps + `pip install -e .`
pytest -q                           # ~170 tests, about 15 s
```

## Slides

`slides/convenient-and-fair.pptx` (and `.pdf`): the 5-minute presentation, with speaker notes. Its numbers come from `results/`.

## Run

| What | Command | Output |
| --- | --- | --- |
| App | `streamlit run app/streamlit_app.py` | 4 tabs: build a schedule, compare methods, fairness, delays |
| All experiments | `python -m experiments.run_all` | `results/*.csv`, `figures/*.png` (about 25 min on a laptop) |
| One experiment | `python -m experiments.e6_tradeoff` | its CSV + figure |
| Fairness demo (PR #1) | `python src/vsched/sim/tournament.py` | sanity checks + 16-team result |

`run_all` is deterministic: experiments stop on node and move limits, not wall
clock, so two runs give identical CSVs. Wall-clock runtimes go to
`results/*_timing.csv`, which is not expected to match between runs.

| # | Question | Files |
| --- | --- | --- |
| E1 | How close does each method get to the true optimum on tiny instances? | `e1_gap_table.csv`, `e1_nodes_vs_size.png` |
| E2 | Which method wins on medium and large instances, and how fast? | `e2_summary.csv`, `e2_cost_by_method.png` |
| E3 | Does the ranking hold when the weights change? | `e3_rank_table.csv` |
| E4 | Our methods vs VolleyLens on real divisions | `e4_real.csv` (skipped if `data/real/` is empty) |
| E5 | Which formats are fairest? | `e5_fairness.csv`, `e5_fairness.png` |
| E6 | What does each format trade between convenience and fairness? | `e6_tradeoff.csv`, `e6_tradeoff.png` (headline) |
| E7 | Which schedules still finish on time when matches run long? | `e7_delays.csv`, `e7_delays.png` |

All seeds and sizes are in `configs/experiments.yaml`; cost weights are in
`configs/weights.yaml`.

## The model

- **Teams** have an id, seed, rating (used only by the fairness simulator) and
  an optional coach.
- **Pools**: snake seeding into pools of size k (3–6). For 12 teams, k = 4:
  A = 1, 6, 7, 12; B = 2, 5, 8, 11; C = 3, 4, 9, 10. Each pool plays a full
  round robin.
- **Grid**: C courts × T slots, T = floor((end − start) / match length). Each
  pool stays on one court; pools share a court only when there are more pools
  than courts.
- **Hard constraints** (`check_feasible`): every match exactly once; one match
  per court per slot; a team does one thing per slot; the ref is from the same
  pool and not playing; every slot is inside the day.
- **Soft cost** (`cost`): idle slots ×1, third straight match ×5, second
  straight ref ×2, per-pool ref imbalance ×3, same-coach teams busy at once
  ×10, makespan ×2 per slot.

### Search

One best-first loop fills the grid cell by cell in time order. The state is
the next cell, the placed matches and a per-team summary. Actions place a
match with a ref, or leave the cell empty. UCS sorts by g, A\* by g + h,
weighted A\* by g + w·h, and greedy by h.

- g charges each penalty as soon as it is certain, and that includes makespan:
  one slot is charged each time the search opens a slot with matches left.
  Every schedule's total cost stays the same. Without this, UCS sees g ≈ 0 on
  every partial schedule and enumerates the whole space.
- The closed set merges states that differ only by relabeling coach-free
  teams in the same pool. This is exact, because the cost never looks at
  their names.
- h has two versions. The basic one is a makespan lower bound from each
  court's remaining matches. The full one adds idle slots that are already
  certain and a lower bound on ref imbalance. Both are consistent, and the
  tests check that A\* equals UCS on 20 tiny instances and that UCS equals
  brute force.
- Greedy uses the basic h. With the full h, its idle term rises whenever a
  match is placed, and greedy stalls on plateaus.

Simulated annealing is not in the course slides. It does local search over
complete schedules: swap two cells, move a match to an empty cell, change a
ref, or swap two pools' courts. Illegal moves are rejected. T is multiplied by
0.995 every 100 moves, and the run restarts with several seeds.

### Fairness

The fairness simulator was started in PR #1 (`sim/ratings.py`,
`sim/tournament.py`) and generalized on top of it.

- **Set probability**: P(A wins a set) = 1 / (1 + 10^(−(rA − rB)/400)).
- **Pool matches**: best of 3, or 2 sets.
- **Standings**: match wins, then set ratio, then head-to-head for a two-way
  tie, then a coin flip.
- **Brackets**: cut by pool finishing place (Gold = 1st and 2nd of each pool
  by default). Single elimination, pool winners seeded first, byes to the top
  seeds.
- **Metrics**: P(best team wins Gold) with a 95% CI, Kendall τ between true
  strength and final placement, and matches per team.

### Templates

`data/templates.json` holds the pool sheets keyed by pool size: `slot`,
`teams` and `ref`, all by seed position within the pool. The 4-team sheet is
the real one events use. The 3-, 5- and 6-team sheets were derived from its
structure (circle-method rounds; the ref is the free team with the fewest refs
so far, never twice in a row) and are marked as such in the file.

## Layout

```
configs/      weights.yaml, experiments.yaml
data/         templates.json, real/ (anonymized divisions, if cleared)
src/vsched/   model, roundrobin, cost, generator, baselines, methods, present, anonymize
  search/     bestfirst (UCS/A*/wA*/greedy), heuristics, annealing, problem
  sim/        ratings, tournament, fairness, durations
app/          streamlit_app.py
experiments/  e1–e7, run_all
notes/        fairness-log.md
results/ figures/ tests/
```

## Deploy

On Streamlit Community Cloud, create an app from this repo with main file
`app/streamlit_app.py` and Python 3.11. `requirements.txt` installs the
package.

## Credits

COMP 560 team project. Real-data baseline from VolleyLens (anonymized), used
with permission.

## Step-by-step: run it locally

1. **Get the code**
   ```bash
   git clone https://github.com/LevMarushevskyi/Volley_Scheduler.git
   cd Volley_Scheduler
   ```
2. **Create a Python 3.11 environment** (check with `python3.11 --version`)
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. **Install everything**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the tests** (about 15 s; expect `167 passed`)
   ```bash
   pytest -q
   ```
5. **Open the app**
   ```bash
   streamlit run app/streamlit_app.py
   ```
6. **Reproduce all results** (writes `results/` and `figures/`)
   ```bash
   python -m experiments.run_all
   ```
7. **Optional: run one experiment or the fairness demo**
   ```bash
   python -m experiments.e6_tradeoff
   python src/vsched/sim/tournament.py
   ```

In every new terminal, run `source .venv/bin/activate` first.
