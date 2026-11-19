# Real divisions (anonymized)

E4 (`python -m experiments.e4_real`) reads every pair
`<name>.instance.json` + `<name>.schedule.json` in this folder and compares
the VolleyLens schedule with our methods under the same cost function.

Only anonymized data belongs here: teams are `T01, T02, …` (by seed), clubs
and coaches are `C01, C02, …`, and there are no team names, cities or venues.
Produce the files with:

    python -m vsched.anonymize export.json data/real/div01

where `export.json` is `{"instance": {...}, "schedule": {...}}` in the repo's
JSON formats with the real names still in it (keep that file out of the repo;
`data/raw/` is git-ignored for this).

If this folder holds no divisions, E4 prints a note and skips; every other
result runs from generated instances.
