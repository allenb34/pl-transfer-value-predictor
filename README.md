# Premier League Transfer Value Predictor

Predicts a Premier League player's transfer/market value from basic stats (goals, assists,
age, position) using a Linear Regression model. Built as a small end-to-end data science
project: data collection from two independent sources, cleaning/matching them against each
other, model comparison, and a CLI for querying real or hypothetical players.

This README documents what was actually built and why, including the dead ends and the
places the model is currently weak. It is not a sales pitch for the model's accuracy.

**Tech stack**: `requests` (API calls), `pandas` (data wrangling), `scikit-learn`
(Linear Regression / Random Forest), `matplotlib` (visualization). Also `python-dotenv`
(API key loading), `joblib` (model persistence), `truststore` (see Setup), `numpy`.

## Architecture

```
data_collection/
  fetch_player_stats.py     -> football-data.org API: squads, goals, assists, age, position
  fetch_transfer_values.py  -> dcaribou/transfermarkt-datasets CSV: transfer values
  sanity_check.py           -> diagnostics + the club-scoped name-matching logic (reused by merge)
  merge_dataset.py          -> matches the two sources, cleans, produces the modeling dataset
  data_quality_report.py    -> distribution/skew diagnostics on the merged dataset

modeling/
  train_models.py           -> Linear Regression vs Random Forest, train/test comparison
  train_final_model.py      -> retrains the chosen model (Linear Regression) on all data
  predict.py                -> CLI: look up a real player or enter a hypothetical one
  visualize.py              -> predicted-vs-actual scatter plot + accuracy summary

data/
  raw/                      -> raw API responses and per-source CSVs, plus collection_errors.log
  processed/                -> player_dataset.csv (final modeling data) + excluded_players.csv
```

Pipeline: **collect stats** (football-data.org) + **collect values** (Transfermarkt dataset,
separately) → **match & merge** the two sources by player → **clean** (age filter, outlier
removal, log transform) → **train/compare models** → **predict**.

## Key decisions and why

**football-data.org has no real market-value endpoint, so a second source was required.**
The docs describe a `marketValue` field on the squad endpoint, and it does exist in live
responses - but it's undocumented, refreshes on an unknown/slow cadence, and has no
historical tracking. Rather than rely on it, transfer values come from a separate, dedicated
source: [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)
(CC0-licensed, weekly-refreshed CSV mirror of Transfermarkt data), joined against
football-data.org's stats after the fact.

**Matching players between two independently-sourced datasets required club-scoping, not
just name matching.** football-data.org and the Transfermarkt dataset use different name
formats ("Man City" vs "Manchester City", "Alisson" vs "Alisson Becker") and, more
importantly, can each be current for a different club if a player transferred between the
two sources' snapshots. Matching on name alone produced two failure modes, both real and both
found during development:
- **Cross-club false positives on genuinely different people**: name-only fuzzy matching
  paired Crystal Palace's Ben Chilwell with Cardiff City's Rubin Colwill purely because the
  strings looked similar - two unrelated players.
- **A subtler, more dangerous failure**: even *exact* name matches ignoring club silently
  paired the right person to the wrong (stale) club and value - e.g. Emiliano Martínez was
  matched to a Transfermarkt row still tagged "Aston Villa" after he'd moved to Chelsea in
  the current football-data.org squad list.

  The fix: match on (normalized name, normalized club) first, fall back to same-club-only
  token-subset/fuzzy matching, and never match across clubs. This raised the match rate from
  34.3% to 56.8% of `player_stats.csv` rows and eliminated cross-club false matches by
  construction. Anything left unmatched is written to `excluded_players.csv` with a reason
  rather than guessed at.

**`minutes_played` was dropped from the feature set entirely.** Two paths were tried:
1. football-data.org's free tier - confirmed live, not just from docs, that
   `/v4/matches/{id}` doesn't return `lineup`, `bench`, `substitutions`, `bookings`, or
   `goals` at all on this plan. There's no way to derive minutes from it.
2. The Transfermarkt dataset has an `appearances.csv` table with real per-match minutes -
   but its most recent Premier League date is 2026-05-24 (last season), with zero rows for
   the current season. Using it would mean mixing *last season's* minutes with *this
   season's* everything else - a season-currency mismatch, not a fix.

   Given neither option produced current-season data, `minutes_played` was removed rather
   than filled with a proxy that doesn't actually describe the current season.

**The target (`transfer_value`) was heavily right-skewed (skew ≈ 2.01), so it was
log-transformed (`log1p`, not `log`, to handle the €50K floor safely).** The transform alone
only got skew to -1.63 (still "heavily skewed" by the |skew|>1 rule of thumb, and flipped
sign). Investigating why: a cluster of 6 players priced at €50K-€200K sat far enough below
the bulk of the data (bulk is roughly €10M-€65M) that they distorted the log-space
distribution into a long left tail. These were identified programmatically - 3×IQR below Q1
in log-space, the standard "extreme outlier" fence - which reproduced the exact same 6 rows
found by inspection, confirming they weren't representative noise. They were moved to
`excluded_players.csv` (reason: `"low-value outlier, distorts target skew"`), which improved
log-space skew to -1.31. Still not close to 0 - documented as a known limitation below, not
papered over.

**Linear Regression was chosen over Random Forest because performance was statistically
tied.** Test-set R² (log-space): 0.362 (Linear Regression) vs 0.359 (Random Forest) - a
trivial margin, unsurprising with ~240 training rows and 7 features (not enough complexity
for a Random Forest to exploit). Given the tie, the simpler, more interpretable model won.
Random Forest's code and trained artifact are kept in the repo (`train_models.py`,
`modeling/artifacts/random_forest.pkl`) but aren't used by the prediction interface.

## Known limitations

State plainly, not undersold:

- **Small dataset: 302 players in the final dataset, with only a 61-row test set.** Metrics
  from `train_models.py` are directional, not precise - they can shift meaningfully with a
  different `random_state` or a few more weeks of season data.
- **87.7% of players had 0 goals and 0 assists at the time of data collection** (only ~20
  matches played this season). Random Forest feature importances confirm it: `age` accounts
  for 0.741 of importance, `goals` for 0.042, `assists` for 0.006. In practice, both models
  are currently predicting transfer value mostly from **age and position**, not performance -
  which undercuts the point of a stats-driven predictor until more of the season plays out.
- **The model extrapolates poorly outside the range it was trained on.** A hypothetical
  22-year-old midfielder with 5 goals / 3 assists (values above the training data's observed
  max of 3 goals / 2 assists) produced a **predicted value of €501.9M** - more than Erling
  Haaland's actual €200M valuation - because a log-space linear fit compounds multiplicatively
  once you extrapolate past its training range. This is a known, guarded-against failure mode:
  `predict.py` now warns explicitly when goals/assists (or age) fall outside the range the
  model was trained/validated on, but the underlying model still produces the bad number if
  the warning is ignored.
- **Real-player accuracy is inconsistent.** From testing: Erling Haaland is under-predicted
  by 75% (€200M actual vs €50.1M predicted), while low-value squad players are sometimes
  over-predicted by more than 20x (Adam Smith: €300K actual vs €7.6M predicted, roughly 25x).
  Across all 302 players, only 19.2% land within ±20% of their actual value, and 37.1% are
  off by more than 2x (see `modeling/artifacts/predicted_vs_actual.png`).

## How to run it

**Setup:**
```bash
cd pl-transfer-value-predictor
pip install -r requirements.txt
```
Register a free API key at https://www.football-data.org/client/register, then put it in
`.env` (copy `.env.example` if starting fresh):
```
FOOTBALL_DATA_API_KEY=your_key_here
```
Note: if you hit `CERTIFICATE_VERIFY_FAILED` errors on Windows behind a TLS-inspecting proxy,
that's what `truststore` in `requirements.txt` is for - it's already wired in via
`data_collection/http_utils.py`.

**Run in order:**
```bash
cd data_collection
python fetch_player_stats.py       # -> data/raw/player_stats.csv (needs the API key)
python fetch_transfer_values.py    # -> data/raw/player_values.csv (no key needed)
python sanity_check.py             # optional: match-rate / data-quality diagnostics
python merge_dataset.py            # -> data/processed/player_dataset.csv (+ excluded_players.csv)
python data_quality_report.py      # optional: distribution/skew diagnostics

cd ../modeling
python train_models.py             # Linear Regression vs Random Forest comparison (80/20 split)
python train_final_model.py        # retrains Linear Regression on all rows -> linear_model.pkl
python predict.py                  # interactive CLI: look up a player or enter a hypothetical one
python visualize.py                # optional: predicted-vs-actual plot -> artifacts/predicted_vs_actual.png
```

`fetch_player_stats.py` is rate-limited to football-data.org's free tier (10 requests/minute)
and can take a few minutes to run through all 20 squads. Everything else is fast.

## Automatic updates

The data, model, and static site (`docs/`) refresh automatically every **Sunday at 03:00
UTC** via GitHub Actions (`.github/workflows/refresh_data.yml`). It runs the same pipeline
described above end to end - `fetch_player_stats.py` → `fetch_transfer_values.py` →
`merge_dataset.py` → `train_final_model.py` → `docs/export_for_web.py` - then commits and
pushes whatever changed in `data/`, `modeling/artifacts/`, and `docs/` using the repo's
built-in `GITHUB_TOKEN` (no personal access token needed). If the data is identical to last
week's, it skips the commit rather than creating an empty one.

**Safety**: if any pipeline step fails (API down, rate-limited, bad data) or the resulting
dataset has an implausibly low row count, the job fails loudly and stops *before* the
commit step - the live site is never overwritten with broken or partial data. Each run's
Actions log includes a summary (row counts before/after, whether a commit happened, and the
likely cause if it failed) at the top of the run page.

**Triggering it manually**: go to the repo's **Actions** tab → **Weekly data refresh** →
**Run workflow**.

**Adding/rotating the API key secret**: the workflow needs `FOOTBALL_DATA_API_KEY` available
as a GitHub Actions secret (never committed to the repo). To add or update it:
**Settings → Secrets and variables → Actions → New repository secret** (or edit the existing
one), name it exactly `FOOTBALL_DATA_API_KEY`, and paste in a key from
https://www.football-data.org/client/register.

## Future improvements

- **Re-run data collection later in the season.** Once more matches are played, goals/assists
  will have real variance, and the model should start using them instead of leaning on
  age/position as a proxy. Worth re-comparing Linear Regression vs Random Forest at that
  point too - the tie seen here may not hold with richer features.
- **Revisit `minutes_played`** once the Transfermarkt `appearances.csv` dataset rolls over to
  cover the current season (it was capped at 2026-05-24, last season, at the time this was
  built) - or once football-data.org's plan/tier is upgraded to include match lineup data.
- Consider whether the residual log-space skew (-1.31) needs a different transform, or
  whether it's structural (a genuine handful of superstar-valued outliers) rather than a data
  artifact worth removing further.
