"""Merge player_stats.csv and player_values.csv into a single modeling-ready dataset.

Reuses the exact club-scoped matching logic from sanity_check.py (exact
name+club -> same-club token-subset -> same-club fuzzy -> unmatched) rather
than reimplementing it, so the match rate reported by sanity_check.py and
the rows that actually make it into the merged dataset can never drift apart.

Every row of player_stats.csv ends up in exactly one of:
  - data/processed/player_dataset.csv (matched AND has a non-null transfer_value)
  - data/processed/excluded_players.csv (everything else, with a reason)
so the two files' row counts always sum to len(player_stats.csv).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sanity_check import STATS_PATH, VALUES_PATH, load_csv, match_players

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "player_dataset.csv"
EXCLUDED_PATH = PROCESSED_DIR / "excluded_players.csv"

OUTPUT_COLUMNS = ["name", "club", "position", "age", "goals", "assists", "transfer_value", "log_transfer_value"]

# football-data.org has returned corrupted dateOfBirth values for some players
# (e.g. implying age 1) - anything outside a plausible professional range is
# a source data error, not a real player, and gets excluded rather than fed
# into modeling.
MIN_PLAUSIBLE_AGE = 15
MAX_PLAUSIBLE_AGE = 45

# Extreme-low-end outlier cutoff for the target. Applied in log space (where
# the skew distortion was diagnosed) using 3x IQR below Q1 - the standard
# "extreme outlier" threshold (1.5x is the usual "mild outlier" fence). This
# reproduces the exact same 6-row, EUR50K-200K cluster identified by eye in
# the prior data quality report, but programmatically rather than by name.
OUTLIER_IQR_MULTIPLIER = 3.0


def main():
    stats_df = load_csv(STATS_PATH, "player_stats.csv")
    values_df = load_csv(VALUES_PATH, "player_values.csv")
    if stats_df is None or values_df is None:
        print("\nCannot merge without both files. Run the fetch scripts first.")
        sys.exit(1)

    matches, unmatched_stats, _unmatched_values, stats_club_key, values_club_key = match_players(stats_df, values_df)
    values_clubs_present = set(values_club_key)

    dataset_rows = []
    excluded_rows = []

    for m in matches:
        s_row = stats_df.loc[m["stats_idx"]]
        v_row = values_df.loc[m["values_idx"]]
        transfer_value = v_row["transfer_value"]
        if pd.isna(transfer_value):
            excluded_rows.append(
                {"name": s_row["name"], "team": s_row["team"], "reason": "no transfer value"}
            )
            continue
        age = s_row["age"]
        if pd.isna(age) or not (MIN_PLAUSIBLE_AGE <= age <= MAX_PLAUSIBLE_AGE):
            excluded_rows.append({"name": s_row["name"], "team": s_row["team"], "reason": "implausible age"})
            continue
        dataset_rows.append(
            {
                "name": s_row["name"],
                "club": s_row["team"],
                "position": s_row["position"],
                "age": age,
                "goals": s_row["goals"],
                "assists": s_row["assists"],
                "transfer_value": transfer_value,
            }
        )

    for idx in unmatched_stats:
        s_row = stats_df.loc[idx]
        reason = (
            "promoted-club gap (club not present in player_values.csv)"
            if stats_club_key[idx] not in values_clubs_present
            else "unmatched name"
        )
        excluded_rows.append({"name": s_row["name"], "team": s_row["team"], "reason": reason})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if dataset_rows:
        dataset_df = pd.DataFrame(dataset_rows)
        dataset_df["log_transfer_value"] = np.log1p(dataset_df["transfer_value"])
        dataset_df = dataset_df[OUTPUT_COLUMNS]
    else:
        dataset_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    outlier_lower_bound = None
    if len(dataset_df) > 0:
        q1 = dataset_df["log_transfer_value"].quantile(0.25)
        q3 = dataset_df["log_transfer_value"].quantile(0.75)
        iqr = q3 - q1
        outlier_lower_bound = q1 - OUTLIER_IQR_MULTIPLIER * iqr

        is_outlier = dataset_df["log_transfer_value"] < outlier_lower_bound
        for _, row in dataset_df[is_outlier].iterrows():
            excluded_rows.append(
                {"name": row["name"], "team": row["club"], "reason": "low-value outlier, distorts target skew"}
            )
        dataset_df = dataset_df[~is_outlier].reset_index(drop=True)

    dataset_df.to_csv(DATASET_PATH, index=False)

    excluded_df = pd.DataFrame(excluded_rows, columns=["name", "team", "reason"])
    excluded_df.to_csv(EXCLUDED_PATH, index=False)

    assert len(dataset_df) + len(excluded_df) == len(stats_df), (
        f"{len(dataset_df)} + {len(excluded_df)} != {len(stats_df)} - "
        "every player_stats.csv row must land in exactly one output file"
    )

    print(f"player_stats.csv rows:  {len(stats_df)}")
    print(f"  -> player_dataset.csv:   {len(dataset_df)}")
    print(f"  -> excluded_players.csv: {len(excluded_df)}")
    print("\nExclusion reasons:")
    print(excluded_df["reason"].value_counts().to_string())

    implausible_age_count = (excluded_df["reason"] == "implausible age").sum()
    print(f"\nExcluded for implausible age (outside {MIN_PLAUSIBLE_AGE}-{MAX_PLAUSIBLE_AGE}): {implausible_age_count}")
    print("Age distribution in player_dataset.csv after the filter:")
    print(f"  min:    {dataset_df['age'].min():.0f}")
    print(f"  max:    {dataset_df['age'].max():.0f}")
    print(f"  median: {dataset_df['age'].median():.0f}")

    outlier_count = (excluded_df["reason"] == "low-value outlier, distorts target skew").sum()
    print(
        f"\nExcluded as low-value outliers ({OUTLIER_IQR_MULTIPLIER}x IQR below Q1 in log space, "
        f"bound={outlier_lower_bound:.3f} log / {np.expm1(outlier_lower_bound):,.0f} EUR): {outlier_count}"
    )
    print("Skewness after removing them:")
    print(f"  transfer_value skew:     {dataset_df['transfer_value'].skew():.2f}")
    print(f"  log_transfer_value skew: {dataset_df['log_transfer_value'].skew():.2f}")

    print(f"\nSaved {DATASET_PATH}")
    print(f"Saved {EXCLUDED_PATH}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
