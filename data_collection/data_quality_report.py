"""Data quality report for data/processed/player_dataset.csv.

Prints row count, transfer_value distribution (with a skew flag relevant to
choosing a log transform later), goals/assists distribution, position
breakdown, and age distribution. Reports only - no modeling.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "player_dataset.csv"

# A common rule of thumb: |skewness| > 1 is "highly skewed".
SKEW_THRESHOLD = 1.0


def main():
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} does not exist. Run merge_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)

    print("=" * 60)
    print("DATA QUALITY REPORT: player_dataset.csv")
    print("=" * 60)

    print(f"\n[1] Final row count: {len(df)}")

    print("\n[2] transfer_value distribution")
    tv = df["transfer_value"]
    print(f"  min:    {tv.min():,.0f}")
    print(f"  max:    {tv.max():,.0f}")
    print(f"  median: {tv.median():,.0f}")
    print(f"  mean:   {tv.mean():,.0f}")
    skew = tv.skew()
    print(f"  skewness: {skew:.2f}", end=" ")
    if abs(skew) > SKEW_THRESHOLD:
        print(f"-> HEAVILY SKEWED (|skew| > {SKEW_THRESHOLD}); consider a log transform of the target before modeling.")
    else:
        print(f"-> within +/-{SKEW_THRESHOLD}, not strongly skewed.")

    if "log_transfer_value" in df.columns:
        print("\n[2b] log_transfer_value distribution (log1p of transfer_value)")
        ltv = df["log_transfer_value"]
        print(f"  min:    {ltv.min():.3f}")
        print(f"  max:    {ltv.max():.3f}")
        print(f"  median: {ltv.median():.3f}")
        print(f"  mean:   {ltv.mean():.3f}")
        log_skew = ltv.skew()
        print(f"  skewness: {log_skew:.2f}", end=" ")
        if abs(log_skew) > SKEW_THRESHOLD:
            print(f"-> still HEAVILY SKEWED (|skew| > {SKEW_THRESHOLD}).")
        else:
            print(f"-> within +/-{SKEW_THRESHOLD}, not strongly skewed (improved from {skew:.2f}).")

    print("\n[3] goals/assists distribution")
    both_zero = ((df["goals"] == 0) & (df["assists"] == 0)).sum()
    pct_both_zero = 100 * both_zero / len(df) if len(df) else 0
    print(f"  goals:   min={df['goals'].min()}, max={df['goals'].max()}, mean={df['goals'].mean():.2f}")
    print(f"  assists: min={df['assists'].min()}, max={df['assists'].max()}, mean={df['assists'].mean():.2f}")
    print(f"  players with 0 goals AND 0 assists: {both_zero}/{len(df)} ({pct_both_zero:.1f}%)")
    print("  (expected: only ~20 matches played so far this season)")

    print("\n[4] Position breakdown")
    print(df["position"].value_counts(dropna=False).to_string())

    print("\n[5] Age distribution")
    age = df["age"]
    print(f"  min:    {age.min():.0f}")
    print(f"  max:    {age.max():.0f}")
    print(f"  median: {age.median():.0f}")
    print(f"  mean:   {age.mean():.1f}")

    print("\n" + "=" * 60)
    print("Report complete.")
    print("=" * 60)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except Exception as e:
        print(f"\nUNEXPECTED ERROR during data quality report: {e}", file=sys.stderr)
        sys.exit(1)
