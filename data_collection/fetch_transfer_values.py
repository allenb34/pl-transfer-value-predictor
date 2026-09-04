"""Collect current Premier League player market/transfer values.

football-data.org (used in fetch_player_stats.py) is NOT the source here.
See README notes in this repo / the assistant's chat summary for why a
second source was needed, and specifically why this one was chosen over
alternatives (live Transfermarkt scrapers).

Source: dcaribou/transfermarkt-datasets (CC0 1.0 Universal - public domain,
no attribution or license required). It's a community-maintained, weekly
refreshed mirror of Transfermarkt data, published as flat CSVs. We use
`players.csv`, which carries each player's current club and current market
value directly - no need to join against the separate valuation-history
table for a "current value" snapshot.

Direct file (no API key required): validated by hand on 2026-09-03 by
downloading and inspecting the header row.
"""
import sys
from pathlib import Path

import pandas as pd
import requests

from http_utils import log_error

PLAYERS_CSV_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz"
PREMIER_LEAGUE_COMPETITION_ID = "GB1"  # this dataset's code for the English top flight

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DOWNLOAD_PATH = RAW_DIR / "transfermarkt_players_raw.csv.gz"


def download_players_csv() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(PLAYERS_CSV_URL, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        log_error("fetch_transfer_values", f"failed to download players.csv.gz: {e}")
        print(f"FATAL: could not download transfer value dataset from {PLAYERS_CSV_URL}\n{e}", file=sys.stderr)
        sys.exit(1)

    RAW_DOWNLOAD_PATH.write_bytes(resp.content)
    return RAW_DOWNLOAD_PATH


def main():
    print(f"Downloading player market value dataset from {PLAYERS_CSV_URL} ...")
    raw_path = download_players_csv()
    print(f"  saved raw file to {raw_path}")

    try:
        df = pd.read_csv(raw_path, compression="gzip")
    except Exception as e:
        log_error("fetch_transfer_values", f"failed to parse downloaded CSV: {e}")
        print(f"FATAL: downloaded file could not be parsed as CSV: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {
        "name",
        "current_club_name",
        "current_club_domestic_competition_id",
        "market_value_in_eur",
        "last_season",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        log_error("fetch_transfer_values", f"dataset schema changed, missing columns: {missing_cols}")
        print(f"FATAL: expected columns not found: {missing_cols}. Dataset schema may have changed.", file=sys.stderr)
        sys.exit(1)

    pl_df = df[df["current_club_domestic_competition_id"] == PREMIER_LEAGUE_COMPETITION_ID].copy()

    # `current_club_id` lags for players who haven't appeared recently (e.g. released,
    # or their club was relegated years ago and the row was never revisited), so the raw
    # GB1 filter above pulls in ~35+ clubs going back years, not just the current 20.
    # Restricting to each player's most recent recorded season (dynamically, not a
    # hardcoded year - the dataset's own refresh cadence can lag right after a season
    # rolls over) reliably isolates the actual current top-flight roster.
    latest_season = pl_df["last_season"].max()
    stale_count = len(pl_df) - len(pl_df[pl_df["last_season"] == latest_season])
    pl_df = pl_df[pl_df["last_season"] == latest_season].copy()
    print(
        f"  {len(pl_df)} players on a current Premier League club (season {latest_season}); "
        f"dropped {stale_count} stale rows from clubs no longer in the league."
    )

    missing_value_count = 0
    for _, row in pl_df[pl_df["market_value_in_eur"].isna()].iterrows():
        log_error(
            "fetch_transfer_values",
            f"no market value listed for {row.get('name')} ({row.get('current_club_name')})",
        )
        missing_value_count += 1
    if missing_value_count:
        print(f"  {missing_value_count} players have no listed market value (logged to collection_errors.log).")

    out_df = pl_df.rename(columns={"current_club_name": "team", "market_value_in_eur": "transfer_value"})[
        ["name", "team", "transfer_value"]
    ]

    out_path = RAW_DIR / "player_values.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} player rows to {out_path}")


if __name__ == "__main__":
    main()
