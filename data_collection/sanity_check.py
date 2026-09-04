"""Load player_stats.csv and player_values.csv and report data quality.

Reports: row counts, % missing stats fields, % missing transfer values, and
name-matching coverage between the two sources.

Matching strategy (name-only matching was unreliable - e.g. it paired
Crystal Palace's "Ben Chilwell" with Cardiff City's "Rubin Colwill" purely
because the strings look similar):
  1. Exact match on (normalized name, normalized club).
  2. Fuzzy name match, but only among candidates in the SAME normalized
     club - a fuzzy match is never allowed to cross clubs.
  3. Anything still unmatched is written to unmatched_players.csv for
     manual review rather than guessed.
"""
import difflib
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STATS_PATH = RAW_DIR / "player_stats.csv"
VALUES_PATH = RAW_DIR / "player_values.csv"
UNMATCHED_PATH = RAW_DIR / "unmatched_players.csv"

# football-data.org uses short club names ("Man City"), the Transfermarkt
# dataset uses full/legal names ("Manchester City"). Generic suffix-stripping
# (below) resolves most of these (e.g. "Bournemouth" <-> "AFC Bournemouth"),
# but a handful of short forms have no algorithmic relationship to the full
# name and need an explicit lookup. Keyed and valued by the *already*
# generic-normalized form (lowercase, no punctuation, no fc/afc token).
CLUB_ALIASES = {
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "brighton": "brighton hove albion",
    "brighton hove": "brighton hove albion",
    "wolves": "wolverhampton wanderers",
    "nottingham": "nottingham forest",
    "forest": "nottingham forest",
    "newcastle": "newcastle united",
    "west brom": "west bromwich albion",
    "albion": "west bromwich albion",
    "sheffield utd": "sheffield united",
    "leeds": "leeds united",
    "leicester": "leicester city",
    "norwich": "norwich city",
    "west ham": "west ham united",
    "wigan": "wigan athletic",
    "cardiff": "cardiff city",
    "swansea": "swansea city",
    "huddersfield": "huddersfield town",
    "stoke": "stoke city",
    "hull": "hull city",
    "qpr": "queens park rangers",
    "coventry": "coventry city",
    "ipswich": "ipswich town",
    "southampton": "southampton fc",
}

NAME_SUFFIXES = {"jr", "jnr", "snr", "sr", "ii", "iii", "iv"}
GENERIC_CLUB_TOKENS = {"fc", "afc"}


def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    text = _strip_accents(name).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [t for t in text.split() if t not in NAME_SUFFIXES]
    return " ".join(tokens)


def normalize_club(club: str) -> str:
    if not isinstance(club, str):
        return ""
    text = _strip_accents(club).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [t for t in text.split() if t not in GENERIC_CLUB_TOKENS]
    normalized = " ".join(tokens)
    return CLUB_ALIASES.get(normalized, normalized)


def load_csv(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [MISSING] {label}: {path} does not exist. Run the corresponding fetch script first.")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [ERROR] {label}: failed to read {path}: {e}")
        return None


def report_missing(df: pd.DataFrame, columns: list, label: str) -> None:
    total = len(df)
    if total == 0:
        print(f"  {label}: 0 rows, nothing to check.")
        return
    for col in columns:
        if col not in df.columns:
            print(f"  {label}: column '{col}' not found.")
            continue
        missing = df[col].isna().sum()
        pct = 100 * missing / total
        print(f"  {label}: '{col}' missing in {missing}/{total} rows ({pct:.1f}%)")


def match_players(stats_df: pd.DataFrame, values_df: pd.DataFrame):
    """Returns (matches, unmatched_stats_idx, unmatched_values_idx).

    matches: list of dicts with stats_idx, values_idx, method.
    """
    stats_name_key = stats_df["name"].apply(normalize_name)
    stats_club_key = stats_df["team"].apply(normalize_club)
    values_name_key = values_df["name"].apply(normalize_name)
    values_club_key = values_df["team"].apply(normalize_club)

    # index values rows by (name, club) for O(1) exact lookup, and by club
    # alone for the fuzzy fallback pool.
    values_by_key: dict[tuple, list[int]] = {}
    values_by_club: dict[str, list[int]] = {}
    for idx in values_df.index:
        key = (values_name_key[idx], values_club_key[idx])
        values_by_key.setdefault(key, []).append(idx)
        values_by_club.setdefault(values_club_key[idx], []).append(idx)

    used_values: set = set()
    matches = []
    unmatched_stats = []

    # Pass 1: exact (name, club) match.
    for idx in stats_df.index:
        key = (stats_name_key[idx], stats_club_key[idx])
        candidates = [v for v in values_by_key.get(key, []) if v not in used_values]
        if candidates:
            v_idx = candidates[0]
            used_values.add(v_idx)
            matches.append({"stats_idx": idx, "values_idx": v_idx, "method": "exact_name_club"})
        else:
            unmatched_stats.append(idx)

    # Pass 2: same-club matches only, in two flavors:
    #   (a) token containment - e.g. Transfermarkt's "Alisson" vs football-data's
    #       "Alisson Becker": every token of the shorter name appears in the
    #       longer one. Only applied when exactly one same-club candidate
    #       qualifies, so it can't silently pick between two teammates who
    #       share a first name.
    #   (b) difflib string-similarity, for typos/transliteration differences
    #       (Vitalii/Vitaliy, Hakon/Hákon) that token containment won't catch.
    still_unmatched_stats = []
    for idx in unmatched_stats:
        club = stats_club_key[idx]
        pool = [v for v in values_by_club.get(club, []) if v not in used_values]
        if not pool:
            still_unmatched_stats.append(idx)
            continue

        query_tokens = set(stats_name_key[idx].split())
        containment_hits = []
        for v in pool:
            candidate_tokens = set(values_name_key[v].split())
            if not query_tokens or not candidate_tokens:
                continue
            if query_tokens <= candidate_tokens or candidate_tokens <= query_tokens:
                containment_hits.append(v)

        if len(containment_hits) == 1:
            v_idx = containment_hits[0]
            used_values.add(v_idx)
            matches.append({"stats_idx": idx, "values_idx": v_idx, "method": "same_club_token_subset"})
            continue

        pool_names = {values_name_key[v]: v for v in pool}
        close = difflib.get_close_matches(stats_name_key[idx], list(pool_names.keys()), n=1, cutoff=0.82)
        if close:
            v_idx = pool_names[close[0]]
            used_values.add(v_idx)
            matches.append({"stats_idx": idx, "values_idx": v_idx, "method": "fuzzy_same_club"})
        else:
            still_unmatched_stats.append(idx)

    unmatched_values = [v for v in values_df.index if v not in used_values]
    return matches, still_unmatched_stats, unmatched_values, stats_club_key, values_club_key


def report_name_matches(stats_df: pd.DataFrame, values_df: pd.DataFrame) -> None:
    matches, unmatched_stats, unmatched_values, stats_club_key, values_club_key = match_players(stats_df, values_df)
    stats_clubs_present = set(stats_club_key)
    values_clubs_present = set(values_club_key)

    total_stats = len(stats_df)
    match_rate = 100 * len(matches) / total_stats if total_stats else 0
    exact_count = sum(1 for m in matches if m["method"] == "exact_name_club")
    subset_count = sum(1 for m in matches if m["method"] == "same_club_token_subset")
    fuzzy_count = sum(1 for m in matches if m["method"] == "fuzzy_same_club")

    print(f"  Matched: {len(matches)}/{total_stats} player_stats rows ({match_rate:.1f}%)")
    print(f"    - exact (name + club): {exact_count}")
    print(f"    - same club, name token subset (e.g. 'Alisson' <-> 'Alisson Becker'): {subset_count}")
    print(f"    - same club, fuzzy string match: {fuzzy_count}")
    print(f"  Unmatched in player_stats.csv: {len(unmatched_stats)}")
    print(f"  Unmatched in player_values.csv: {len(unmatched_values)}")

    non_exact = [m for m in matches if m["method"] != "exact_name_club"][:10]
    if non_exact:
        print("\n  Sample non-exact matches - spot-check these:")
        for m in non_exact:
            s_row = stats_df.loc[m["stats_idx"]]
            v_row = values_df.loc[m["values_idx"]]
            print(f"    [{m['method']}] '{s_row['name']}' ({s_row['team']})  <->  '{v_row['name']}' ({v_row['team']})")

    # Most unmatched rows aren't a matching-algorithm failure at all: a club can be
    # entirely absent from the other source (promoted/relegated between the two
    # sources' season snapshots), or the Transfermarkt dataset can track reserve/
    # fringe players who never appear in football-data.org's registered squad.
    # Tagging that up front is what actually makes this file hand-reviewable.
    unmatched_rows = []
    for idx in unmatched_stats:
        row = stats_df.loc[idx]
        reason = (
            "club not present in player_values.csv (season/roster mismatch)"
            if stats_club_key[idx] not in values_clubs_present
            else "no name match found within same club - needs manual review"
        )
        unmatched_rows.append({"name": row["name"], "team": row["team"], "source": "player_stats", "likely_reason": reason})
    for idx in unmatched_values:
        row = values_df.loc[idx]
        reason = (
            "club not present in player_stats.csv (season/roster mismatch)"
            if values_club_key[idx] not in stats_clubs_present
            else "no name match found within same club - needs manual review"
        )
        unmatched_rows.append({"name": row["name"], "team": row["team"], "source": "player_values", "likely_reason": reason})

    unmatched_df = pd.DataFrame(unmatched_rows, columns=["name", "team", "source", "likely_reason"])
    unmatched_df.to_csv(UNMATCHED_PATH, index=False)

    reason_counts = unmatched_df["likely_reason"].value_counts()
    needs_review = reason_counts.get("no name match found within same club - needs manual review", 0)
    structural = len(unmatched_df) - needs_review
    print(f"\n  Wrote {len(unmatched_df)} unmatched rows to {UNMATCHED_PATH}:")
    print(f"    - {structural} are club-level season/roster mismatches (nothing to fix - the player's club isn't in the other source at all)")
    print(f"    - {needs_review} have a same-club counterpart missing and are worth a manual look")


def main():
    print("=" * 60)
    print("SANITY CHECK: Premier League Transfer Value Predictor data")
    print("=" * 60)

    print("\n[1] Loading files")
    stats_df = load_csv(STATS_PATH, "player_stats.csv")
    values_df = load_csv(VALUES_PATH, "player_values.csv")

    if stats_df is None or values_df is None:
        print("\nCannot continue sanity check without both files. Exiting.")
        sys.exit(1)

    print("\n[2] Row counts")
    print(f"  player_stats.csv: {len(stats_df)} rows")
    print(f"  player_values.csv: {len(values_df)} rows")

    print("\n[3] Missing data in player_stats.csv")
    report_missing(stats_df, ["position", "age"], "player_stats")

    print("\n[4] Missing data in player_values.csv")
    report_missing(values_df, ["transfer_value"], "player_values")

    print("\n[5] Name matching between sources")
    report_name_matches(stats_df, values_df)

    print("\n" + "=" * 60)
    print("Sanity check complete.")
    print("=" * 60)


if __name__ == "__main__":
    # Windows consoles default to a non-UTF-8 codepage, which crashes print()
    # on non-ASCII player names (e.g. "Jens Hjertø-Dahl") - replace
    # rather than crash on characters the terminal itself can't render.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except Exception as e:
        print(f"\nUNEXPECTED ERROR during sanity check: {e}", file=sys.stderr)
        sys.exit(1)
