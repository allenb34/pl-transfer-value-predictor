"""Collect Premier League player stats from football-data.org (v4).

Free tier limits (confirmed against https://docs.football-data.org and
https://www.football-data.org/pricing as of 2026-09-03): 10 requests/minute,
12 competitions included (Premier League is one of them). Requests are
throttled to stay under that limit, with retry/backoff on 429s and 5xx
errors (see http_utils.py).

Note: minutes_played was dropped from this pipeline entirely (2026-09-04
decision) after confirming with a live key that the free tier's
/v4/matches/{id} response doesn't include lineup/substitution/booking data
at all - there's no way to derive it from this API on this plan.

Output: data/raw/player_stats.csv with columns
    name, team, position, age, goals, assists
Raw API responses are cached under data/raw/ for reproducibility/debugging.
"""
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from http_utils import RateLimiter, get_json, log_error

BASE_URL = "https://api.football-data.org/v4"
COMPETITION_CODE = "PL"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SQUADS_DIR = RAW_DIR / "squads"

# Free tier: 10 requests/minute. Use 6.5s spacing to stay safely under that.
RATE_LIMITER = RateLimiter(min_interval=6.5)


def load_api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print(
            "ERROR: FOOTBALL_DATA_API_KEY is not set.\n"
            "Get a free key at https://www.football-data.org/client/register "
            "and add it to .env as FOOTBALL_DATA_API_KEY=<your key>.",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def calculate_age(date_of_birth: str | None) -> float | None:
    if not date_of_birth:
        return None
    try:
        dob = datetime.strptime(date_of_birth[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def fetch_teams(session, headers) -> list[dict]:
    data = get_json(
        session,
        f"{BASE_URL}/competitions/{COMPETITION_CODE}/teams",
        RATE_LIMITER,
        context="competitions/PL/teams",
        headers=headers,
    )
    if data is None:
        print("FATAL: could not fetch the Premier League team list. See collection_errors.log.")
        sys.exit(1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / "teams_raw.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    teams = data.get("teams", [])
    if not teams:
        log_error("competitions/PL/teams", "response contained no teams")
    return teams


def fetch_squads(session, headers, teams: list[dict]) -> list[dict]:
    """Returns a flat list of player dicts with team/position/age attached."""
    SQUADS_DIR.mkdir(parents=True, exist_ok=True)
    players = []
    for team in teams:
        team_id = team.get("id")
        team_name = team.get("shortName") or team.get("name") or f"team_{team_id}"
        if team_id is None:
            log_error("fetch_squads", f"team entry missing id: {team}")
            continue

        data = get_json(
            session,
            f"{BASE_URL}/teams/{team_id}",
            RATE_LIMITER,
            context=f"teams/{team_id} ({team_name})",
            headers=headers,
        )
        if data is None:
            log_error(f"team {team_name} ({team_id})", "squad fetch failed, skipping team")
            continue

        with open(SQUADS_DIR / f"team_{team_id}.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

        squad = data.get("squad", [])
        if not squad:
            log_error(f"team {team_name} ({team_id})", "squad list empty in response")

        for player in squad:
            try:
                player_id = player["id"]
                name = player["name"]
            except KeyError as e:
                log_error(f"team {team_name} ({team_id})", f"player record missing required field {e}: {player}")
                continue

            players.append(
                {
                    "player_id": player_id,
                    "name": name,
                    "team": team_name,
                    "position": player.get("position"),
                    "age": calculate_age(player.get("dateOfBirth")),
                }
            )
    return players


def fetch_goals_assists(session, headers) -> dict:
    """Returns {player_id: {"goals": int, "assists": int}} from the scorers endpoint.

    Only players who have scored or assisted appear here; anyone else is
    genuinely 0, not missing, and is filled in later during the merge.
    """
    data = get_json(
        session,
        f"{BASE_URL}/competitions/{COMPETITION_CODE}/scorers",
        RATE_LIMITER,
        context="competitions/PL/scorers",
        headers=headers,
        params={"limit": 500},
    )
    if data is None:
        log_error("competitions/PL/scorers", "fetch failed; goals/assists will be 0 for all players")
        return {}

    with open(RAW_DIR / "scorers_raw.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

    stats = {}
    for entry in data.get("scorers", []):
        player = entry.get("player", {})
        player_id = player.get("id")
        if player_id is None:
            log_error("competitions/PL/scorers", f"scorer entry missing player id: {entry}")
            continue
        stats[player_id] = {
            "goals": entry.get("goals") or 0,
            "assists": entry.get("assists") or 0,
        }
    return stats


def main():
    api_key = load_api_key()
    headers = {"X-Auth-Token": api_key}
    session = requests.Session()

    print("Fetching Premier League team list...")
    teams = fetch_teams(session, headers)
    print(f"  {len(teams)} teams found.")

    print("Fetching squads (stats: position, age)...")
    players = fetch_squads(session, headers, teams)
    print(f"  {len(players)} players collected across all squads.")

    print("Fetching goals/assists from the scorers leaderboard...")
    goals_assists = fetch_goals_assists(session, headers)
    print(f"  {len(goals_assists)} players with recorded goals/assists.")

    for player in players:
        pid = player.pop("player_id")
        ga = goals_assists.get(pid, {"goals": 0, "assists": 0})
        player["goals"] = ga["goals"]
        player["assists"] = ga["assists"]

    df = pd.DataFrame(players, columns=["name", "team", "position", "age", "goals", "assists"])
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "player_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} player rows to {out_path}")


if __name__ == "__main__":
    main()
