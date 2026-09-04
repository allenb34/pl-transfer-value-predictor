"""'Predict any player' CLI - looks up a real player or takes hypothetical
stats and predicts transfer value with the finalized Linear Regression model.

Uses the model saved by train_final_model.py (trained on all 302 rows) and
reuses the exact position_encoder.pkl fit during train_models.py's
train/test split - never refit here, so encoding stays consistent with how
the model was evaluated.
"""
import difflib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_models import DATASET_PATH, build_features

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "linear_model.pkl"
ENCODER_PATH = ARTIFACTS_DIR / "position_encoder.pkl"

MIN_PLAUSIBLE_AGE = 15
MAX_PLAUSIBLE_AGE = 45


class PredictionError(ValueError):
    pass


def _load_artifacts():
    missing = [p for p in (MODEL_PATH, ENCODER_PATH) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        print(
            f"ERROR: missing artifact(s): {names}\n"
            "Run train_models.py (fits the encoder) then train_final_model.py (trains the "
            "final model) before using predict.py.",
            file=sys.stderr,
        )
        sys.exit(1)
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


MODEL, ENCODER = _load_artifacts()
VALID_POSITIONS = list(ENCODER.categories_[0])


def _training_data_ranges() -> tuple[int, int]:
    """Max goals/assists actually seen in training data. The model was fit on a
    log-transformed target with only ~20 matches played this season (87.7%
    of players at 0/0) - feeding it goals/assists well above what it's seen
    extrapolates the log-linear fit and can blow up wildly after expm1.
    """
    if not DATASET_PATH.exists():
        return 3, 2  # fallback matching the dataset at the time this was written
    df = pd.read_csv(DATASET_PATH)
    return int(df["goals"].max()), int(df["assists"].max())


MAX_TRAINED_GOALS, MAX_TRAINED_ASSISTS = _training_data_ranges()


def predict_transfer_value(name: str, club: str, position: str, age: float, goals: int, assists: int) -> float:
    """Predicts real-euro transfer value for a player's stats.

    `name`/`club` aren't model features (the model only uses position/age/
    goals/assists) - they're accepted so callers can pass a full player
    record without picking it apart first, and so error messages below can
    refer to who the prediction was for.

    Raises PredictionError if `position` isn't one of the categories the
    encoder/model were fit on.
    """
    if position not in VALID_POSITIONS:
        raise PredictionError(
            f"Unknown position '{position}' for {name or 'this player'}. "
            f"Valid options are: {', '.join(VALID_POSITIONS)}"
        )

    row = pd.DataFrame([{"goals": goals, "assists": assists, "age": age, "position": position}])
    X, _ = build_features(row, encoder=ENCODER)
    pred_log = MODEL.predict(X)[0]
    return float(np.expm1(pred_log))


def _load_player_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} does not exist. Run merge_dataset.py first.", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(DATASET_PATH)


def _prompt(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        print("\nNo more input - exiting.")
        sys.exit(0)


def _prompt_float(msg: str, allow_blank_cancel: bool = True):
    while True:
        raw = _prompt(msg).strip()
        if allow_blank_cancel and raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            print(f"  '{raw}' isn't a number. Please try again.")


def _prompt_int(msg: str, allow_blank_cancel: bool = True):
    while True:
        raw = _prompt(msg).strip()
        if allow_blank_cancel and raw == "":
            return None
        try:
            value = int(raw)
        except ValueError:
            print(f"  '{raw}' isn't a whole number. Please try again.")
            continue
        if value < 0:
            print("  Value can't be negative. Please try again.")
            continue
        return value


def _prompt_position() -> str | None:
    while True:
        raw = _prompt(f"Position ({'/'.join(VALID_POSITIONS)}), or blank to cancel: ").strip()
        if raw == "":
            return None
        # allow case-insensitive / partial match against the trained categories
        matches = [p for p in VALID_POSITIONS if p.lower() == raw.lower()]
        if not matches:
            matches = [p for p in VALID_POSITIONS if raw.lower() in p.lower()]
        if len(matches) == 1:
            return matches[0]
        print(f"  '{raw}' isn't one of the trained positions. Valid options: {', '.join(VALID_POSITIONS)}")


def _find_player(df: pd.DataFrame, query: str) -> pd.Series | None:
    """Case-insensitive substring search over `name`; if multiple/no exact
    hits, offers close matches and lets the user pick or retype.
    """
    query = query.strip()
    if not query:
        return None

    exact = df[df["name"].str.lower() == query.lower()]
    if len(exact) == 1:
        return exact.iloc[0]

    substring = df[df["name"].str.contains(query, case=False, na=False, regex=False)]
    if len(substring) == 1:
        return substring.iloc[0]
    if len(substring) > 1:
        print(f"\n  Multiple players match '{query}':")
        options = substring.reset_index(drop=True)
        for i, row in options.iterrows():
            print(f"    [{i + 1}] {row['name']} ({row['club']})")
        choice = _prompt_int("  Pick a number, or blank to cancel: ")
        if choice is None or not (1 <= choice <= len(options)):
            return None
        return options.iloc[choice - 1]

    close = difflib.get_close_matches(query, df["name"].tolist(), n=5, cutoff=0.5)
    if close:
        print(f"\n  No player named '{query}'. Did you mean:")
        for i, name in enumerate(close, start=1):
            print(f"    [{i}] {name}")
        choice = _prompt_int("  Pick a number, or blank to cancel: ")
        if choice is None or not (1 <= choice <= len(close)):
            return None
        return df[df["name"] == close[choice - 1]].iloc[0]

    print(f"\n  No player named '{query}' found, and no close matches either.")
    return None


def _show_existing_player_prediction(row: pd.Series) -> None:
    actual = row["transfer_value"]
    predicted = predict_transfer_value(
        row["name"], row["club"], row["position"], row["age"], row["goals"], row["assists"]
    )
    diff = predicted - actual
    pct_error = 100 * diff / actual if actual else float("nan")

    print(f"\n{'=' * 60}")
    print(f"{row['name']} ({row['club']})")
    print("=" * 60)
    print(f"  Position:        {row['position']}")
    print(f"  Age:             {row['age']:.0f}")
    print(f"  Goals:           {row['goals']:.0f}")
    print(f"  Assists:         {row['assists']:.0f}")
    print("-" * 60)
    print(f"  Actual value:    EUR{actual:,.0f}")
    print(f"  Predicted value: EUR{predicted:,.0f}")
    print(f"  Difference:      EUR{diff:,.0f} ({'over' if diff > 0 else 'under'}-predicted)")
    print(f"  % error:         {pct_error:+.1f}%")
    print("=" * 60)


def _run_existing_player_flow(df: pd.DataFrame) -> None:
    query = _prompt("\nType a player name to search (or blank to go back): ")
    if not query.strip():
        return
    row = _find_player(df, query)
    if row is None:
        return
    _show_existing_player_prediction(row)


def _run_custom_player_flow() -> None:
    print("\n--- Custom / hypothetical player ---")
    position = _prompt_position()
    if position is None:
        return

    age = _prompt_float("Age: ")
    if age is None:
        return
    if not (MIN_PLAUSIBLE_AGE <= age <= MAX_PLAUSIBLE_AGE):
        print(
            f"  WARNING: age {age:.0f} is outside {MIN_PLAUSIBLE_AGE}-{MAX_PLAUSIBLE_AGE}, the range the "
            "model was trained and validated on. The prediction below is an extrapolation and may be unreliable."
        )

    goals = _prompt_int("Goals: ")
    if goals is None:
        return
    assists = _prompt_int("Assists: ")
    if assists is None:
        return
    if goals > MAX_TRAINED_GOALS or assists > MAX_TRAINED_ASSISTS:
        print(
            f"  WARNING: the training data only goes up to {MAX_TRAINED_GOALS} goals / "
            f"{MAX_TRAINED_ASSISTS} assists this early in the season (87.7% of players are still at "
            "0/0). This is a log-space linear model, so goals/assists above that range extrapolate "
            "and the prediction below can be wildly inflated - treat it with real skepticism."
        )

    try:
        predicted = predict_transfer_value(None, None, position, age, goals, assists)
    except PredictionError as e:
        print(f"  ERROR: {e}")
        return

    print(f"\n{'=' * 60}")
    print("HYPOTHETICAL PLAYER (not a real listed player)")
    print("=" * 60)
    print(f"  Position: {position}")
    print(f"  Age:      {age:.0f}")
    print(f"  Goals:    {goals}")
    print(f"  Assists:  {assists}")
    print("-" * 60)
    print(f"  Predicted value: EUR{predicted:,.0f}")
    print("=" * 60)


def main():
    df = _load_player_dataset()
    print("Premier League Transfer Value Predictor")
    print(f"({len(df)} players loaded from player_dataset.csv, Linear Regression model)")

    while True:
        print("\nWhat would you like to do?")
        print("  1) Look up an existing player")
        print("  2) Enter a custom/hypothetical player")
        print("  3) Quit")
        choice = _prompt("> ").strip()

        if choice == "1":
            _run_existing_player_flow(df)
        elif choice == "2":
            _run_custom_player_flow()
        elif choice == "3":
            print("Goodbye.")
            return
        else:
            print(f"  '{choice}' isn't a valid option. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted - goodbye.")
