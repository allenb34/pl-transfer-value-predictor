"""Train and compare two baseline regression models on player_dataset.csv.

Features: goals, assists, age, position (one-hot encoded).
Target: log_transfer_value (the raw target is heavily right-skewed - see
data_quality_report.py - so we fit in log space and convert predictions
back to euros with np.expm1 for anything a human needs to read).

This step only trains and evaluates two models for comparison. It does not
build the "predict any player" interface - that's a separate step. To keep
that step consistent with what's fit here, the trained models and the
position OneHotEncoder are persisted to modeling/artifacts/ so the interface
reuses the exact same encoding rather than re-deriving it.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "player_dataset.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

RANDOM_STATE = 42
TEST_SIZE = 0.20
NUMERIC_FEATURES = ["goals", "assists", "age"]

# Below this, a test-set metric is too noisy to trust on its own.
MIN_MEANINGFUL_TEST_ROWS = 100


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} does not exist. Run merge_dataset.py first.", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(DATASET_PATH)


def build_features(df: pd.DataFrame, encoder: OneHotEncoder | None = None):
    """One-hot encodes `position` and concatenates it with the numeric
    features. If `encoder` is None, a new one is fit on this data (use only
    on the training split); otherwise the given fitted encoder is reused
    (use for the test split, and later for the interface step).
    """
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        position_encoded = encoder.fit_transform(df[["position"]])
    else:
        position_encoded = encoder.transform(df[["position"]])

    position_cols = encoder.get_feature_names_out(["position"])
    position_df = pd.DataFrame(position_encoded, columns=position_cols, index=df.index)

    X = pd.concat([df[NUMERIC_FEATURES], position_df], axis=1)
    return X, encoder


def euro_metrics(actual_euros: np.ndarray, predicted_log: np.ndarray, model_name: str) -> dict:
    predicted_euros = np.expm1(predicted_log)

    negative_mask = predicted_euros < 0
    if negative_mask.any():
        print(
            f"  WARNING [{model_name}]: {negative_mask.sum()} prediction(s) came back as a negative "
            f"transfer value after expm1 (impossible in reality) - e.g. {predicted_euros[negative_mask][:3]}. "
            "This signals the model is extrapolating badly for those rows, not a code bug."
        )

    return {
        "rmse_euros": np.sqrt(mean_squared_error(actual_euros, predicted_euros)),
        "mae_euros": mean_absolute_error(actual_euros, predicted_euros),
        "predicted_euros_array": predicted_euros,
    }


def main():
    df = load_dataset()
    print(f"Loaded {len(df)} rows from {DATASET_PATH}\n")

    X_all, _ = build_features(df)
    y_all = df["log_transfer_value"]

    # Split indices so we can recover names/actual euros for the test rows later.
    train_idx, test_idx = train_test_split(df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df, test_df = df.loc[train_idx], df.loc[test_idx]

    print(f"Train/test split: {len(train_df)} train / {len(test_df)} test (test_size={TEST_SIZE}, random_state={RANDOM_STATE})")
    if len(test_df) < MIN_MEANINGFUL_TEST_ROWS:
        print(
            f"  CAVEAT: only {len(test_df)} rows in the test set (from {len(df)} total). Metrics below are "
            "based on a small sample and can swing a lot with a different random_state - treat them as "
            "directional, not precise, until more of the season's data is available.\n"
        )
    else:
        print()

    # Fit the encoder on the TRAIN split only (standard practice - avoids
    # leaking test-set category frequencies into training) and reuse the
    # same fitted encoder to transform the test split.
    X_train, encoder = build_features(train_df, encoder=None)
    X_test, _ = build_features(test_df, encoder=encoder)
    y_train = train_df["log_transfer_value"]
    y_test = test_df["log_transfer_value"]
    actual_euros_test = test_df["transfer_value"].to_numpy()

    print(f"Features used: {list(X_train.columns)}\n")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_log = model.predict(X_test)

        log_rmse = np.sqrt(mean_squared_error(y_test, pred_log))
        r2_log = r2_score(y_test, pred_log)
        euro = euro_metrics(actual_euros_test, pred_log, name)

        results[name] = {
            "model": model,
            "pred_log": pred_log,
            "r2_log": r2_log,
            "log_rmse": log_rmse,
            "rmse_euros": euro["rmse_euros"],
            "mae_euros": euro["mae_euros"],
            "predicted_euros": pd.Series(euro["predicted_euros_array"], index=test_df.index),
        }

    # --- Comparison table ---
    print("=" * 78)
    print("MODEL COMPARISON (test set)")
    print("=" * 78)
    header = f"{'Metric':<28}{'Linear Regression':>24}{'Random Forest':>24}"
    print(header)
    print("-" * 78)
    rows = [
        ("R^2 (log-space, fit objective)", "r2_log", "{:.3f}"),
        ("RMSE (euros)", "rmse_euros", "EUR{:,.0f}"),
        ("MAE (euros)", "mae_euros", "EUR{:,.0f}"),
        ("RMSE (log-space, reference)", "log_rmse", "{:.3f}"),
    ]
    for label, key, fmt in rows:
        lr_val = fmt.format(results["Linear Regression"][key])
        rf_val = fmt.format(results["Random Forest"][key])
        print(f"{label:<28}{lr_val:>24}{rf_val:>24}")
    print(
        "\nNote: R^2 is reported in log-space because that's the scale both models were actually fit "
        "on (log_transfer_value) - R^2 computed after an expm1 back-transform isn't a coherent 'variance "
        "explained' statistic and can look misleadingly bad or good depending on a few large residuals."
    )

    # --- Feature importances (Random Forest) ---
    print("\n" + "=" * 78)
    print("RANDOM FOREST FEATURE IMPORTANCES")
    print("=" * 78)
    rf_model = results["Random Forest"]["model"]
    importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"  {feat:<20} {imp:.3f}")
    print(
        "\nWith 87.7% of players at 0 goals and 0 assists (only ~20 matches played this season), "
        "expect age and position to currently dominate over goals/assists - there just isn't enough "
        "goal-scoring variance yet for the model to learn much from it."
    )

    # --- Sample predictions ---
    print("\n" + "=" * 78)
    print("SAMPLE PREDICTIONS (10 test-set players, real euros)")
    print("=" * 78)
    sample_n = min(10, len(test_df))
    sample_df = test_df.sample(n=sample_n, random_state=RANDOM_STATE)
    lr_pred = results["Linear Regression"]["predicted_euros"]
    rf_pred = results["Random Forest"]["predicted_euros"]

    print(f"{'Name':<25}{'Actual':>15}{'LR Predicted':>18}{'RF Predicted':>18}")
    for idx, row in sample_df.iterrows():
        print(
            f"{row['name']:<25}{'EUR{:,.0f}'.format(row['transfer_value']):>15}"
            f"{'EUR{:,.0f}'.format(lr_pred.loc[idx]):>18}{'EUR{:,.0f}'.format(rf_pred.loc[idx]):>18}"
        )

    # --- Verdict ---
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    better = "Random Forest" if results["Random Forest"]["r2_log"] > results["Linear Regression"]["r2_log"] else "Linear Regression"
    print(f"Better R^2 (log-space): {better}")
    print(
        f"  Linear Regression R^2={results['Linear Regression']['r2_log']:.3f}, "
        f"Random Forest R^2={results['Random Forest']['r2_log']:.3f}"
    )
    print(
        "\nCaveats:\n"
        f"  - Test set is only {len(test_df)} rows out of {len(df)} total - small enough that these metrics "
        "could shift noticeably with a different random_state or a few more weeks of data.\n"
        "  - 87.7% of players have 0 goals and 0 assists this early in the season, so goals/assists "
        "currently carry little real signal; both models are mostly learning an age/position-based prior "
        "on transfer value right now, not a performance-based one.\n"
        "  - Random Forest with only 3 numeric + a handful of one-hot features and ~240 training rows "
        "has limited room to actually outperform a linear model - if it wins here, treat the margin as "
        "small and revisit once more of the season's stats are in."
    )

    # --- Persist artifacts for the future "predict any player" interface ---
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, ARTIFACTS_DIR / "position_encoder.pkl")
    joblib.dump(results["Linear Regression"]["model"], ARTIFACTS_DIR / "linear_regression.pkl")
    joblib.dump(results["Random Forest"]["model"], ARTIFACTS_DIR / "random_forest.pkl")
    joblib.dump(list(X_train.columns), ARTIFACTS_DIR / "feature_columns.pkl")
    print(f"\nSaved fitted encoder + both models to {ARTIFACTS_DIR} for reuse in the next (interface) step.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
