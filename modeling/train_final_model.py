"""Train the final Linear Regression model on ALL 302 rows for the prediction interface.

Model choice (Linear Regression over Random Forest) was finalized after the
honest train/test comparison in train_models.py - see that script's output
for the R^2/RMSE/MAE comparison this decision was based on. Random Forest's
code and artifacts are left in place but are not used from here on.

Reuses the position_encoder.pkl fit during that train/test split - it is
NOT refit here, so the categories/column order predict.py relies on stay
identical to what the model was evaluated against.
"""
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression

from train_models import DATASET_PATH, build_features, load_dataset

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ENCODER_PATH = ARTIFACTS_DIR / "position_encoder.pkl"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "linear_model.pkl"


def main():
    if not ENCODER_PATH.exists():
        print(
            f"ERROR: {ENCODER_PATH} not found. Run train_models.py first - it fits and saves "
            "the position encoder this script reuses.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = load_dataset()
    encoder = joblib.load(ENCODER_PATH)

    X_all, _ = build_features(df, encoder=encoder)  # reuse, do not refit
    y_all = df["log_transfer_value"]

    model = LinearRegression()
    model.fit(X_all, y_all)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, FINAL_MODEL_PATH)
    joblib.dump(list(X_all.columns), ARTIFACTS_DIR / "feature_columns.pkl")

    print(f"Trained Linear Regression on all {len(df)} rows from {DATASET_PATH}")
    print(f"Features: {list(X_all.columns)}")
    print(f"Saved to {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
