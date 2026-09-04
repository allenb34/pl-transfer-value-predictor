"""Standalone visualization: predicted vs actual transfer value.

Read-only with respect to the rest of the project - loads the already
-trained linear_model.pkl and position_encoder.pkl and player_dataset.csv,
generates predictions, and writes a plot + prints a summary. Doesn't retrain
anything or touch any other file.
"""
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: this script saves a file, never opens a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_models import DATASET_PATH, build_features

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "linear_model.pkl"
ENCODER_PATH = ARTIFACTS_DIR / "position_encoder.pkl"
OUTPUT_PATH = ARTIFACTS_DIR / "predicted_vs_actual.png"

# Highlighted for continuity with earlier discussion (predict.py test run) -
# purely cosmetic, skipped cleanly if a name isn't found.
HIGHLIGHT_NAMES = ["Erling Haaland", "Richarlison", "Adam Smith"]

WITHIN_PCT = 0.20  # "reasonably accurate": predicted within +/-20% of actual
BADLY_WRONG_MULTIPLE = 2.0  # "badly wrong": off by more than 2x in either direction


def main():
    for path in (DATASET_PATH, MODEL_PATH, ENCODER_PATH):
        if not path.exists():
            print(f"ERROR: {path} does not exist. Run merge_dataset.py, train_models.py, and "
                  "train_final_model.py first.", file=sys.stderr)
            sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    X, _ = build_features(df, encoder=encoder)
    pred_log = model.predict(X)
    predicted = np.expm1(pred_log)
    actual = df["transfer_value"].to_numpy()

    # --- plot ---
    fig, ax = plt.subplots(figsize=(9, 8))

    highlight_mask = df["name"].isin(HIGHLIGHT_NAMES)
    ax.scatter(
        actual[~highlight_mask], predicted[~highlight_mask],
        alpha=0.5, s=35, color="#3b7dd8", edgecolors="none", label=f"Players (n={(~highlight_mask).sum()})",
    )

    found_highlights = df.loc[highlight_mask, "name"].tolist()
    missing_highlights = [n for n in HIGHLIGHT_NAMES if n not in found_highlights]
    if missing_highlights:
        print(f"Note: could not find {missing_highlights} in player_dataset.csv - skipping their highlight.")

    if highlight_mask.any():
        ax.scatter(
            actual[highlight_mask], predicted[highlight_mask],
            s=140, color="#e2542a", edgecolors="black", linewidths=1.2, zorder=5,
            label="Highlighted players",
        )
        for name, x, y in zip(df.loc[highlight_mask, "name"], actual[highlight_mask], predicted[highlight_mask]):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 8), fontsize=9, fontweight="bold")

    lo = min(actual.min(), predicted.min()) * 0.7
    hi = max(actual.max(), predicted.max()) * 1.3
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.5, label="Perfect prediction (y = x)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual Transfer Value (EUR, log scale)", fontsize=11)
    ax.set_ylabel("Predicted Transfer Value (EUR, log scale)", fontsize=11)
    ax.set_title(f"Linear Regression: Predicted vs Actual Transfer Value (n={len(df)})", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {OUTPUT_PATH}")

    # --- text summary ---
    ratio = predicted / actual
    within_pct = np.abs(ratio - 1) <= WITHIN_PCT
    badly_wrong = (ratio > BADLY_WRONG_MULTIPLE) | (ratio < 1 / BADLY_WRONG_MULTIPLE)

    n = len(df)
    print(f"\n{'=' * 60}")
    print("PREDICTED VS ACTUAL - ACCURACY SUMMARY")
    print("=" * 60)
    print(f"  Total players:                          {n}")
    print(f"  Within +/-{int(WITHIN_PCT * 100)}% of actual (reasonably accurate): {within_pct.sum()} ({100 * within_pct.sum() / n:.1f}%)")
    print(f"  Off by more than {BADLY_WRONG_MULTIPLE:.0f}x (badly wrong):        {badly_wrong.sum()} ({100 * badly_wrong.sum() / n:.1f}%)")
    print(f"  Everything in between:                  {n - within_pct.sum() - badly_wrong.sum()} "
          f"({100 * (n - within_pct.sum() - badly_wrong.sum()) / n:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
