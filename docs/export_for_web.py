"""One-off export: turns the trained Linear Regression model + player_dataset.csv
into plain JSON that docs/index.html can consume with zero Python at runtime.

Standalone - reads existing pipeline outputs (modeling/artifacts/*.pkl,
data/processed/player_dataset.csv) but doesn't modify or retrain anything.
Re-run this whenever the model or dataset changes and the site needs updating.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "modeling" / "artifacts"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "player_dataset.csv"
DOCS_DIR = Path(__file__).resolve().parent

MODEL_PATH = ARTIFACTS_DIR / "linear_model.pkl"
ENCODER_PATH = ARTIFACTS_DIR / "position_encoder.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.pkl"

TEMPLATE_PATH = DOCS_DIR / "index_template.html"
INDEX_PATH = DOCS_DIR / "index.html"


def export_model_params():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    dataset = pd.read_csv(DATASET_PATH)
    max_trained_goals = int(dataset["goals"].max())
    max_trained_assists = int(dataset["assists"].max())

    params = {
        "feature_order": list(feature_columns),
        "coefficients": [float(c) for c in model.coef_],
        "intercept": float(model.intercept_),
        "position_categories": list(encoder.categories_[0]),
        "training_ranges": {
            "min_age": 15,
            "max_age": 45,
            "max_goals": max_trained_goals,
            "max_assists": max_trained_assists,
        },
    }

    assert len(params["feature_order"]) == len(params["coefficients"]), (
        "feature_order and coefficients must be the same length - the JS dot product "
        "depends on positional alignment between them"
    )

    out_path = DOCS_DIR / "model_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"  feature_order: {params['feature_order']}")
    print(f"  intercept: {params['intercept']}")
    print(f"  position_categories: {params['position_categories']}")
    print(f"  training_ranges: {params['training_ranges']}")
    return params


def export_players():
    df = pd.read_csv(DATASET_PATH)
    columns = ["name", "club", "position", "age", "goals", "assists", "transfer_value"]
    players = df[columns].to_dict(orient="records")

    out_path = DOCS_DIR / "players.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2)
    print(f"Wrote {out_path} ({len(players)} players)")
    return players


def build_index_html(params: dict, players: list):
    """index.html embeds the model params + player list directly as JS
    literals rather than fetch()-ing model_params.json/players.json at
    runtime - fetch() of local files is blocked by CORS when a page is
    opened via file:// (no server), which is a hard requirement here. The
    .json files are still written above as plain, inspectable/reusable
    exports; index.html just doesn't load its data from them.
    """
    if not TEMPLATE_PATH.exists():
        print(f"ERROR: {TEMPLATE_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__MODEL_PARAMS_JSON__", json.dumps(params))
    html = html.replace("__PLAYERS_JSON__", json.dumps(players))

    if "__MODEL_PARAMS_JSON__" in html or "__PLAYERS_JSON__" in html:
        print("ERROR: template placeholder(s) were not fully substituted.", file=sys.stderr)
        sys.exit(1)

    INDEX_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {INDEX_PATH}")


def main():
    for path in (MODEL_PATH, ENCODER_PATH, FEATURE_COLUMNS_PATH, DATASET_PATH):
        if not path.exists():
            print(f"ERROR: {path} does not exist. Run the modeling pipeline first.", file=sys.stderr)
            sys.exit(1)

    params = export_model_params()
    players = export_players()
    build_index_html(params, players)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
