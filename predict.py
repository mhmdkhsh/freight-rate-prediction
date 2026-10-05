
from pathlib import Path
import json

import pandas as pd
from catboost import CatBoostRegressor, Pool

from features import make_features, CAT_FEATURES


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
MODEL_DIR = BASE / "models"


def main():

    model_path = MODEL_DIR / "freight_rate_v4.cbm"
    metadata_path = MODEL_DIR / "feature_metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = CatBoostRegressor()
    model.load_model(model_path)

    train_medians = metadata["train_medians"]

    city_coords = {
        k: tuple(v)
        for k, v in metadata["city_coordinates"].items()
    }

    start_date = metadata["start_date"]

    validation = pd.read_csv(
        DATA / "validation.csv"
    )

    X = make_features(
        validation,
        train_medians=train_medians,
        city_coords=city_coords,
        start_date=start_date,
    )

    preds = model.predict(
        Pool(X, cat_features=CAT_FEATURES)
    )

    output = pd.DataFrame({
        "load_id": validation["load_id"],
        "predicted_rate": preds,
    })

    output.to_csv(
        BASE / "validation_predictions.csv",
        index=False,
    )

    december = pd.read_csv(
        DATA / "december_chart_inputs.csv"
    )

    X_dec = make_features(
        december,
        train_medians=train_medians,
        city_coords=city_coords,
        start_date=start_date,
    )

    dec_preds = model.predict(
        Pool(X_dec, cat_features=CAT_FEATURES)
    )

    december["predicted_rate"] = dec_preds

    december.to_csv(
        DATA / "december_chart_inputs.csv",
        index=False,
    )

    print("Validation predictions saved.")
    print("December predictions saved.")


if __name__ == "__main__":
    main()
