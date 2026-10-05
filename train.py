
from pathlib import Path
import json

import pandas as pd
import numpy as np

from catboost import CatBoostRegressor, Pool

from features import make_features, CAT_FEATURES, NUM_FEATURES


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def build_city_coordinates(df):
    coords = {}

    for city in pd.concat(
        [df["pickup"], df["delivery"]]
    ).dropna().unique():

        pickup_rows = df[df["pickup"] == city]

        if len(pickup_rows):
            row = pickup_rows.iloc[0]
            if pd.notna(row["pickup_lat"]) and pd.notna(row["pickup_lon"]):
                coords[city] = (
                    float(row["pickup_lat"]),
                    float(row["pickup_lon"])
                )
                continue

        delivery_rows = df[df["delivery"] == city]

        if len(delivery_rows):
            row = delivery_rows.iloc[0]
            if pd.notna(row["delivery_lat"]) and pd.notna(row["delivery_lon"]):
                coords[city] = (
                    float(row["delivery_lat"]),
                    float(row["delivery_lon"])
                )

    return coords


def main():

    train_path = DATA / "train_test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            "Place the assessment train_test.csv inside data/."
        )

    df = pd.read_csv(train_path)

    train_medians = {}

    for col in NUM_FEATURES:
        if col in df.columns:
            train_medians[col] = df[col].median()

    city_coords = build_city_coordinates(df)

    X = make_features(
        df,
        train_medians=train_medians,
        city_coords=city_coords,
        start_date=df["date"].min(),
    )

    y = df["posted_rate"]

    model = CatBoostRegressor(
        iterations=220,
        learning_rate=0.045,
        depth=8,
        l2_leaf_reg=6,
        loss_function="RMSE",
        random_seed=42,
        random_strength=1,
        bagging_temperature=1,
        thread_count=4,
        verbose=False,
    )

    pool = Pool(
        X,
        y,
        cat_features=CAT_FEATURES,
    )

    model.fit(pool)

    model.save_model(
        MODEL_DIR / "freight_rate_v4.cbm"
    )

    metadata = {
        "model": "CatBoostRegressor",
        "iterations": 220,
        "learning_rate": 0.045,
        "depth": 8,
        "l2_leaf_reg": 6,
        "loss_function": "RMSE",
        "random_seed": 42,
        "random_strength": 1,
        "bagging_temperature": 1,
        "categorical_features": CAT_FEATURES,
        "numeric_features": NUM_FEATURES,
        "city_coordinates": city_coords,
        "train_medians": train_medians,
        "start_date": str(df["date"].min()),
    }

    with open(
        MODEL_DIR / "feature_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, default=str)

    print("Final model saved.")


if __name__ == "__main__":
    main()
