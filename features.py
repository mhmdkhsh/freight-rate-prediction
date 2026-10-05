
import numpy as np
import pandas as pd

CAT_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]

NUM_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight_clean",
    "weight_missing",
    "market_index",
    "quote_signal",
    "lat_diff",
    "lon_diff",
    "abs_lat_diff",
    "abs_lon_diff",
    "distance_squared",
    "distance_cubed",
    "log_distance",
    "sqrt_distance",
    "distance_0_500",
    "distance_500_1000",
    "distance_1000_1500",
    "distance_1500_2000",
    "distance_2000_3000",
    "distance_3000_plus",
    "distance_x_reefer",
    "distance_x_flatbed",
    "distance_x_dryvan",
    "distance_squared_x_reefer",
    "distance_squared_x_flatbed",
    "weight_x_distance",
    "weight_missing_x_distance",
    "market_x_distance",
    "quote_x_distance",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "days_since_start",
]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


def make_features(
    df,
    train_medians=None,
    city_coords=None,
    start_date=None,
):
    """
    Create the final V4 feature set.

    The function is designed so the same feature engineering
    can be applied to training, validation and December data.
    """

    x = df.copy()

    # --------------------------------------------------------
    # Basic categorical features
    # --------------------------------------------------------

    for col in ["pickup", "delivery", "equipment"]:
        if col not in x.columns:
            x[col] = "Unknown"

        x[col] = x[col].fillna("Unknown").astype(str)

    x["route"] = x["pickup"] + "__" + x["delivery"]

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    if city_coords is not None:

        if "pickup_lat" not in x.columns:
            x["pickup_lat"] = np.nan

        if "pickup_lon" not in x.columns:
            x["pickup_lon"] = np.nan

        if "delivery_lat" not in x.columns:
            x["delivery_lat"] = np.nan

        if "delivery_lon" not in x.columns:
            x["delivery_lon"] = np.nan

        pickup_lat = x["pickup"].map(
            lambda v: city_coords.get(v, (np.nan, np.nan))[0]
        )
        pickup_lon = x["pickup"].map(
            lambda v: city_coords.get(v, (np.nan, np.nan))[1]
        )
        delivery_lat = x["delivery"].map(
            lambda v: city_coords.get(v, (np.nan, np.nan))[0]
        )
        delivery_lon = x["delivery"].map(
            lambda v: city_coords.get(v, (np.nan, np.nan))[1]
        )

        x["pickup_lat"] = x["pickup_lat"].fillna(pickup_lat)
        x["pickup_lon"] = x["pickup_lon"].fillna(pickup_lon)
        x["delivery_lat"] = x["delivery_lat"].fillna(delivery_lat)
        x["delivery_lon"] = x["delivery_lon"].fillna(delivery_lon)

    # --------------------------------------------------------
    # Numeric base variables
    # --------------------------------------------------------

    for col in [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
    ]:
        if col not in x.columns:
            x[col] = np.nan

    # Negative weights are treated as anomalous observations.
    # We keep the observations but clean the feature used by
    # the model.
    x["weight_missing"] = x["weight"].isna().astype(int)
    x["weight_clean"] = x["weight"].clip(lower=0)

    # --------------------------------------------------------
    # Geographic features
    # --------------------------------------------------------

    x["lat_diff"] = x["delivery_lat"] - x["pickup_lat"]
    x["lon_diff"] = x["delivery_lon"] - x["pickup_lon"]

    x["abs_lat_diff"] = x["lat_diff"].abs()
    x["abs_lon_diff"] = x["lon_diff"].abs()

    # --------------------------------------------------------
    # Nonlinear distance features
    # --------------------------------------------------------

    d = x["distance"].clip(lower=0)

    x["distance_squared"] = d ** 2
    x["distance_cubed"] = d ** 3
    x["log_distance"] = np.log1p(d)
    x["sqrt_distance"] = np.sqrt(d)

    x["distance_0_500"] = (d <= 500).astype(int)
    x["distance_500_1000"] = ((d > 500) & (d <= 1000)).astype(int)
    x["distance_1000_1500"] = ((d > 1000) & (d <= 1500)).astype(int)
    x["distance_1500_2000"] = ((d > 1500) & (d <= 2000)).astype(int)
    x["distance_2000_3000"] = ((d > 2000) & (d <= 3000)).astype(int)
    x["distance_3000_plus"] = (d > 3000).astype(int)

    # --------------------------------------------------------
    # Equipment interactions
    # --------------------------------------------------------

    eq = x["equipment"].str.lower()

    reefer = eq.eq("reefer").astype(int)
    flatbed = eq.eq("flatbed").astype(int)
    dryvan = eq.eq("dry van").astype(int)

    x["distance_x_reefer"] = d * reefer
    x["distance_x_flatbed"] = d * flatbed
    x["distance_x_dryvan"] = d * dryvan

    x["distance_squared_x_reefer"] = x["distance_squared"] * reefer
    x["distance_squared_x_flatbed"] = x["distance_squared"] * flatbed

    # --------------------------------------------------------
    # Other interactions
    # --------------------------------------------------------

    x["weight_x_distance"] = x["weight_clean"] * d
    x["weight_missing_x_distance"] = x["weight_missing"] * d

    x["market_x_distance"] = x["market_index"] * d
    x["quote_x_distance"] = x["quote_signal"] * d

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

    date = pd.to_datetime(x["date"])

    x["year"] = date.dt.year
    x["month"] = date.dt.month
    x["day"] = date.dt.day
    x["day_of_week"] = date.dt.dayofweek
    x["day_of_year"] = date.dt.dayofyear
    x["week_of_year"] = date.dt.isocalendar().week.astype(int)

    if start_date is None:
        start_date = date.min()

    start_date = pd.Timestamp(start_date)
    x["days_since_start"] = (date - start_date).dt.days

    # --------------------------------------------------------
    # Missing-value handling
    # --------------------------------------------------------

    if train_medians is None:
        train_medians = {}

    for col in NUM_FEATURES:
        if col not in x.columns:
            x[col] = np.nan

        if col in train_medians:
            median_value = train_medians[col]
        else:
            median_value = x[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        x[col] = x[col].replace([np.inf, -np.inf], np.nan)
        x[col] = x[col].fillna(median_value)

    # CatBoost categorical columns must remain strings.
    for col in CAT_FEATURES:
        x[col] = x[col].fillna("Unknown").astype(str)

    return x[ALL_FEATURES]
