# tests/test_features.py

import pandas as pd
import numpy as np

from src.features import (
    prepare_data,
    create_calendar_features,
    create_lag_features,
    create_rolling_features,
    create_features,
    FEATURES,
    TARGET
)


# ---------------------------------------------------------
# Create sample data
# ---------------------------------------------------------

def create_sample_data():

    dates = pd.date_range(
        start="2025-01-01",
        periods=40,
        freq="D"
    )

    rows = []

    for store in ["S001", "S002"]:

        for product in ["P001", "P002"]:

            for i, date in enumerate(dates):

                rows.append({

                    "Date": date,

                    "Store ID": store,

                    "Product ID": product,

                    "Category": "Electronics",

                    "Region": "North",

                    "Inventory Level": 100,

                    "Units Sold": 20,

                    "Units Ordered": 25,

                    "Price": 100,

                    "Discount": 10,

                    "Weather Condition": "Sunny",

                    "Promotion": 1,

                    "Competitor Pricing": 95,

                    "Seasonality": "Normal",

                    "Epidemic": 0,

                    "Demand": 20 + i

                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Test 1 - Data preparation
# ---------------------------------------------------------

def test_prepare_data():

    df = create_sample_data()

    result = prepare_data(df)

    # Date should be datetime
    assert pd.api.types.is_datetime64_any_dtype(
        result["Date"]
    )

    # Required grouping columns should exist
    assert "Store ID" in result.columns
    assert "Product ID" in result.columns

    # Data should be sorted by Store, Product and Date
    assert result[
        ["Store ID", "Product ID", "Date"]
    ].equals(
        result[
            ["Store ID", "Product ID", "Date"]
        ].sort_values(
            ["Store ID", "Product ID", "Date"]
        ).reset_index(drop=True)
    )


# ---------------------------------------------------------
# Test 2 - Calendar features
# ---------------------------------------------------------

def test_calendar_features():

    df = create_sample_data()

    result = prepare_data(df)

    result = create_calendar_features(result)

    expected_columns = [
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "month",
        "quarter",
        "year",
        "is_weekend"
    ]

    for column in expected_columns:

        assert column in result.columns


# ---------------------------------------------------------
# Test 3 - Lag features
# ---------------------------------------------------------

def test_lag_features():

    df = create_sample_data()

    result = prepare_data(df)

    result = create_lag_features(result)

    expected_columns = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28"
    ]

    for column in expected_columns:

        assert column in result.columns

    # First row of each Store/Product group
    # must have no lag_1 value.
    first_rows = (
        result
        .groupby(
            ["Store ID", "Product ID"]
        )
        .head(1)
    )

    assert first_rows["lag_1"].isna().all()


# ---------------------------------------------------------
# Test 4 - Rolling features
# ---------------------------------------------------------

def test_rolling_features():

    df = create_sample_data()

    result = prepare_data(df)

    result = create_rolling_features(result)

    expected_columns = [
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_14"
    ]

    for column in expected_columns:

        assert column in result.columns


# ---------------------------------------------------------
# Test 5 - Complete feature pipeline
# ---------------------------------------------------------

def test_create_features():

    df = create_sample_data()

    result = create_features(df)

    # All model features should exist
    for feature in FEATURES:

        assert feature in result.columns

    # Target should exist
    assert TARGET in result.columns

    # Historical feature columns should not contain NaN
    historical_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_14"
    ]

    assert not result[
        historical_features
    ].isna().any().any()


# ---------------------------------------------------------
# Test 6 - No target leakage
# ---------------------------------------------------------

def test_no_target_leakage():

    df = create_sample_data()

    result = create_features(df)

    # Demand itself must not be part of FEATURES
    assert TARGET not in FEATURES

    # Units Sold is deliberately excluded because
    # it may represent realized same-day sales.
    assert "Units Sold" not in FEATURES


# ---------------------------------------------------------
# Test 7 - Categorical features
# ---------------------------------------------------------

def test_categorical_features():

    df = create_sample_data()

    result = create_features(df)

    categorical_columns = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality",
        "Epidemic"
    ]

    for column in categorical_columns:

        assert (
            str(result[column].dtype)
            == "category"
        )
