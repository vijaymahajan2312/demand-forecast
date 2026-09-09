# test/test_features.py

import numpy as np
import pandas as pd

from src.features import (
    FEATURES,
    TARGET,
    create_calendar_features,
    create_features,
    create_lag_features,
    create_rolling_features,
    prepare_data,
)


def create_sample_data():
    """Create sample demand data for testing feature engineering."""

    dates = pd.date_range(
        start="2025-01-01",
        periods=60,
        freq="D",
    )

    rows = []

    stores = ["Store_1", "Store_2"]
    products = ["Product_1", "Product_2"]

    for store in stores:
        for product in products:
            for i, date in enumerate(dates):
                rows.append(
                    {
                        "Date": date,
                        "Store ID": store,
                        "Product ID": product,
                        "Category": "Category_A",
                        "Region": "Region_A",
                        "Inventory Level": 100 + i,
                        "Units Sold": 20 + i,
                        "Units Ordered": 25 + i,
                        "Price": 50.0,
                        "Discount": 0.10,
                        "Weather Condition": "Sunny",
                        "Promotion": 0,
                        "Competitor Pricing": 48.0,
                        "Seasonality": "Regular",
                        "Epidemic": 0,
                        "Demand": 30 + i,
                    }
                )

    return pd.DataFrame(rows)


def test_prepare_data():
    """Test date conversion and group-wise sorting."""

    df = create_sample_data()

    result = prepare_data(df)

    # Date should be converted to datetime.
    assert pd.api.types.is_datetime64_any_dtype(
        result["Date"]
    )

    # Data should be sorted by Store ID, Product ID, and Date.
    assert result[
        ["Store ID", "Product ID", "Date"]
    ].equals(
        result[
            ["Store ID", "Product ID", "Date"]
        ].sort_values(
            ["Store ID", "Product ID", "Date"]
        ).reset_index(drop=True)
    )

    # Date should be increasing within every Store × Product group.
    for _, group in result.groupby(
        ["Store ID", "Product ID"]
    ):
        assert group["Date"].is_monotonic_increasing

    # Number of rows and columns should remain unchanged.
    assert result.shape == df.shape


def test_calendar_features():
    """Test calendar feature creation."""

    df = create_sample_data()

    result = create_calendar_features(df)

    expected_features = [
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "month",
        "quarter",
        "year",
        "is_weekend",
    ]

    for feature in expected_features:
        assert feature in result.columns

    assert result["day_of_week"].between(0, 6).all()

    assert result["month"].between(1, 12).all()

    assert result["quarter"].between(1, 4).all()

    assert result["is_weekend"].isin([0, 1]).all()


def test_lag_features():
    """Test lag feature creation."""

    df = create_sample_data()

    df = prepare_data(df)

    result = create_lag_features(df)

    expected_lags = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
    ]

    for feature in expected_lags:
        assert feature in result.columns

    store_product = result[
        (result["Store ID"] == "Store_1")
        & (result["Product ID"] == "Product_1")
    ].reset_index(drop=True)

    # lag_1 should contain the previous day's demand.
    assert pd.isna(store_product.loc[0, "lag_1"])

    assert (
        store_product.loc[1, "lag_1"]
        == store_product.loc[0, "Demand"]
    )

    # lag_7 should contain demand from 7 days earlier.
    assert pd.isna(store_product.loc[6, "lag_7"])

    assert (
        store_product.loc[7, "lag_7"]
        == store_product.loc[0, "Demand"]
    )


def test_rolling_features():
    """Test rolling mean and standard deviation features."""

    df = create_sample_data()

    df = prepare_data(df)

    result = create_rolling_features(df)

    expected_features = [
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_14",
    ]

    for feature in expected_features:
        assert feature in result.columns

    store_product = result[
        (result["Store ID"] == "Store_1")
        & (result["Product ID"] == "Product_1")
    ].reset_index(drop=True)

    # The first 7 rows should not have a 7-day rolling value
    # because the current day's demand is excluded.
    assert pd.isna(
        store_product.loc[6, "rolling_mean_7"]
    )

    # At index 7, rolling_mean_7 should use demand
    # from indices 0 through 6.
    expected_mean = store_product.loc[
        0:6, "Demand"
    ].mean()

    assert np.isclose(
        store_product.loc[7, "rolling_mean_7"],
        expected_mean,
    )


def test_create_features():
    """Test complete feature engineering pipeline."""

    df = create_sample_data()

    result = create_features(df)

    # All expected model features should exist.
    for feature in FEATURES:
        assert feature in result.columns

    # Target should still exist.
    assert TARGET in result.columns

    # Rows with insufficient lag/rolling history
    # should have been removed.
    assert len(result) < len(df)

    lag_and_rolling_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_14",
    ]

    # Required lag and rolling features should not contain NaN.
    assert not result[
        lag_and_rolling_features
    ].isna().any().any()


def test_no_target_leakage():
    """Verify that current Demand is not used as a feature."""

    df = create_sample_data()

    result = create_features(df)

    # Demand itself must not be part of FEATURES.
    assert TARGET not in FEATURES

    # Current Demand should not be used as a model feature.
    assert TARGET not in result[FEATURES].columns

    # Check that lag_1 refers to the previous observation,
    # not the current Demand.
    store_product = result[
        (result["Store ID"] == "Store_1")
        & (result["Product ID"] == "Product_1")
    ].reset_index(drop=True)

    assert (
        store_product.loc[1, "lag_1"]
        == store_product.loc[0, "Demand"]
    )

    assert (
        store_product.loc[1, "lag_1"]
        != store_product.loc[1, "Demand"]
    )


def test_categorical_features():
    """Test categorical feature data types."""

    df = create_sample_data()

    result = create_features(df)

    categorical_features = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality",
        "Epidemic",
    ]

    for feature in categorical_features:
        assert feature in result.columns

        assert isinstance(
            result[feature].dtype,
            pd.CategoricalDtype,
        )
