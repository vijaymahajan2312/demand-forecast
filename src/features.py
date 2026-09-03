# src/features.py

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GROUP_COLS = [
    "Store ID",
    "Product ID"
]

TARGET = "Demand"


CATEGORICAL_FEATURES = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
    "Epidemic"
]


BUSINESS_FEATURES = [
    "Inventory Level",
    "Units Ordered",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing"
]


TIME_FEATURES = [
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend"
]


LAG_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28"
]


ROLLING_FEATURES = [
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14"
]


FEATURES = (
    LAG_FEATURES
    + ROLLING_FEATURES
    + BUSINESS_FEATURES
    + TIME_FEATURES
    + CATEGORICAL_FEATURES
)


# ---------------------------------------------------------
# Basic preprocessing
# ---------------------------------------------------------

def prepare_data(df):
    """
    Prepare raw demand forecasting data.

    Steps:
    1. Convert Date to datetime
    2. Sort by Store, Product and Date
    3. Reset index
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df
        .sort_values(
            GROUP_COLS + ["Date"]
        )
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------
# Calendar features
# ---------------------------------------------------------

def create_calendar_features(df):
    """
    Create calendar/time-based features.
    """

    df = df.copy()

    df["day_of_week"] = df["Date"].dt.dayofweek

    df["day_of_month"] = df["Date"].dt.day

    df["week_of_year"] = (
        df["Date"].dt.isocalendar().week.astype(int)
    )

    df["month"] = df["Date"].dt.month

    df["quarter"] = df["Date"].dt.quarter

    df["year"] = df["Date"].dt.year

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


# ---------------------------------------------------------
# Lag features
# ---------------------------------------------------------

def create_lag_features(df):
    """
    Create historical demand lag features.

    Lags:
    lag_1
    lag_7
    lag_14
    lag_28
    """

    df = df.copy()

    for lag in [1, 7, 14, 28]:

        df[f"lag_{lag}"] = (
            df
            .groupby(GROUP_COLS)[TARGET]
            .shift(lag)
        )

    return df


# ---------------------------------------------------------
# Rolling features
# ---------------------------------------------------------

def create_rolling_features(df):
    """
    Create rolling demand statistics.

    shift(1) is used before rolling
    to prevent target leakage.
    """

    df = df.copy()

    grouped_demand = (
        df
        .groupby(GROUP_COLS)[TARGET]
    )

    df["rolling_mean_7"] = (
        grouped_demand
        .transform(
            lambda x:
            x.shift(1)
             .rolling(7)
             .mean()
        )
    )

    df["rolling_mean_14"] = (
        grouped_demand
        .transform(
            lambda x:
            x.shift(1)
             .rolling(14)
             .mean()
        )
    )

    df["rolling_mean_28"] = (
        grouped_demand
        .transform(
            lambda x:
            x.shift(1)
             .rolling(28)
             .mean()
        )
    )

    df["rolling_std_7"] = (
        grouped_demand
        .transform(
            lambda x:
            x.shift(1)
             .rolling(7)
             .std()
        )
    )

    df["rolling_std_14"] = (
        grouped_demand
        .transform(
            lambda x:
            x.shift(1)
             .rolling(14)
             .std()
        )
    )

    return df


# ---------------------------------------------------------
# Complete feature engineering pipeline
# ---------------------------------------------------------

def create_features(df):
    """
    Complete feature engineering pipeline.
    """

    df = prepare_data(df)

    df = create_calendar_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    # Remove rows where historical features
    # cannot be calculated.
    df = df.dropna(
        subset=LAG_FEATURES + ROLLING_FEATURES
    ).reset_index(drop=True)

    # Convert categorical columns
    # to pandas category dtype.
    for col in CATEGORICAL_FEATURES:

        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------
# Model dataset
# ---------------------------------------------------------

def get_model_data(df):
    """
    Create X and y for model training.
    """

    X = df[FEATURES].copy()

    y = df[TARGET].copy()

    return X, y
