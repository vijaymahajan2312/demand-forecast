
# src/features.py

import pandas as pd

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


# ---------------------------------------------------------
# Complete model feature list
# ---------------------------------------------------------

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
    1. Copy input dataframe
    2. Convert Date to datetime
    3. Sort by Store, Product and Date
    4. Reset index
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
        df["Date"]
        .dt.isocalendar()
        .week
        .astype(int)
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
    - lag_1
    - lag_7
    - lag_14
    - lag_28
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

    # Rolling mean - 7 days
    df["rolling_mean_7"] = (
        grouped_demand
        .transform(
            lambda x:
            x.shift(1)
             .rolling(7)
             .mean()
        )
    )

    # Rolling mean - 14 days
    df["rolling_mean_14"] = (
        grouped_demand
        .transform(
            lambda x:
            x.shift(1)
             .rolling(14)
             .mean()
        )
    )

    # Rolling mean - 28 days
    df["rolling_mean_28"] = (
        grouped_demand
        .transform(
            lambda x:
            x.shift(1)
             .rolling(28)
             .mean()
        )
    )

    # Rolling standard deviation - 7 days
    df["rolling_std_7"] = (
        grouped_demand
        .transform(
            lambda x:
            x.shift(1)
             .rolling(7)
             .std()
        )
    )

    # Rolling standard deviation - 14 days
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

    Steps:
    1. Prepare raw data
    2. Create calendar features
    3. Create lag features
    4. Create rolling features
    5. Remove rows with insufficient history
    6. Convert categorical columns to pandas category dtype
    """

    df = prepare_data(df)

    df = create_calendar_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    # -----------------------------------------------------
    # Remove rows where historical features
    # cannot be calculated.
    # -----------------------------------------------------

    df = df.dropna(
        subset=LAG_FEATURES + ROLLING_FEATURES
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Convert categorical columns
    # to pandas category dtype.
    # -----------------------------------------------------

    for col in CATEGORICAL_FEATURES:

        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------
# Feature name cleaning
# ---------------------------------------------------------

def clean_feature_names(X):
    """
    Clean model feature names.

    Example:
        Store ID -> Store_ID
        Weather Condition -> Weather_Condition

    This ensures that training and inference use
    consistent feature names.
    """

    X = X.copy()

    rename_map = {
        col: col.strip().replace(" ", "_")
        for col in X.columns
    }

    X = X.rename(
        columns=rename_map
    )

    return X


# ---------------------------------------------------------
# Categorical feature preparation
# ---------------------------------------------------------

def prepare_categorical_features(
    X_train,
    X_valid=None,
    X_test=None
):
    """
    Prepare categorical columns consistently.

    All categorical columns are converted to string
    first and then assigned the same category set.

    This prevents differences between train,
    validation and test datasets.
    """

    X_train = X_train.copy()

    if X_valid is not None:
        X_valid = X_valid.copy()

    if X_test is not None:
        X_test = X_test.copy()

    categorical_columns = [
        col.strip().replace(" ", "_")
        for col in CATEGORICAL_FEATURES
    ]

    for col in categorical_columns:

        if col not in X_train.columns:
            continue

        # Convert to string
        X_train[col] = X_train[col].astype(str)

        if X_valid is not None:
            X_valid[col] = X_valid[col].astype(str)

        if X_test is not None:
            X_test[col] = X_test[col].astype(str)

        # Build one common category list
        categories = set(
            X_train[col].unique()
        )

        if X_valid is not None:
            categories.update(
                X_valid[col].unique()
            )

        if X_test is not None:
            categories.update(
                X_test[col].unique()
            )

        categories = sorted(
            categories
        )

        # Apply same categories everywhere
        X_train[col] = pd.Categorical(
            X_train[col],
            categories=categories
        )

        if X_valid is not None:
            X_valid[col] = pd.Categorical(
                X_valid[col],
                categories=categories
            )

        if X_test is not None:
            X_test[col] = pd.Categorical(
                X_test[col],
                categories=categories
            )

    return X_train, X_valid, X_test


# ---------------------------------------------------------
# Numeric feature preparation
# ---------------------------------------------------------

def prepare_numeric_features(
    X_train,
    X_valid=None,
    X_test=None
):
    """
    Convert numeric model features to float64.

    This helps MLflow model signatures handle
    missing numeric values safely during inference.

    Categorical columns are excluded.
    """

    X_train = X_train.copy()

    if X_valid is not None:
        X_valid = X_valid.copy()

    if X_test is not None:
        X_test = X_test.copy()

    categorical_columns = [
        col.strip().replace(" ", "_")
        for col in CATEGORICAL_FEATURES
    ]

    datasets = [
        X_train,
        X_valid,
        X_test
    ]

    for dataset in datasets:

        if dataset is None:
            continue

        for col in dataset.columns:

            if col in categorical_columns:
                continue

            dataset[col] = pd.to_numeric(
                dataset[col],
                errors="coerce"
            ).astype("float64")

    return X_train, X_valid, X_test


# ---------------------------------------------------------
# Model feature preparation
# ---------------------------------------------------------

def prepare_model_features(
    X_train,
    X_valid=None,
    X_test=None
):
    """
    Prepare model input features consistently.

    This function should be used by both training
    and prediction pipelines.

    Processing:
    1. Clean feature names
    2. Prepare categorical features
    3. Convert numeric features to float64
    """

    # -----------------------------------------------------
    # Clean feature names
    # -----------------------------------------------------

    X_train = clean_feature_names(X_train)

    if X_valid is not None:
        X_valid = clean_feature_names(X_valid)

    if X_test is not None:
        X_test = clean_feature_names(X_test)

    # -----------------------------------------------------
    # Prepare categorical features
    # -----------------------------------------------------

    X_train, X_valid, X_test = (
        prepare_categorical_features(
            X_train,
            X_valid,
            X_test
        )
    )

    # -----------------------------------------------------
    # Prepare numeric features
    # -----------------------------------------------------

    X_train, X_valid, X_test = (
        prepare_numeric_features(
            X_train,
            X_valid,
            X_test
        )
    )

    return X_train, X_valid, X_test


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


# ---------------------------------------------------------
# Feature count validation
# ---------------------------------------------------------

def validate_features(X):
    """
    Validate that the model input contains
    exactly the expected feature columns.
    """

    expected_features = [
        col.strip().replace(" ", "_")
        for col in FEATURES
    ]

    actual_features = list(X.columns)

    missing_features = [
        col
        for col in expected_features
        if col not in actual_features
    ]

    extra_features = [
        col
        for col in actual_features
        if col not in expected_features
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    if extra_features:
        raise ValueError(
            f"Unexpected model features: {extra_features}"
        )

    return True
