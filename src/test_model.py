# src/test_model.py

import sys

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from features import (
    create_features,
    get_model_data,
    prepare_model_features,
    validate_features,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = (
    "/Volumes/demand_catalog/default/"
    "demand_volume/demand_forecasting.csv"
)

MODEL_NAME = "demand_catalog.default.demand_forecasting"
MODEL_VERSION = "3"

MODEL_URI = (
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)


CATEGORICAL_COLUMNS = [
    "Store_ID",
    "Product_ID",
    "Category",
    "Region",
    "Weather_Condition",
    "Seasonality",
    "Epidemic"
]


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)

print("Python:", sys.version)
print("MLflow:", mlflow.__version__)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)

print()


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------

mlflow.set_registry_uri("databricks-uc")


# ---------------------------------------------------------
# Step 1 - Load data
# ---------------------------------------------------------

print("=" * 60)
print("STEP 1 - Loading data")
print("=" * 60)

print("Data path:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("Raw rows:", len(df))
print("Raw columns:", len(df.columns))

print()


# ---------------------------------------------------------
# Step 2 - Create features
# ---------------------------------------------------------

print("=" * 60)
print("STEP 2 - Creating features")
print("=" * 60)

df_features = create_features(df)

print(
    "Feature-engineered rows:",
    len(df_features)
)

print(
    "Feature-engineered columns:",
    len(df_features.columns)
)

print()


# ---------------------------------------------------------
# Step 3 - Prepare model data
# ---------------------------------------------------------

print("=" * 60)
print("STEP 3 - Preparing model data")
print("=" * 60)

X, y = get_model_data(df_features)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Original model feature names:")
print(list(X.columns))

print()


# ---------------------------------------------------------
# Step 4 - Prepare model features
# ---------------------------------------------------------

print("=" * 60)
print("STEP 4 - Preparing model features")
print("=" * 60)

X, _, _ = prepare_model_features(X)

print("Prepared X shape:", X.shape)

print("Prepared feature names:")
print(list(X.columns))

print()


# ---------------------------------------------------------
# Step 5 - Validate features
# ---------------------------------------------------------

print("=" * 60)
print("STEP 5 - Validating features")
print("=" * 60)

validate_features(X)

print("Feature validation: PASSED")
print("Number of model features:", len(X.columns))

print()


# ---------------------------------------------------------
# Step 6 - Load registered model
# ---------------------------------------------------------

print("=" * 60)
print("STEP 6 - Loading registered model")
print("=" * 60)

print("Model URI:")
print(MODEL_URI)

model = mlflow.lightgbm.load_model(
    MODEL_URI
)

print()

print("LightGBM model loaded successfully!")

print()

print("Model type:")
print(type(model))

print()


# ---------------------------------------------------------
# Step 7 - Inspect model metadata
# ---------------------------------------------------------

print("=" * 60)
print("STEP 7 - Inspecting model metadata")
print("=" * 60)

if hasattr(model, "feature_name_"):

    print("Model feature names:")
    print(model.feature_name_)

else:

    print(
        "Model does not expose feature_name_"
    )


print()

print(
    "Checking pandas categorical metadata..."
)


if hasattr(model, "pandas_categorical"):

    pandas_categorical = (
        model.pandas_categorical
    )

    print(
        "pandas_categorical found."
    )

    print(
        "Number of categorical metadata entries:",
        len(pandas_categorical)
    )

    for i, categories in enumerate(
        pandas_categorical
    ):

        print()

        print(
            f"Categorical metadata {i}:"
        )

        print(
            "Number of categories:",
            len(categories)
        )

        print(
            "Categories:",
            list(categories)
        )

else:

    pandas_categorical = None

    print(
        "WARNING: pandas_categorical "
        "metadata not found."
    )


print()


# ---------------------------------------------------------
# Step 8 - Prepare inference sample
# ---------------------------------------------------------

print("=" * 60)
print("STEP 8 - Preparing inference data")
print("=" * 60)

X_sample = X.iloc[[0]].copy()

y_actual = y.iloc[0]

print(
    "Sample shape:",
    X_sample.shape
)

print()

print("Sample before categorical alignment:")

print(X_sample.dtypes)

print()


# ---------------------------------------------------------
# Step 9 - Align categorical features
# ---------------------------------------------------------

print("=" * 60)
print("STEP 9 - Aligning categorical features")
print("=" * 60)


if pandas_categorical is not None:

    print(
        "Applying stored LightGBM "
        "categorical metadata..."
    )

    if len(pandas_categorical) != len(
        CATEGORICAL_COLUMNS
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            "Number of stored categorical "
            "metadata entries:",
            len(pandas_categorical)
        )

        print(
            "Number of expected categorical "
            "columns:",
            len(CATEGORICAL_COLUMNS)
        )

        print()

    for i, col in enumerate(
        CATEGORICAL_COLUMNS
    ):

        if col not in X_sample.columns:

            print(
                f"Skipping {col} - "
                "column not found"
            )

            continue

        if i >= len(
            pandas_categorical
        ):

            print(
                f"No stored category metadata "
                f"available for {col}"
            )

            continue

        categories = (
            pandas_categorical[i]
        )

        X_sample[col] = pd.Categorical(
            X_sample[col].astype(str),
            categories=categories
        )

        print(
            f"{col}: "
            f"{len(categories)} categories"
        )


else:

    print(
        "No pandas categorical metadata "
        "available."
    )


print()

print(
    "Final inference dtypes:"
)

print(
    X_sample.dtypes
)

print()


# ---------------------------------------------------------
# Step 10 - Ensure feature order
# ---------------------------------------------------------

print("=" * 60)
print("STEP 10 - Ensuring feature order")
print("=" * 60)


if hasattr(model, "feature_name_"):

    model_features = list(
        model.feature_name_
    )

    print(
        "Model expects:",
        len(model_features),
        "features"
    )

    print(
        "Inference contains:",
        len(X_sample.columns),
        "features"
    )

    missing = [
        col
        for col in model_features
        if col not in X_sample.columns
    ]

    extra = [
        col
        for col in X_sample.columns
        if col not in model_features
    ]

    if missing:

        raise ValueError(
            f"Missing model features: {missing}"
        )

    if extra:

        raise ValueError(
            f"Unexpected inference features: {extra}"
        )

    X_sample = X_sample[
        model_features
    ]

    print(
        "Feature order: PASSED"
    )

else:

    print(
        "Model feature_name_ unavailable."
    )


print()


# ---------------------------------------------------------
# Step 11 - Final categorical inspection
# ---------------------------------------------------------

print("=" * 60)
print("STEP 11 - Final categorical inspection")
print("=" * 60)

for col in CATEGORICAL_COLUMNS:

    if col not in X_sample.columns:
        continue

    print(
        col,
        "| dtype =",
        X_sample[col].dtype
    )

    if pd.api.types.is_categorical_dtype(
        X_sample[col]
    ):

        print(
            "  categories =",
            list(
                X_sample[col].cat.categories
            )
        )

        print(
            "  value =",
            X_sample[col].iloc[0]
        )


print()


# ---------------------------------------------------------
# Step 12 - Prediction
# ---------------------------------------------------------

print("=" * 60)
print("STEP 12 - Making prediction")
print("=" * 60)

prediction = model.predict(
    X_sample
)

predicted_demand = float(
    np.asarray(prediction)
    .reshape(-1)[0]
)

print(
    "Predicted Demand:",
    predicted_demand
)

print()


# ---------------------------------------------------------
# Step 13 - Result
# ---------------------------------------------------------

print("=" * 60)
print("STEP 13 - Prediction result")
print("=" * 60)

actual_demand = float(
    y_actual
)

absolute_error = abs(
    actual_demand -
    predicted_demand
)

print(
    "Actual Demand:   ",
    actual_demand
)

print(
    "Predicted Demand:",
    predicted_demand
)

print(
    "Absolute Error:  ",
    absolute_error
)

print()


# ---------------------------------------------------------
# Final
# ---------------------------------------------------------

print("=" * 60)
print("MODEL TEST COMPLETED SUCCESSFULLY")
print("=" * 60)

print()

print(
    "Model:",
    MODEL_NAME
)

print(
    "Version:",
    MODEL_VERSION
)

print(
    "Model URI:",
    MODEL_URI
)

print()

print(
    "The registered LightGBM model was "
    "successfully loaded and used "
    "for inference."
)

print("=" * 60)
