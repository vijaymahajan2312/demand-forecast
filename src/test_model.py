# src/test_model.py

import sys

import mlflow
import mlflow.lightgbm
import pandas as pd
import numpy as np

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


# ---------------------------------------------------------
# Environment diagnostics
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
# Configure Unity Catalog model registry
# ---------------------------------------------------------

mlflow.set_registry_uri("databricks-uc")


# ---------------------------------------------------------
# Load raw data
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
# Create features
# ---------------------------------------------------------

print("=" * 60)
print("STEP 2 - Creating features")
print("=" * 60)

df_features = create_features(df)

print("Feature-engineered rows:", len(df_features))
print("Feature-engineered columns:", len(df_features.columns))

print()


# ---------------------------------------------------------
# Create model dataset
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
# Prepare model features
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
# Validate feature columns
# ---------------------------------------------------------

print("=" * 60)
print("STEP 5 - Validating features")
print("=" * 60)

validate_features(X)

print("Feature validation: PASSED")
print("Number of model features:", len(X.columns))

print()


# ---------------------------------------------------------
# Load registered model
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


# ---------------------------------------------------------
# Display model information
# ---------------------------------------------------------

print("=" * 60)
print("STEP 7 - Model information")
print("=" * 60)

print("Model type:")
print(type(model))

print()

print("Model feature names:")
print(model.feature_name_)

print()

print("Model categorical features:")
print(model.categorical_feature_)

print()


# ---------------------------------------------------------
# Prepare inference data
# ---------------------------------------------------------

print("=" * 60)
print("STEP 8 - Preparing inference data")
print("=" * 60)

X_sample = X.iloc[[0]].copy()

y_actual = y.iloc[0]

print("Sample shape:", X_sample.shape)

print()

print("Categorical columns in sample:")

for col in X_sample.columns:

    if str(X_sample[col].dtype) == "category":
        print(
            col,
            "dtype =",
            X_sample[col].dtype,
            "categories =",
            list(X_sample[col].cat.categories)
        )

print()


# ---------------------------------------------------------
# Align categorical columns with training model
# ---------------------------------------------------------

print("=" * 60)
print("STEP 9 - Aligning categorical features")
print("=" * 60)

categorical_columns = [
    "Store_ID",
    "Product_ID",
    "Category",
    "Region",
    "Weather_Condition",
    "Seasonality",
    "Epidemic"
]


for col in categorical_columns:

    if col not in X_sample.columns:
        continue

    # Convert to string first
    X_sample[col] = X_sample[col].astype(str)

    # LightGBM needs categorical dtype during prediction.
    #
    # The categories are taken from the training model's
    # stored pandas categorical metadata when available.

    training_categories = None

    if hasattr(model, "pandas_categorical"):

        pandas_categorical = model.pandas_categorical

        print(
            "Stored pandas categorical metadata:",
            len(pandas_categorical)
        )

        break


# ---------------------------------------------------------
# Reconstruct LightGBM categorical metadata
# ---------------------------------------------------------

if hasattr(model, "pandas_categorical"):

    pandas_categorical = model.pandas_categorical

    print()
    print("Applying stored categorical metadata...")

    categorical_index = 0

    for col in categorical_columns:

        if col not in X_sample.columns:
            continue

        if categorical_index >= len(pandas_categorical):
            break

        categories = pandas_categorical[categorical_index]

        X_sample[col] = pd.Categorical(
            X_sample[col],
            categories=categories
        )

        print(
            col,
            "categories:",
            list(categories)
        )

        categorical_index += 1


print()

print("Final inference dtypes:")

print(X_sample.dtypes)

print()


# ---------------------------------------------------------
# Make prediction
# ---------------------------------------------------------

print("=" * 60)
print("STEP 10 - Making prediction")
print("=" * 60)

prediction = model.predict(
    X_sample
)

predicted_demand = float(
    np.asarray(prediction).reshape(-1)[0]
)

print("Predicted Demand:", predicted_demand)

print()


# ---------------------------------------------------------
# Compare prediction with actual value
# ---------------------------------------------------------

print("=" * 60)
print("STEP 11 - Prediction result")
print("=" * 60)

print(
    "Actual Demand:   ",
    float(y_actual)
)

print(
    "Predicted Demand:",
    predicted_demand
)

absolute_error = abs(
    float(y_actual) - predicted_demand
)

print(
    "Absolute Error:  ",
    absolute_error
)

print()


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print("=" * 60)
print("MODEL TEST COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print("Model:", MODEL_NAME)
print("Version:", MODEL_VERSION)
print("Model URI:", MODEL_URI)

print()
print("The registered LightGBM model was successfully")
print("loaded and used to generate a prediction.")

print("=" * 60)
