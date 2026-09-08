# src/test_model.py

import sys

import mlflow
import mlflow.pyfunc
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

model = mlflow.pyfunc.load_model(
    MODEL_URI
)

print()
print("Model loaded successfully!")

print()


# ---------------------------------------------------------
# Display model signature
# ---------------------------------------------------------

print("=" * 60)
print("STEP 7 - Model signature")
print("=" * 60)

if model.metadata.signature is not None:

    print(model.metadata.signature)

else:

    print("Model does not contain a signature.")

print()


# ---------------------------------------------------------
# Select one test row
# ---------------------------------------------------------

print("=" * 60)
print("STEP 8 - Selecting test row")
print("=" * 60)

# Select the first available row.
# This row already contains all required
# lag and rolling features because create_features()
# removed rows with insufficient history.

X_sample = X.iloc[[0]].copy()

y_actual = y.iloc[0]


print("Sample shape:", X_sample.shape)

print("Actual Demand:", y_actual)

print()


# ---------------------------------------------------------
# Make prediction
# ---------------------------------------------------------

print("=" * 60)
print("STEP 9 - Making prediction")
print("=" * 60)

prediction = model.predict(X_sample)

predicted_demand = float(
    np.asarray(prediction).reshape(-1)[0]
)

print("Predicted Demand:", predicted_demand)

print()


# ---------------------------------------------------------
# Compare prediction with actual value
# ---------------------------------------------------------

print("=" * 60)
print("STEP 10 - Prediction result")
print("=" * 60)

print("Actual Demand:   ", float(y_actual))
print("Predicted Demand:", predicted_demand)

absolute_error = abs(
    float(y_actual) - predicted_demand
)

print("Absolute Error:   ", absolute_error)

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
print("The registered model was successfully loaded")
print("and used to generate a prediction.")

print("=" * 60)
