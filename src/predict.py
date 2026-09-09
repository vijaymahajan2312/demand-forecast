# src/predict.py

import sys
from datetime import datetime, timezone

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

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = (
    "demand_catalog.default.demand_forecasting"
)

MODEL_VERSION = "3"

MODEL_URI = (
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)

INPUT_TABLE = (
    "demand_catalog.default.demand_forecast_input"
)

OUTPUT_TABLE = (
    "demand_catalog.default.demand_forecast_predictions"
)


# =========================================================
# CATEGORICAL COLUMNS
# =========================================================

CATEGORICAL_COLUMNS = [
    "Store_ID",
    "Product_ID",
    "Category",
    "Region",
    "Weather_Condition",
    "Seasonality",
    "Epidemic",
]


# =========================================================
# LOAD REGISTERED MODEL
# =========================================================

def load_model():

    print("=" * 60)
    print("LOADING REGISTERED MODEL")
    print("=" * 60)

    print()
    print("Model URI:")
    print(MODEL_URI)
    print()

    # Unity Catalog Model Registry
    mlflow.set_registry_uri(
        "databricks-uc"
    )

    # Use LightGBM flavor.
    # This is important because the trained model
    # uses pandas categorical features.
    model = mlflow.lightgbm.load_model(
        MODEL_URI
    )

    print()
    print("Model loaded successfully.")
    print(
        "Model type:",
        type(model)
    )
    print()

    return model


# =========================================================
# LOAD INPUT DATA
# =========================================================

def load_input_data(spark):

    print("=" * 60)
    print("LOADING INPUT DELTA TABLE")
    print("=" * 60)

    print()
    print("Input table:")
    print(INPUT_TABLE)
    print()

    # Read Delta table
    df = (
        spark
        .table(INPUT_TABLE)
        .toPandas()
    )

    print(
        "Input rows:",
        len(df)
    )

    print(
        "Input columns:",
        len(df.columns)
    )

    print()

    # -----------------------------------------------------
    # Convert Delta-safe column names back to the original
    # names expected by features.py
    # -----------------------------------------------------

    df = df.rename(
        columns={
            "Store_ID": "Store ID",
            "Product_ID": "Product ID",
            "Inventory_Level": "Inventory Level",
            "Units_Sold": "Units Sold",
            "Units_Ordered": "Units Ordered",
            "Weather_Condition": "Weather Condition",
            "Competitor_Pricing": "Competitor Pricing",
        }
    )

    print(
        "Columns after renaming:"
    )

    for column in df.columns:
        print(
            " -",
            column
        )

    print()

    return df


# =========================================================
# PREPARE PREDICTION DATA
# =========================================================

def prepare_prediction_data(
    input_data
):

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    print()

    # -----------------------------------------------------
    # Create exactly the same features used during training
    # -----------------------------------------------------

    df_features = create_features(
        input_data
    )

    print(
        "Feature-engineered rows:",
        len(df_features)
    )

    print(
        "Feature-engineered columns:",
        len(df_features.columns)
    )

    print()

    # -----------------------------------------------------
    # Get X and y
    # -----------------------------------------------------

    X, y = get_model_data(
        df_features
    )

    print(
        "Model feature count:",
        len(X.columns)
    )

    print()

    # -----------------------------------------------------
    # Prepare categorical and numeric features
    # -----------------------------------------------------

    X, _, _ = prepare_model_features(
        X
    )

    # -----------------------------------------------------
    # Validate model features
    # -----------------------------------------------------

    validate_features(
        X
    )

    print(
        "Feature validation: PASSED"
    )

    print()

    return (
        df_features,
        X,
        y
    )


# =========================================================
# ALIGN FEATURES WITH TRAINED MODEL
# =========================================================

def align_features_with_model(
    model,
    X
):

    print("=" * 60)
    print("ALIGNING FEATURES WITH MODEL")
    print("=" * 60)

    print()

    X = X.copy()

    # -----------------------------------------------------
    # Get feature names from trained LightGBM model
    # -----------------------------------------------------

    if hasattr(
        model,
        "feature_name_"
    ):

        model_features = list(
            model.feature_name_
        )

        print(
            "Model expects:",
            len(model_features),
            "features"
        )

        print()

        # -------------------------------------------------
        # Check for missing features
        # -------------------------------------------------

        missing_features = [
            col
            for col in model_features
            if col not in X.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing model features: "
                f"{missing_features}"
            )

        # -------------------------------------------------
        # Check for unexpected features
        # -------------------------------------------------

        extra_features = [
            col
            for col in X.columns
            if col not in model_features
        ]

        if extra_features:

            raise ValueError(
                "Unexpected model features: "
                f"{extra_features}"
            )

        # -------------------------------------------------
        # Force exact same feature order as training
        # -------------------------------------------------

        X = X[
            model_features
        ]

    # -----------------------------------------------------
    # Preserve pandas categorical dtype
    # -----------------------------------------------------

    for col in CATEGORICAL_COLUMNS:

        if col in X.columns:

            X[col] = X[col].astype(
                "category"
            )

    print(
        "Feature alignment: PASSED"
    )

    print(
        "Final prediction shape:",
        X.shape
    )

    print()

    return X


# =========================================================
# GENERATE PREDICTIONS
# =========================================================

def generate_predictions(
    model,
    X
):

    print("=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)

    print()

    # -----------------------------------------------------
    # Generate predictions
    # -----------------------------------------------------

    predictions = model.predict(
        X
    )

    # -----------------------------------------------------
    # Demand cannot be negative
    # -----------------------------------------------------

    predictions = np.maximum(
        predictions,
        0
    )

    print(
        "Predictions generated:",
        len(predictions)
    )

    print(
        "Minimum prediction:",
        float(
            np.min(predictions)
        )
    )

    print(
        "Maximum prediction:",
        float(
            np.max(predictions)
        )
    )

    print(
        "Average prediction:",
        float(
            np.mean(predictions)
        )
    )

    print()

    return predictions


# =========================================================
# BUILD PREDICTION OUTPUT
# =========================================================

def build_prediction_output(
    df_features,
    y,
    predictions
):

    print("=" * 60)
    print("BUILDING PREDICTION OUTPUT")
    print("=" * 60)

    print()

    output = pd.DataFrame()

    # -----------------------------------------------------
    # Date
    # -----------------------------------------------------

    output["Date"] = (
        df_features["Date"].values
    )

    # -----------------------------------------------------
    # Store
    # -----------------------------------------------------

    output["Store_ID"] = (
        df_features["Store ID"].values
    )

    # -----------------------------------------------------
    # Product
    # -----------------------------------------------------

    output["Product_ID"] = (
        df_features["Product ID"].values
    )

    # -----------------------------------------------------
    # Actual demand
    # -----------------------------------------------------

    output["Actual_Demand"] = (
        y.values
    )

    # -----------------------------------------------------
    # Predicted demand
    # -----------------------------------------------------

    output["Predicted_Demand"] = (
        predictions
    )

    # -----------------------------------------------------
    # Absolute error
    # -----------------------------------------------------

    output["Absolute_Error"] = np.abs(
        output["Actual_Demand"]
        -
        output["Predicted_Demand"]
    )

    # -----------------------------------------------------
    # Model information
    # -----------------------------------------------------

    output["Model_Name"] = (
        MODEL_NAME
    )

    output["Model_Version"] = (
        MODEL_VERSION
    )

    # -----------------------------------------------------
    # Prediction timestamp
    # -----------------------------------------------------

    output["Prediction_Timestamp"] = (
        datetime.now(
            timezone.utc
        )
    )

    print(
        "Output rows:",
        len(output)
    )

    print()

    print(
        "Output columns:"
    )

    for column in output.columns:
        print(
            " -",
            column
        )

    print()

    return output


# =========================================================
# SAVE PREDICTIONS TO DELTA
# =========================================================

def save_predictions(
    spark,
    output
):

    print("=" * 60)
    print("SAVING PREDICTIONS")
    print("=" * 60)

    print()

    print(
        "Output table:"
    )

    print(
        OUTPUT_TABLE
    )

    print()

    # -----------------------------------------------------
    # Convert Pandas DataFrame to Spark DataFrame
    # -----------------------------------------------------

    prediction_df = (
        spark
        .createDataFrame(
            output
        )
    )

    # -----------------------------------------------------
    # Save as Delta table
    # -----------------------------------------------------

    (
        prediction_df
        .write
        .format("delta")
        .mode("overwrite")
        .option(
            "overwriteSchema",
            "true"
        )
        .saveAsTable(
            OUTPUT_TABLE
        )
    )

    print()
    print(
        "Predictions saved successfully."
    )
    print()


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    print("=" * 60)
    print("DEMAND FORECASTING PREDICTION JOB")
    print("=" * 60)

    print()

    # -----------------------------------------------------
    # Environment information
    # -----------------------------------------------------

    print(
        "Python:",
        sys.version
    )

    print(
        "Pandas:",
        pd.__version__
    )

    print(
        "NumPy:",
        np.__version__
    )

    print(
        "MLflow:",
        mlflow.__version__
    )

    print()

    # -----------------------------------------------------
    # Create Spark session
    # -----------------------------------------------------

    from pyspark.sql import SparkSession

    spark = (
        SparkSession
        .builder
        .getOrCreate()
    )

    # -----------------------------------------------------
    # STEP 1
    # Load registered model
    # -----------------------------------------------------

    model = load_model()

    # -----------------------------------------------------
    # STEP 2
    # Load input data
    # -----------------------------------------------------

    input_data = load_input_data(
        spark
    )

    # -----------------------------------------------------
    # STEP 3
    # Feature engineering
    # -----------------------------------------------------

    (
        df_features,
        X,
        y
    ) = prepare_prediction_data(
        input_data
    )

    # -----------------------------------------------------
    # STEP 4
    # Align features with model
    # -----------------------------------------------------

    X = align_features_with_model(
        model,
        X
    )

    # -----------------------------------------------------
    # STEP 5
    # Generate predictions
    # -----------------------------------------------------

    predictions = generate_predictions(
        model,
        X
    )

    # -----------------------------------------------------
    # STEP 6
    # Build prediction output
    # -----------------------------------------------------

    output = build_prediction_output(
        df_features,
        y,
        predictions
    )

    # -----------------------------------------------------
    # STEP 7
    # Save predictions
    # -----------------------------------------------------

    save_predictions(
        spark,
        output
    )

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("=" * 60)
    print("PREDICTION JOB COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print()

    print(
        "Model:",
        MODEL_NAME
    )

    print(
        "Model Version:",
        MODEL_VERSION
    )

    print(
        "Input Table:",
        INPUT_TABLE
    )

    print(
        "Output Table:",
        OUTPUT_TABLE
    )

    print(
        "Predictions:",
        len(predictions)
    )

    print(
        "Average Predicted Demand:",
        float(
            np.mean(predictions)
        )
    )

    print(
        "Minimum Predicted Demand:",
        float(
            np.min(predictions)
        )
    )

    print(
        "Maximum Predicted Demand:",
        float(
            np.max(predictions)
        )
    )

    print()

    print("=" * 60)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
