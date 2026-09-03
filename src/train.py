# src/train.py

import mlflow
import mlflow.lightgbm

import pandas as pd
import numpy as np

from mlflow.models import infer_signature
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from lightgbm import LGBMRegressor

from src.features import (
    create_features,
    FEATURES,
    TARGET
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = "demand_forecasting.csv"

EXPERIMENT_NAME = "/Shared/demand-forecasting"


MODEL_PARAMS = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}


# ---------------------------------------------------------
# MAPE
# ---------------------------------------------------------

def safe_mape(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0

    if mask.sum() == 0:
        return 0.0

    return (
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100
    )


# ---------------------------------------------------------
# Time-based split
# ---------------------------------------------------------

def time_split(df):

    dates = sorted(
        df["Date"].unique()
    )

    train_end = dates[
        int(len(dates) * 0.70)
    ]

    valid_end = dates[
        int(len(dates) * 0.85)
    ]

    train_df = df[
        df["Date"] <= train_end
    ].copy()

    valid_df = df[
        (df["Date"] > train_end)
        & (df["Date"] <= valid_end)
    ].copy()

    test_df = df[
        df["Date"] > valid_end
    ].copy()

    return (
        train_df,
        valid_df,
        test_df
    )


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_model():

    print("Loading data...")

    df = pd.read_csv(DATA_PATH)

    print("Creating features...")

    df_model = create_features(df)

    print(
        "Feature-engineered rows:",
        len(df_model)
    )

    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------

    train_df, valid_df, test_df = time_split(
        df_model
    )

    print(
        "Train rows:",
        len(train_df)
    )

    print(
        "Validation rows:",
        len(valid_df)
    )

    print(
        "Test rows:",
        len(test_df)
    )

    # -----------------------------------------------------
    # X / y
    # -----------------------------------------------------

    X_train = train_df[FEATURES].copy()
    y_train = train_df[TARGET].copy()

    X_valid = valid_df[FEATURES].copy()
    y_valid = valid_df[TARGET].copy()

    X_test = test_df[FEATURES].copy()
    y_test = test_df[TARGET].copy()

    # -----------------------------------------------------
    # MLflow
    # -----------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="lightgbm_demand_forecasting"
    ):

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        model = LGBMRegressor(
            **MODEL_PARAMS
        )

        print("Training LightGBM...")

        model.fit(
            X_train,
            y_train
        )

        # -------------------------------------------------
        # Validation prediction
        # -------------------------------------------------

        y_valid_pred = model.predict(
            X_valid
        )

        y_valid_pred = np.maximum(
            y_valid_pred,
            0
        )

        # -------------------------------------------------
        # Validation metrics
        # -------------------------------------------------

        val_mae = mean_absolute_error(
            y_valid,
            y_valid_pred
        )

        val_rmse = np.sqrt(
            mean_squared_error(
                y_valid,
                y_valid_pred
            )
        )

        val_r2 = r2_score(
            y_valid,
            y_valid_pred
        )

        val_mape = safe_mape(
            y_valid,
            y_valid_pred
        )

        # -------------------------------------------------
        # Baseline
        # -------------------------------------------------

        baseline_pred = valid_df[
            "lag_1"
        ].values

        baseline_mae = mean_absolute_error(
            y_valid,
            baseline_pred
        )

        baseline_rmse = np.sqrt(
            mean_squared_error(
                y_valid,
                baseline_pred
            )
        )

        # -------------------------------------------------
        # Test
        # -------------------------------------------------

        y_test_pred = model.predict(
            X_test
        )

        y_test_pred = np.maximum(
            y_test_pred,
            0
        )

        test_mae = mean_absolute_error(
            y_test,
            y_test_pred
        )

        test_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_test_pred
            )
        )

        test_r2 = r2_score(
            y_test,
            y_test_pred
        )

        test_mape = safe_mape(
            y_test,
            y_test_pred
        )

        # -------------------------------------------------
        # Model signature
        # -------------------------------------------------

        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        input_example = X_train.iloc[
            [0]
        ].copy()

        # -------------------------------------------------
        # MLflow logging
        # -------------------------------------------------

        mlflow.log_params(
            MODEL_PARAMS
        )

        mlflow.log_metrics({

            "val_mae": val_mae,

            "val_rmse": val_rmse,

            "val_r2": val_r2,

            "val_mape": val_mape,

            "baseline_mae": baseline_mae,

            "baseline_rmse": baseline_rmse,

            "test_mae": test_mae,

            "test_rmse": test_rmse,

            "test_r2": test_r2,

            "test_mape": test_mape
        })

        # -------------------------------------------------
        # Log model
        # -------------------------------------------------

        model_info = mlflow.lightgbm.log_model(

            model,

            name="model",

            signature=signature,

            input_example=input_example
        )

        print("\nTraining completed.")

        print("\nValidation Metrics")
        print("------------------")
        print("MAE  :", val_mae)
        print("RMSE :", val_rmse)
        print("R2   :", val_r2)
        print("MAPE :", val_mape)

        print("\nTest Metrics")
        print("------------")
        print("MAE  :", test_mae)
        print("RMSE :", test_rmse)
        print("R2   :", test_r2)
        print("MAPE :", test_mape)

        print("\nBaseline")
        print("--------")
        print("MAE  :", baseline_mae)
        print("RMSE :", baseline_rmse)

        print("\nMLflow Model URI:")
        print(model_info.model_uri)

        return model_info.model_uri


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    train_model()
