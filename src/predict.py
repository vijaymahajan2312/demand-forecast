# src/predict.py

import mlflow
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_URI = (
    "models:/main.default.demand_forecasting/1"
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

def load_model():

    print(
        "Loading model:",
        MODEL_URI
    )

    model = mlflow.pyfunc.load_model(
        MODEL_URI
    )

    print(
        "Model loaded successfully."
    )

    return model


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict(
    model,
    input_data
):

    predictions = model.predict(
        input_data
    )

    predictions = np.maximum(
        predictions,
        0
    )

    return predictions


# ---------------------------------------------------------
# Example
# ---------------------------------------------------------

if __name__ == "__main__":

    model = load_model()

    print(
        "Model loaded and ready for prediction."
    )
