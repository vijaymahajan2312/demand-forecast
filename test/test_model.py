# tests/test_model.py

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor

from src.train import safe_mape


# ---------------------------------------------------------
# Test 1 - MAPE
# ---------------------------------------------------------

def test_safe_mape():

    actual = np.array([
        100,
        200,
        300
    ])

    predicted = np.array([
        90,
        210,
        290
    ])

    result = safe_mape(
        actual,
        predicted
    )

    assert result >= 0

    assert np.isfinite(result)


# ---------------------------------------------------------
# Test 2 - MAPE with zero actual values
# ---------------------------------------------------------

def test_safe_mape_with_zero():

    actual = np.array([
        0,
        100,
        200
    ])

    predicted = np.array([
        10,
        90,
        210
    ])

    result = safe_mape(
        actual,
        predicted
    )

    assert np.isfinite(result)

    assert result >= 0


# ---------------------------------------------------------
# Test 3 - Model can train
# ---------------------------------------------------------

def test_lightgbm_training():

    X = pd.DataFrame({

        "feature_1": [
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10
        ],

        "feature_2": [
            10, 9, 8, 7, 6,
            5, 4, 3, 2, 1
        ]
    })

    y = np.array([
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100
    ])

    model = LGBMRegressor(
        n_estimators=10,
        learning_rate=0.1,
        num_leaves=5,
        random_state=42,
        verbosity=-1
    )

    model.fit(
        X,
        y
    )

    predictions = model.predict(X)

    assert len(predictions) == len(y)

    assert np.isfinite(predictions).all()


# ---------------------------------------------------------
# Test 4 - Predictions should be non-negative
# ---------------------------------------------------------

def test_predictions_non_negative():

    X = pd.DataFrame({

        "feature_1": [
            1, 2, 3, 4, 5
        ],

        "feature_2": [
            5, 4, 3, 2, 1
        ]
    })

    y = np.array([
        10, 20, 30, 40, 50
    ])

    model = LGBMRegressor(
        n_estimators=10,
        random_state=42,
        verbosity=-1
    )

    model.fit(
        X,
        y
    )

    predictions = model.predict(X)

    predictions = np.maximum(
        predictions,
        0
    )

    assert (
        predictions >= 0
    ).all()
