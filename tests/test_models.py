# tests/test_models.py

"""
Basic end-to-end tests for the Steam retention modeling code.

These tests check that:
1. The modeling dataset loads correctly.
2. Feature columns can be generated.
3. The selected Gradient Boosting model can fit and predict.
4. Temporal cross-validation returns the expected number of folds.
"""

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_DIR),
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.models.train import (
    TARGET,
    DEFAULT_FOLDS,
    get_feature_columns,
    build_final_gradient_boosting,
    build_pipeline,
    evaluate_model_over_time,
)


# ============================================================
# DATA
# ============================================================

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "steam_modeling_data_large.csv"
)


def load_modeling_data():
    """
    Load the processed modeling dataset.
    """

    return pd.read_csv(DATA_PATH)


# ============================================================
# TEST 1: DATA LOADS
# ============================================================

def test_modeling_data_loads():

    data = load_modeling_data()

    assert len(data) > 0

    assert TARGET in data.columns

    assert "release_year" in data.columns


# ============================================================
# TEST 2: FEATURE COLUMNS
# ============================================================

def test_feature_columns():

    data = load_modeling_data()

    feature_cols = get_feature_columns(data)

    assert len(feature_cols) > 0

    assert "asinh_month1_players" in feature_cols

    assert "years_since_2015" in feature_cols

    for feature in feature_cols:
        assert feature in data.columns


# ============================================================
# TEST 3: MODEL FITS AND PREDICTS
# ============================================================

def test_model_fits_and_predicts():

    data = load_modeling_data()

    feature_cols = get_feature_columns(data)

    # Use a small subset so the test runs quickly
    sample = data[
        data["release_year"] <= 2018
    ].head(300)

    X = sample[feature_cols]
    y = sample[TARGET]

    model = build_final_gradient_boosting()

    pipeline = build_pipeline(model)

    pipeline.fit(
        X,
        y,
    )

    predictions = pipeline.predict(X)

    assert len(predictions) == len(sample)

    assert predictions.ndim == 1


# ============================================================
# TEST 4: TEMPORAL CV OUTPUT SHAPE
# ============================================================

def test_temporal_cv_returns_expected_folds():

    data = load_modeling_data()

    development_data = data[
        data["release_year"] <= 2021
    ].copy()

    feature_cols = get_feature_columns(
        development_data
    )

    model = build_final_gradient_boosting()

    results = evaluate_model_over_time(
        model_name="Test Gradient Boosting",
        model=model,
        data=development_data,
        feature_cols=feature_cols,
        target=TARGET,
        fold_definitions=DEFAULT_FOLDS,
    )

    assert len(results) == len(DEFAULT_FOLDS)

    assert set(
        [
            "MAE",
            "RMSE",
            "R2",
            "validation_year",
        ]
    ).issubset(
        results.columns
    )

    assert results["MAE"].notna().all()

    assert results["RMSE"].notna().all()

    assert results["R2"].notna().all()