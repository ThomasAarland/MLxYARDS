"""Training and inference share the same sklearn preprocessing pipeline."""
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import FEATURES, NUMERIC_FEATURES, TARGET, prepare_features

SEED = 42


def train_models(train: pd.DataFrame) -> dict:
    candidates = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=10.0),
        "Gradient boosting (15 leaves)": HistGradientBoostingRegressor(
            max_iter=160, max_leaf_nodes=15, learning_rate=0.06,
            min_samples_leaf=100, l2_regularization=10, early_stopping=False, random_state=SEED),
        "Gradient boosting (31 leaves)": HistGradientBoostingRegressor(
            max_iter=160, max_leaf_nodes=31, learning_rate=0.06,
            min_samples_leaf=100, l2_regularization=10, early_stopping=False, random_state=SEED),
    }
    fitted = {}
    for name, estimator in candidates.items():
        print(f"Training {name} ...", flush=True)
        preprocessor = ColumnTransformer([
            ("play_type", OneHotEncoder(categories=[["run", "pass"]], handle_unknown="error", sparse_output=False), ["play_type"]),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ])
        pipeline = Pipeline([("features", preprocessor), ("model", estimator)])
        pipeline.fit(prepare_features(train), train[TARGET])
        fitted[name] = pipeline
    return fitted


def regression_metrics(actual, predicted) -> dict:
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted))}


def evaluate_models(models: dict, data: pd.DataFrame) -> dict:
    return {name: regression_metrics(data[TARGET], model.predict(prepare_features(data)))
            for name, model in models.items()}


def select_best_model(validation_metrics: dict) -> str:
    # Squared error targets the conditional mean (expected yards), unlike MAE.
    return min(validation_metrics, key=lambda name: validation_metrics[name]["RMSE"])


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_model(path: Path):
    # Only load repository-owned, trusted artifacts: joblib uses pickle.
    return joblib.load(path)


def validate_inputs(situation: dict) -> None:
    if situation.get("play_type") not in ("run", "pass"):
        raise ValueError("Play type must be run or pass.")
    bounds = {"down": (1, 4), "ydstogo": (1, 99), "yardline_100": (1, 99),
              "score_differential": (-28, 28), "qtr": (1, 4), "quarter_seconds_remaining": (0, 900)}
    for field, (low, high) in bounds.items():
        value = situation.get(field)
        if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{field} must be a number.")
        if not math.isfinite(value) or not low <= value <= high or int(value) != value:
            raise ValueError(f"{field} must be a whole number between {low} and {high}.")
    if situation["ydstogo"] > situation["yardline_100"]:
        raise ValueError("Yards to go cannot exceed the distance to the goal line.")


def predict_yards(model, situation: dict) -> float:
    validate_inputs(situation)
    return float(model.predict(pd.DataFrame([situation], columns=FEATURES))[0])


def compare_play_types(model, situation: dict) -> dict:
    values = {play: predict_yards(model, {**situation, "play_type": play}) for play in ("run", "pass")}
    return {**values, "difference": values["pass"] - values["run"]}
