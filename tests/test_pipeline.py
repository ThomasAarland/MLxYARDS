import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from nfl_yards.data import FEATURES, clean_data, prepare_features, split_dataset
from nfl_yards.model import (compare_play_types, load_model, predict_yards, regression_metrics,
                             save_model, select_best_model, validate_inputs)
from nfl_yards.visuals import render_football_field

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def situation():
    return dict(play_type="run", down=1, ydstogo=10, yardline_100=75,
                score_differential=0, qtr=1, quarter_seconds_remaining=600)


@pytest.fixture
def raw(situation):
    return pd.DataFrame([{**situation, "game_id": f"{year}_01_A_B", "play_id": 1,
                          "season": year, "season_type": "REG", "yards_gained": 5,
                          "qb_dropback": 0, "qb_scramble": 0, "qb_kneel": 0,
                          "qb_spike": 0, "two_point_attempt": 0} for year in range(2021, 2026)])


def test_cleaning_scrambles_sacks_and_invalid_plays(raw):
    rows = []
    overrides = [
        {"qb_scramble": 1, "qb_dropback": 1, "yards_gained": 8},
        {"play_type": "pass", "qb_dropback": 1, "yards_gained": -7},
        {"play_type": "no_play"}, {"qb_kneel": 1}, {"qb_spike": 1},
        {"two_point_attempt": 1}, {"qtr": 5}, {"down": np.nan},
        {"yards_gained": 100}, {"ydstogo": 80}, {"quarter_seconds_remaining": 901},
    ]
    for i, change in enumerate(overrides):
        rows.append({**raw.iloc[0].to_dict(), **change, "play_id": i + 10})
    cleaned, audit = clean_data(pd.concat([raw, pd.DataFrame(rows), raw.iloc[[0]]], ignore_index=True))
    assert len(cleaned) == 7
    assert audit["duplicates_removed"] == 1
    assert cleaned.loc[cleaned.play_id.isin([10, 11]), "play_type"].tolist() == ["pass", "pass"]
    assert -7 in cleaned.yards_gained.values
    assert list(prepare_features(cleaned)) == FEATURES
    assert "yards_gained" not in prepare_features(cleaned)
    assert "qb_scramble" not in prepare_features(cleaned)


def test_temporal_splits_and_empty_guard(raw):
    cleaned, _ = clean_data(raw)
    splits = split_dataset(cleaned)
    assert set(splits["train"].season) == {2021, 2022, 2023}
    assert set(splits["validation"].season) == {2024}
    assert set(splits["test"].season) == {2025}
    with pytest.raises(ValueError, match="test split is empty"):
        split_dataset(cleaned[cleaned.season != 2025])


@pytest.mark.parametrize("field,value", [
    ("down", 0), ("down", 1.5), ("ydstogo", 76), ("qtr", 5),
    ("quarter_seconds_remaining", 901), ("score_differential", -29),
    ("yardline_100", 0), ("play_type", "punt"), ("down", True),
    ("ydstogo", float("nan")), ("qtr", "1"),
])
def test_rejects_impossible_input(situation, field, value):
    with pytest.raises(ValueError):
        validate_inputs({**situation, field: value})


def test_goal_line_and_zero_clock_are_valid(situation):
    validate_inputs({**situation, "yardline_100": 1, "ydstogo": 1, "quarter_seconds_remaining": 0})


def test_selection_uses_rmse_not_mae():
    assert select_best_model({"a": {"MAE": 1, "RMSE": 5}, "b": {"MAE": 2, "RMSE": 4}}) == "b"


def test_saved_artifact_and_dashboard_are_consistent(situation, tmp_path):
    model = load_model(ROOT / "artifacts/model.joblib")
    expected = predict_yards(model, situation)
    save_model(model, tmp_path / "roundtrip.joblib")
    assert predict_yards(load_model(tmp_path / "roundtrip.joblib"), situation) == expected
    comparison = compare_play_types(model, situation)
    assert comparison["run"] == expected
    assert comparison["difference"] == pytest.approx(comparison["pass"] - comparison["run"])
    predictions = pd.read_parquet(ROOT / "artifacts/test_predictions.parquet")
    np.testing.assert_allclose(model.predict(prepare_features(predictions)), predictions.predicted_yards)
    metrics = json.loads((ROOT / "artifacts/metrics.json").read_text())
    assert regression_metrics(predictions.yards_gained, predictions.predicted_yards) == metrics["test"][metrics["selected_model"]]


def test_field_markers_match_input(situation):
    fig = render_football_field(situation, 4.5)
    assert list(fig.data[0].x) == [25, 25]
    assert list(fig.data[1].x) == [35, 35]
    assert list(fig.data[2].x) == [25, 29.5]
    assert all(shape.layer == "below" for shape in fig.layout.shapes)
