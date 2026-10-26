"""Run: python train.py. No test outcomes participate in model selection."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

# Keep CPU use predictable on laptops and cloud hosts; set before sklearn import.
os.environ.setdefault("OMP_NUM_THREADS", "4")
from sklearn.inspection import permutation_importance
from nfl_yards.data import load_nfl_data, clean_data, split_dataset, prepare_features, TARGET
from nfl_yards.model import SEED, train_models, evaluate_models, select_best_model, save_model, regression_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    raw, provenance = load_nfl_data(args.cache_dir)
    clean, audit = clean_data(raw)
    splits = split_dataset(clean)
    models = train_models(splits["train"])
    training = evaluate_models(models, splits["train"])
    validation = evaluate_models(models, splits["validation"])
    winner = select_best_model(validation)
    print(f"Selected by validation RMSE: {winner}", flush=True)
    selected = models[winner]
    test = splits["test"].copy()
    test["predicted_yards"] = selected.predict(prepare_features(test))
    test["residual"] = test[TARGET] - test.predicted_yards
    test_metrics = regression_metrics(test[TARGET], test.predicted_yards)
    importance_data = splits["validation"].sample(min(6000, len(splits["validation"])), random_state=SEED)
    importance = permutation_importance(selected, prepare_features(importance_data), importance_data[TARGET],
                                        scoring="neg_root_mean_squared_error", n_repeats=5, random_state=SEED, n_jobs=1)
    import pandas as pd
    feature_importance = pd.DataFrame({"feature": prepare_features(importance_data).columns,
                                      "rmse_increase": importance.importances_mean,
                                      "std": importance.importances_std}).sort_values("rmse_increase", ascending=False)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    save_model(selected, output / "model.joblib")
    test.to_parquet(output / "test_predictions.parquet", index=False)
    feature_importance.to_csv(output / "feature_importance.csv", index=False)
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "random_seed": SEED,
        "selection_metric": "validation RMSE", "selected_model": winner,
        "fit_seasons": [2021, 2022, 2023], "validation_season": 2024, "test_season": 2025,
        "refit_on_validation": False, "train": training, "validation": validation,
        "test": {winner: test_metrics},
        "test_by_play_type": {p: regression_metrics(f[TARGET], f.predicted_yards) for p, f in test.groupby("play_type")},
        "split_counts": {name: len(frame) for name, frame in splits.items()}, "cleaning": audit,
        "data_provenance": provenance,
        "environment": {"python": platform.python_version(), **{p: version(p) for p in
                        ["nflreadpy", "numpy", "pandas", "scikit-learn", "streamlit", "plotly", "joblib", "pyarrow"]}},
        "feature_importance": {"split": "validation", "sample_size": len(importance_data), "repeats": 5,
                               "method": "Permutation increase in RMSE; not causal or local attribution."},
    }
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"test": test_metrics, "counts": results["split_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
