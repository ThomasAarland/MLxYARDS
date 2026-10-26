# Generated model artifacts

`python train.py` creates all four files used by the website:

- `model.joblib`: selected sklearn pipeline, including preprocessing.
- `metrics.json`: actual scores, chronological splits, cleaning audit, cache hashes and package versions.
- `test_predictions.parquet`: every retained 2025 test play, its observed target and frozen-model prediction.
- `feature_importance.csv`: validation permutation importance and shuffle standard deviations.

Keep these files together when retraining. They are intentionally versioned so the app
starts without downloading data or fitting models. Only load a joblib artifact you trust.
The fitted model uses seasons 2021–2023; 2024 selects its configuration; 2025 is the final test.
Downloaded raw data caches are ignored; exact compact source-column snapshots are included
in `data/snapshot/`. Use `python train.py --cache-dir data/snapshot` to reproduce this run.
nflverse may revise its
published data; the SHA-256 cache fingerprints in `metrics.json` identify this run's inputs.
