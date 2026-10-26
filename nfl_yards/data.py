"""Read nflverse data and enforce an explicit, pre-snap feature contract."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

SEASONS = [2021, 2022, 2023, 2024, 2025]
FEATURES = ["play_type", "down", "ydstogo", "yardline_100", "score_differential", "qtr", "quarter_seconds_remaining"]
NUMERIC_FEATURES = FEATURES[1:]
TARGET = "yards_gained"
SOURCE_COLUMNS = list(dict.fromkeys([
    "game_id", "play_id", "season", "season_type", *FEATURES, TARGET,
    "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike", "two_point_attempt",
]))


def load_nfl_data(cache_dir: Path = Path("data/raw")) -> tuple[pd.DataFrame, list[dict]]:
    """Download each season with nflreadpy; cache only columns used by this study."""
    import nflreadpy as nfl
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames, provenance = [], []
    for season in SEASONS:
        path = cache_dir / f"pbp_{season}.parquet"
        if not path.exists():
            print(f"Downloading nflverse season {season} ...", flush=True)
            nfl.load_pbp(season).select(SOURCE_COLUMNS).write_parquet(path)
        frame = pd.read_parquet(path)
        frames.append(frame)
        provenance.append({
            "season": season, "rows": len(frame), "cache_file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet",
        })
    return pd.concat(frames, ignore_index=True), provenance


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep regulation offensive plays with recorded outcomes, including sacks.

    Scrambles are labeled pass/dropback to approximate the intended pre-snap
    choice. No outcome flag is supplied to the estimator as an input feature.
    Valid run/pass penalty plays retain nflverse's recorded yards_gained target.
    """
    missing = set(SOURCE_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    audit = {"raw_rows": len(raw)}
    data = raw.drop_duplicates(["game_id", "play_id"]).copy()
    audit["duplicates_removed"] = len(raw) - len(data)
    valid_play = (data.play_type.isin(["run", "pass"])
                  & data.season_type.isin(["REG", "POST"])
                  & data.season.isin(SEASONS)
                  & data.qb_kneel.eq(0) & data.qb_spike.eq(0)
                  & data.two_point_attempt.eq(0))
    data = data.loc[valid_play].copy()
    audit["relevant_play_rows"] = len(data)
    data["play_type"] = np.where(data.qb_dropback.eq(1) | data.qb_scramble.eq(1)
                                 | data.play_type.eq("pass"), "pass", "run")
    for column in [*NUMERIC_FEATURES, TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURES, TARGET, "game_id", "play_id"])
    valid = (data.down.isin([1, 2, 3, 4]) & data.qtr.isin([1, 2, 3, 4])
             & data.ydstogo.between(1, 99) & data.yardline_100.between(1, 99)
             & (data.ydstogo <= data.yardline_100)
             & data.quarter_seconds_remaining.between(0, 900)
             & data.score_differential.between(-99, 99)
             & (data[TARGET] <= data.yardline_100)
             & (data[TARGET] >= -(100 - data.yardline_100)))
    data = data.loc[valid, ["game_id", "play_id", "season", *FEATURES, TARGET]].reset_index(drop=True)
    audit["clean_rows"] = len(data)
    audit["missing_invalid_or_overtime_removed"] = audit["relevant_play_rows"] - len(data)
    audit["rows_by_season"] = {str(int(k)): int(v) for k, v in data.groupby("season").size().items()}
    audit["rows_by_play_type"] = data.play_type.value_counts().to_dict()
    return data, audit


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[:, FEATURES].copy()


def split_dataset(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = {
        "train": data[data.season.isin([2021, 2022, 2023])].copy(),
        "validation": data[data.season.eq(2024)].copy(),
        "test": data[data.season.eq(2025)].copy(),
    }
    for name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"The {name} split is empty. All five seasons are required.")
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if set(splits[left].game_id) & set(splits[right].game_id):
                raise ValueError("A game appears in multiple season splits.")
    return splits
