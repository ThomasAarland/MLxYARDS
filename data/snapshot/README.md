# Reproducible nflverse source-column snapshots

These five Parquet files contain the columns used by this study, before its play filtering.
They were downloaded from nflverse on 26 October 2026 with nflreadpy and reduced to the
columns listed in `nfl_yards/data.py`. Full original release files are not redistributed.

Run `python train.py --cache-dir data/snapshot` to reproduce the supplied experiment.
The SHA-256 hashes in `artifacts/metrics.json` identify these exact files.

Source: https://github.com/nflverse/nflverse-data/releases/tag/pbp

Credit: nflverse contributors

License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

Source license: https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md

Changes: column selection and Parquet re-encoding. No filtering or target modification
has yet been applied to these snapshots. Training subsequently filters invalid/irrelevant
plays and groups scrambles with dropbacks as documented in the report.
