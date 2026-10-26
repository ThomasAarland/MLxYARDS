# ML × YARDS — NFL Expected Yards Predictor

**Course:** DAT158, assignment 2

**Student:** Thomas Årland

**Date:** 26 October 2026

**Repository:** https://github.com/ThomasAarland/MLxYARDS

**Status:** Technical report draft based on the executed experiment; student review and reflection pending.

## 1. Problem and intended users

This project creates a machine learning website for exploring the expected yards gained
on an NFL offensive play. A user supplies a game situation and compares the estimated
mean outcome of a run and a pass/dropback. The intended users are football fans and
students interested in how game context relates to outcomes. The product is an
educational analytics tool, not a system for making definitive tactical recommendations.

The task is supervised regression. The target is recorded yards gained on a play,
including zero gains and losses. Success means a reproducible model that improves on a
simple baseline on later-season data and a usable website that communicates uncertainty
and limitations honestly. No numerical performance threshold was imposed after seeing
the test results.

## 2. Data collection and understanding

The data source is the nflverse play-by-play collection [1], accessed with nflreadpy [2].
The experiment uses 2021–2025 regular-season and postseason records. The original input
contains **247,284 rows**. Source URLs and SHA-256 fingerprints of the selected-column
snapshots are stored alongside the results. The exact compact snapshots are included in
`data/snapshot/` with nflverse attribution. The data dictionary [3] defines the variables.

Inputs are play choice, down, yards to go, distance from the opponent's end zone,
offense score differential, quarter, and seconds remaining in the quarter. These can be
specified before the snap. Identifiers and season are used for auditing and splitting,
not supplied to the model. No completion indicator, air yards, touchdown outcome,
post-play score, or other outcome measurement is used as a predictor.

The recorded play-type field classifies scrambles as runs. For this product, scrambles
are reassigned to the pass/dropback category to approximate the intended play choice.
Sacks remain in that category. The post-play dropback/scramble indicators are used only
to construct this category, never as separate predictors. This proxy cannot perfectly
recover pre-snap intention, especially for options and broken plays.

## 3. Preparation and experimental design

The cleaning pipeline removes duplicate game/play identifiers, special teams, no-play
events, kneels, spikes and two-point attempts. It excludes missing or non-finite model
values, overtime, invalid downs/clocks and physically inconsistent distances or gains.
Run/pass records with valid outcomes remain even if a penalty occurred; the target is
recorded play yardage, not total penalty-adjusted field movement. Complete-case removal
is used rather than inventing missing game states. These filters may introduce selection
bias, which is a limitation.

After play filtering, 175,790 rows remain; removing 1,209 missing, invalid or overtime
rows leaves **174,581 plays**: 105,928 dropbacks and 68,653 runs. No duplicate identifiers
were found. Negative gains and legitimate long gains are retained.

| Split | Seasons | Plays | Purpose |
|---|---|---:|---|
| Training | 2021–2023 | 105,580 | Learn preprocessing and model parameters |
| Validation | 2024 | 34,728 | Select a model configuration |
| Test | 2025 | 34,273 | Final evaluation of the frozen winner |

Chronological splitting reflects deployment into a later season. Entire games stay in
one split. It avoids a random split mixing plays from the same game across training and
evaluation. Within-game correlation still affects uncertainty estimates; no confidence
interval or claim of statistical significance is made.

## 4. Models and selection

All candidates use a scikit-learn pipeline [4]: one-hot encoding of play choice and
standardization of numerical inputs, fitted on training data only. Scaling is useful
for Ridge; the same transformation is harmless for tree splits.

- **Mean baseline:** always predicts the training target mean.
- **Ridge regression:** linear relationships with L2 regularization, alpha = 10.
- **Histogram gradient boosting:** two configurations with 15 or 31 leaves, 160
  iterations, learning rate 0.06, minimum 100 samples per leaf and L2 penalty 10.
  Early stopping is disabled so no random internal validation set is introduced.

This is a small, predefined comparison rather than an exhaustive search. Both boosting
models use squared-error loss and random seed 42. Selection uses validation RMSE because
squared-error minimization targets the conditional mean, matching “expected yards.”
MAE describes absolute misses; R² compares squared error to a constant evaluation-set
mean. The winner is retained with its original training fit rather than refitting on
2024. Only the winner is evaluated on 2025.

## 5. Results and interpretation

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Mean baseline | 5.768 | 8.594 | -0.0001 |
| Ridge | 5.613 | 8.481 | 0.0260 |
| Boosting, 15 leaves | **5.578** | **8.441** | **0.0351** |
| Boosting, 31 leaves | 5.586 | 8.452 | 0.0327 |

The 15-leaf model wins. Its validation RMSE is approximately **1.78% lower** than the
baseline. The 31-leaf model improves training RMSE (8.299 versus 8.348) but performs
slightly worse on validation, illustrating why training scores alone are insufficient.

On the **34,273 test plays in 2025**, the frozen winner achieves:

| Measure | Overall | Run | Pass/dropback |
|---|---:|---:|---:|
| MAE, yards | 5.518 | 3.590 | 6.790 |
| RMSE, yards | 8.279 | 6.102 | 9.445 |
| R² | 0.0329 | 0.0240 | 0.0198 |

The overall R² indicates limited predictive power for individual plays. Useful context
does not identify whether a particular pass is incomplete or becomes a long gain. The
larger dropback errors reflect its broader outcome distribution. Predictions near the
mean can still be appropriate for expected yardage, but this experiment does not establish
precise conditional calibration in every game situation.

Permutation importance uses 6,000 sampled validation plays and five shuffles per input.
Field position and play choice have the largest mean RMSE increases when shuffled:
approximately 0.176 and 0.133 yards. Other features contribute much smaller increases.
This measures global model reliance, not causal effects; correlated inputs can share
importance, and shuffling can construct implausible combinations.

## 6. Software product and verification

The Streamlit website [5] has three tabs: play explorer, model analysis, and methodology.
The explorer validates the situation and immediately loads predictions from a saved
joblib pipeline. The field and first-down marker update with ball position and distance.
Moving inside the previous line to gain clamps yards to go to the goal line. The clock
allows 0:00 through 15:00. Plotly [6] renders the field, comparisons and evaluation charts.

The dashboard reads generated artifacts rather than example results. It shows the
validation comparison, final test scores, observed-versus-predicted outcomes, yardage
distributions, residuals, error by down and feature importance. The scatterplot samples
3,000 test plays for readability; other analysis charts use all retained test plays.

Tests cover filtering, temporal separation, invalid inputs, scrambles/sacks, model
serialization, artifact/metric consistency, field coordinates and Streamlit interactions.
The app test prevents model fitting and checks pass selection, goal-line behavior and
the maximum clock. GitHub Actions is configured to run the suite on Python 3.12.
Installation, retraining and Streamlit Community Cloud deployment steps are in the README.
The included 60-second captioned walkthrough uses screenshots of real browser interactions;
a continuous narrated student recording can supplement it. A public hosted app is not yet configured.

## 7. Limitations and further work

Team/player quality, personnel, weather, formation, play design and injuries are absent.
The model can miss changes across eras and has sparse support in rare situations.
Overtime is excluded. Historical play choices are confounded by coaching decisions and
opponent expectations; changing the play-choice input is not a randomized intervention.
Expected yards also ignores turnover risk and game objectives such as winning or scoring.

Further work could use game-grouped uncertainty estimation, historical team features
computed without future information, temporal cross-validation, and calibrated quantile
models. A future model should be evaluated on a new holdout after any changes informed
by these 2025 results.

## 8. Contribution, reflection and AI assistance

The named student is Thomas Årland. OpenAI Codex assisted with the implementation,
experimental setup, execution, automated tests, interface design and drafting of this
report on 26 October 2026. The attached NFL project specification and assignment brief
were supplied by the student. References were consulted for dataset semantics and tool
usage. AI-generated work must be reviewed and understood before submission.

**Student completion needed:** add your own reflection on what you learned, changes you
made after reviewing the implementation, and actual time spent. These personal claims
have not been invented. Confirm instructor permission for individual submission because
the assignment normally requests groups of 2–3. The course's exact report template was
not used; this report follows the ML lifecycle and the supplied assignment requirements.

## References

Full links, licensing notes and access date are in [sources.md](sources.md).

[1] nflverse data repository and play-by-play releases.

[2] nflreadpy documentation.

[3] nflreadr play-by-play data dictionary.

[4] scikit-learn documentation: DummyRegressor, Ridge, histogram gradient boosting and permutation importance.

[5] Streamlit documentation and Community Cloud deployment guide.

[6] Plotly Python documentation.
