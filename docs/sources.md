# Sources and acknowledgments

Accessed 26 October 2026.

1. **nflverse play-by-play data** — https://github.com/nflverse/nflverse-data/releases/tag/pbp
   and https://github.com/nflverse/nflverse-data. Historical records underlying model training.
   The source repository identifies its license as **CC BY 4.0**:
   https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md.
   Credit: nflverse contributors. The included test records are an adapted subset: filtered,
   play categories transformed, and model predictions/residuals added. The original raw
   release files are not included. License: https://creativecommons.org/licenses/by/4.0/.
2. **nflreadpy** — https://nflreadpy.nflverse.com/ and
   https://nflreadpy.nflverse.com/api/load_functions/. Used for data loading and its Polars API.
3. **nflreadr play-by-play dictionary** — https://nflreadr.nflverse.com/articles/dictionary_pbp.html.
   Used to interpret field position, score margin, clocks, yardage and play categories.
4. **scikit-learn**:
   - https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html
   - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
   - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
   - https://scikit-learn.org/stable/modules/permutation_importance.html
   Model implementations, pipeline preprocessing and interpretation method.
5. **Streamlit** — https://docs.streamlit.io/ and
   https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app.
   Web UI, cached inference and hosting instructions.
6. **Plotly** — https://plotly.com/python/. Interactive plotting library.
7. **OpenAI Codex**, assisted work session on 26 October 2026. Generated and revised project
   code and tests; executed training and checks. This is an
   AI-assistance disclosure, not an independent scientific authority for model performance.
8. **HVL DAT158 assignment 2 brief** and **NFL Expected Yards Predictor project specification**,
   supplied in the task. Requirements and project scope. The linked course Word template
   was not incorporated into the generated documentation.

No pretrained yardage model or external solution's implementation is copied into this
repository. Estimator implementations come from scikit-learn; their parameters are
learned on the specified training seasons. Football field graphics are drawn in code.
