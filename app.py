"""Streamlit entry point: python -m streamlit run app.py."""
import json
import os
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "4")
import pandas as pd
import plotly.express as px
import streamlit as st
from nfl_yards.model import load_model, compare_play_types, validate_inputs
from nfl_yards.visuals import (
    LABELS, field_position_label, render_football_field, display_play_comparison,
    plot_yardage_distribution, plot_prediction_errors, plot_actual_predicted,
    plot_error_by_down, style_chart,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
st.set_page_config(page_title="ML × YARDS | NFL play explorer", page_icon="🏈", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1450px; padding-top: 2.3rem; padding-bottom: 3rem;}
h1 {font-size: clamp(2.2rem, 5vw, 4rem) !important; letter-spacing: -.045em; font-weight: 800 !important;}
h2, h3 {letter-spacing: -.025em;}
[data-testid="stMetricValue"] {color: #c9ff68; font-variant-numeric: tabular-nums;}
[data-testid="stMetric"] {border: 1px solid #2c3b40; border-radius: 12px; padding: 18px; background: #141f26;}
.eyebrow {color: #c9ff68; font-size: .76rem; letter-spacing: .19em; font-weight: 700;}
.intro {color: #a5b8be; font-size: 1.1rem; max-width: 800px; margin-bottom: 1.5rem;}
.small-note {color: #8fa4aa; font-size: .85rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_model(model_stamp):
    return load_model(ARTIFACTS / "model.joblib")


@st.cache_data
def get_results(metrics_stamp):
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(ARTIFACTS / "test_predictions.parquet")
    importance = pd.read_csv(ARTIFACTS / "feature_importance.csv")
    return metrics, predictions, importance


def update_field_position():
    """Keep a keyed distance widget feasible when the ball moves near the goal."""
    maximum = 100 - st.session_state.get("field_position", 25)
    st.session_state["distance"] = min(st.session_state.get("distance", 10), maximum)


def get_user_inputs():
    st.subheader("Set the situation")
    play_type = st.radio("Play choice", ["run", "pass"], horizontal=True,
                         format_func=lambda p: "Run" if p == "run" else "Pass / dropback")
    st.slider("Field position · yards from own goal", 1, 99, 25,
              key="field_position", on_change=update_field_position)
    position = 100 - st.session_state.field_position
    st.caption(f"Ball on: **{field_position_label(position)}** · {position} yards to the end zone")
    c1, c2 = st.columns(2)
    down = c1.selectbox("Down", [1, 2, 3, 4])
    if "distance" not in st.session_state:
        st.session_state.distance = 10
    distance = c2.number_input("Yards to go", 1, position, step=1, key="distance")
    score = st.slider("Offense score differential", -28, 28, 0)
    st.caption("Negative = trailing · Positive = leading")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    quarter = c1.selectbox("Quarter", [1, 2, 3, 4])
    minutes = c2.number_input("Minutes", 0, 15, 10)
    if minutes == 15:
        st.session_state["seconds"] = 0
    seconds = c3.number_input("Seconds", 0, 59, key="seconds", disabled=minutes == 15)
    situation = dict(play_type=play_type, down=int(down), ydstogo=int(distance), yardline_100=int(position),
                     score_differential=int(score), qtr=int(quarter), quarter_seconds_remaining=int(minutes * 60 + seconds))
    validate_inputs(situation)
    return situation


def display_prediction(situation, comparison):
    st.markdown('<div class="eyebrow">THE MODEL’S EXPECTATION</div>', unsafe_allow_html=True)
    selected = comparison[situation["play_type"]]
    columns = st.columns(3)
    columns[0].metric("Selected play · expected yards", f"{selected:.2f}")
    columns[1].metric("Run · expected yards", f"{comparison['run']:.2f}")
    columns[2].metric("Pass · expected yards", f"{comparison['pass']:.2f}")
    st.plotly_chart(render_football_field(situation, selected), width="stretch", config={"displayModeBar": False})
    distance_label = "goal" if situation["ydstogo"] == situation["yardline_100"] else str(situation["ydstogo"])
    st.caption(f"Down {situation['down']} & {distance_label} · {field_position_label(situation['yardline_100'])} · "
               "Blue: ball position · Gold: line to gain · Green: expected gain")
    st.plotly_chart(display_play_comparison(comparison), width="stretch", config={"displayModeBar": False})
    st.markdown(f"**Pass − run: {comparison['difference']:+.2f} yards** for this situation.")
    st.caption("An average across historical situations, not a guaranteed gain or a play-calling recommendation. "
               "Pass includes sacks and scrambles; run excludes kneels. Turnover risk is not modeled.")


def display_model_metrics(results, predictions, importance):
    st.subheader("How well does it generalize?")
    st.write("Candidates learned from 2021–2023. The lowest 2024 validation RMSE selected the model. "
             "Only that frozen model was evaluated on the 2025 test season.")
    winner = results["selected_model"]
    a, b, c = st.columns(3)
    final = results["test"][winner]
    a.metric("2025 test MAE", f"{final['MAE']:.2f} yd")
    b.metric("2025 test RMSE", f"{final['RMSE']:.2f} yd")
    c.metric("2025 test R²", f"{final['R2']:.3f}")
    st.caption("MAE: typical absolute error. RMSE: penalizes large misses. R²: improvement over predicting the test mean; 1 is perfect, 0 matches that baseline.")
    table = pd.DataFrame(results["validation"]).T.rename_axis("Model").reset_index()
    table["Selected"] = table.Model.eq(winner)
    st.markdown("#### Model comparison · 2024 validation")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "MAE": st.column_config.NumberColumn("MAE (yd)", format="%.3f"),
        "RMSE": st.column_config.NumberColumn("RMSE (yd)", format="%.3f"),
        "R2": st.column_config.NumberColumn("R²", format="%.3f")})
    with st.expander("Training scores and split sizes"):
        st.dataframe(pd.DataFrame(results["train"]).T, width="stretch")
        st.json(results["split_counts"])
    left, right = st.columns(2)
    with left:
        st.markdown("#### Actual vs. predicted · 2025")
        st.plotly_chart(plot_actual_predicted(predictions), width="stretch")
        st.caption("A reproducible sample of up to 3,000 test plays. A conditional mean cannot predict every explosive play.")
        st.markdown("#### Where are the errors?")
        st.plotly_chart(plot_prediction_errors(predictions), width="stretch")
        st.caption("Positive error means the model underestimated the gain. All test plays are included.")
    with right:
        st.markdown("#### The spread of outcomes · 2025")
        st.plotly_chart(plot_yardage_distribution(predictions), width="stretch")
        st.caption("Density is normalized within each play type to compare their shapes. Includes incomplete passes and losses.")
        st.markdown("#### Error by down · 2025")
        st.plotly_chart(plot_error_by_down(predictions), width="stretch")
    st.markdown("#### What does the model rely on?")
    chart_data = importance.copy()
    chart_data["feature"] = chart_data.feature.map(LABELS)
    fig = px.bar(chart_data.sort_values("rmse_increase"), x="rmse_increase", y="feature", error_x="std",
                 orientation="h", color_discrete_sequence=["#c9ff68"],
                 labels={"rmse_increase": "Increase in validation RMSE after shuffling (yards)", "feature": ""})
    st.plotly_chart(style_chart(fig, 350), width="stretch")
    st.caption("Permutation importance on up to 6,000 validation plays, five shuffles per feature. "
               "Bars show the average RMSE increase; whiskers show standard deviation. "
               "Correlated features can share importance. This is global model behavior, not a causal explanation of a single play.")
    with st.expander("Results by play type and download"):
        st.dataframe(pd.DataFrame(results["test_by_play_type"]).T)
        st.download_button("Download evaluation summary", json.dumps(results, indent=2), "metrics.json", "application/json")


st.markdown('<div class="eyebrow">NFL ANALYTICS / MACHINE LEARNING LAB</div>', unsafe_allow_html=True)
st.title("ML × YARDS")
st.markdown('<div class="intro">Every snap starts with a situation. Explore the yards that history expects.</div>', unsafe_allow_html=True)
required = ["model.joblib", "metrics.json", "test_predictions.parquet", "feature_importance.csv"]
missing = [name for name in required if not (ARTIFACTS / name).exists()]
if missing:
    st.info("Train the models to begin. Run `python train.py` from the repository, then refresh this page.")
    st.caption("Missing artifacts: " + ", ".join(missing))
    st.stop()
try:
    model = get_model((ARTIFACTS / "model.joblib").stat().st_mtime_ns)
    results, predictions, importance = get_results((ARTIFACTS / "metrics.json").stat().st_mtime_ns)
except Exception as error:
    st.error("The saved model or evaluation files could not be loaded. Install requirements.lock.txt and regenerate with python train.py.")
    st.caption(str(error))
    st.stop()

st.caption(f"2021–2025 NFL DATA  ·  {sum(results['split_counts'].values()):,} PLAYS  ·  MODEL: {results['selected_model'].upper()}")
explorer, analysis, methodology = st.tabs(["01 / Play explorer", "02 / Model analysis", "03 / Method & sources"])
with explorer:
    st.write("")
    controls, output = st.columns([1, 2.15], gap="large")
    with controls:
        with st.container(border=True):
            situation = get_user_inputs()
    with output:
        display_prediction(situation, compare_play_types(model, situation))
with analysis:
    display_model_metrics(results, predictions, importance)
with methodology:
    st.subheader("From play-by-play to expected yards")
    st.markdown("""
1. **Collect:** nflverse play-by-play, 2021–2025, via nflreadpy. Regular season and postseason.
2. **Clean:** keep valid regulation runs and dropbacks; exclude kneels, spikes, two-point attempts,
   no-play penalties, missing inputs and impossible field positions. Scrambles join dropbacks.
3. **Learn:** mean baseline, standardized Ridge, and two gradient boosting configurations.
   Inputs are play choice, down, distance, field position, score margin, quarter and quarter clock.
4. **Select:** compare candidates on 2024 RMSE. Squared-error training targets expected (mean) yards.
5. **Test:** evaluate the selected model once on 2025; keep the 2021–2023 fit for deployment.
6. **Explore:** this page loads the saved model. Changing an input runs inference without retraining.

**Reading the prediction.** A 5-yard expectation does not mean the next play will gain 5 yards.
Many passes gain zero and a few gain a lot. Similar situations can have very different outcomes.
The green field segment displays the mean; it is not a confidence interval.

**Limitations.** Teams choose plays strategically, so run/pass comparisons are observational.
Personnel, team strength, weather, formations, injuries, turnovers and play design are absent.
Pre-snap intent is approximated from recorded play labels; broken plays and option plays can be misclassified.
Rare fourth downs and extreme situations have less support. Overtime is outside the training scope.
Yards gained is the nflverse recorded target, not expected points or total penalty-adjusted field movement.
The UI restricts score margins to ±28; training retains other valid margins.

**Sources**
- [nflverse play-by-play releases](https://github.com/nflverse/nflverse-data/releases/tag/pbp)
- [nflreadpy documentation](https://nflreadpy.nflverse.com/)
- [nflverse data dictionary](https://nflreadr.nflverse.com/articles/dictionary_pbp.html)
- [scikit-learn](https://scikit-learn.org/stable/) · [Streamlit](https://docs.streamlit.io/) · [Plotly](https://plotly.com/python/)

Created for DAT158 by **Thomas Årland**. AI assistance is disclosed in the repository README and source acknowledgments.
    """)
    with st.expander("Data cleaning audit and reproducibility"):
        st.json(results["cleaning"])
        st.json(results["environment"])
st.divider()
st.markdown('<div class="small-note">ML × YARDS · DAT158 project · Historical expectations, one situation at a time.</div>', unsafe_allow_html=True)
