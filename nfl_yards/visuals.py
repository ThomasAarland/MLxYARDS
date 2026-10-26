"""Plotly views use real inference and evaluation artifacts."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLORS = {"run": "#c9ff68", "pass": "#73d5ef"}
LABELS = {"play_type": "Play choice", "down": "Down", "ydstogo": "Yards to go",
          "yardline_100": "Field position", "score_differential": "Score differential",
          "qtr": "Quarter", "quarter_seconds_remaining": "Quarter clock"}


def style_chart(fig, height=340):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color="#dce6e8",
                      margin=dict(l=20, r=20, t=35, b=35), legend_title_text="")
    fig.update_xaxes(gridcolor="#23333c", zerolinecolor="#38505d")
    fig.update_yaxes(gridcolor="#23333c", zerolinecolor="#38505d")
    return fig


def field_position_label(yardline_100):
    if yardline_100 == 50:
        return "Midfield"
    return f"Own {100 - yardline_100}" if yardline_100 > 50 else f"Opponent {yardline_100}"


def render_football_field(situation: dict, prediction: float):
    los = 100 - situation["yardline_100"]
    first = los + situation["ydstogo"]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-10, x1=110, y0=0, y1=53.3, fillcolor="#123a2d", line_color="#b8d6c6")
    for x in range(0, 100, 10):
        if x % 20 == 0:
            fig.add_shape(type="rect", x0=x, x1=x + 10, y0=0, y1=53.3, fillcolor="#174634", line_width=0)
    for x in range(0, 101, 5):
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=53.3,
                      line=dict(color="rgba(224,240,229,0.45)", width=1))
    for x in range(1, 100):
        for y in (21.9, 30.4):
            fig.add_shape(type="line", x0=x, x1=x, y0=y, y1=y+1, line=dict(color="#8aa999", width=1))
    for x in range(10, 100, 10):
        for y in (8, 45.3):
            fig.add_annotation(x=x, y=y, text=str(min(x, 100-x)), showarrow=False,
                               font=dict(size=17, color="#b9cfbf"))
    for x, label in [(-5, "OWN"), (105, "GOAL")]:
        fig.add_annotation(x=x, y=26.65, text=label, textangle=-90, showarrow=False,
                           font=dict(size=15, color="#c9dbd0"))
    fig.add_trace(go.Scatter(x=[los, los], y=[0, 53.3], mode="lines", name="Line of scrimmage",
                             line=dict(color="#73d5ef", width=3), hoverinfo="name"))
    fig.add_trace(go.Scatter(x=[first, first], y=[0, 53.3], mode="lines", name="First down / goal",
                             line=dict(color="#ffd475", width=3, dash="dash"), hoverinfo="name"))
    # Clip the drawing to the field, while the numeric prediction stays unmodified.
    end = float(np.clip(los + prediction, 0, 100))
    fig.add_trace(go.Scatter(x=[los, end], y=[26.65, 26.65], mode="lines+markers",
                             line=dict(color="#c9ff68", width=9), marker=dict(size=[9, 14]),
                             name="Expected gain", hovertemplate=f"Expected gain: {prediction:.2f} yd<extra></extra>"))
    fig.add_annotation(x=50, y=59, text="OFFENSE  →", showarrow=False, font=dict(size=12, color="#c9ff68"))
    # Keep turf and markings behind the live traces (Plotly shapes default above).
    fig.update_shapes(layer="below")
    style_chart(fig, 350)
    fig.update_layout(margin=dict(l=0, r=0, t=5, b=0), legend=dict(orientation="h", y=-0.05, x=0),
                      dragmode=False)
    fig.update_xaxes(range=[-11, 111], visible=False, fixedrange=True)
    fig.update_yaxes(range=[-1, 64], visible=False, fixedrange=True)
    return fig


def display_play_comparison(predictions):
    fig = go.Figure(go.Bar(y=["Run", "Pass / dropback"], x=[predictions["run"], predictions["pass"]],
                           orientation="h", marker_color=list(COLORS.values()),
                           text=[f"{predictions[p]:.2f} yd" for p in ("run", "pass")], textposition="auto",
                           hovertemplate="%{y}: %{x:.2f} yards<extra></extra>"))
    fig.update_layout(xaxis_title="Expected yards gained", showlegend=False)
    return style_chart(fig, 200)


def plot_yardage_distribution(data):
    fig = px.histogram(data, x="yards_gained", color="play_type", histnorm="probability density",
                       barmode="overlay", opacity=0.7, nbins=100, color_discrete_map=COLORS,
                       labels={"yards_gained": "Actual yards gained", "play_type": "Play choice"})
    return style_chart(fig)


def plot_prediction_errors(data):
    fig = px.histogram(data, x="residual", color="play_type", barmode="overlay", opacity=0.7,
                       nbins=100, color_discrete_map=COLORS,
                       labels={"residual": "Error: actual − predicted yards", "play_type": "Play choice"})
    fig.add_vline(x=0, line_dash="dash", line_color="#eef3ed")
    return style_chart(fig)


def plot_actual_predicted(data):
    sample = data.sample(min(3000, len(data)), random_state=42)
    fig = px.scatter(sample, x="yards_gained", y="predicted_yards", color="play_type", opacity=0.25,
                     color_discrete_map=COLORS, labels={"yards_gained": "Actual yards", "predicted_yards": "Predicted yards"})
    lo, hi = float(data.yards_gained.min()), float(data.yards_gained.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(color="#8b9a9f", dash="dash")))
    return style_chart(fig)


def plot_error_by_down(data):
    grouped = (data.assign(absolute_error=data.residual.abs()).groupby(["down", "play_type"])
               .absolute_error.mean().reset_index())
    fig = px.bar(grouped, x="down", y="absolute_error", color="play_type", barmode="group",
                 color_discrete_map=COLORS, labels={"down": "Down", "absolute_error": "Mean absolute error (yards)"})
    fig.update_xaxes(dtick=1)
    return style_chart(fig)
