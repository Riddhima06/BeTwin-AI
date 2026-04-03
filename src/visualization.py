import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(page_title="BeTwin-AI Monitor", layout="wide")

st.markdown(
    """
<style>
.stApp {
  background: radial-gradient(circle at top left, #111827 0%, #0b0f1a 45%, #070b12 100%);
  color: #f8fafc;
}

.block-container {
  padding-top: 1.5rem;
  padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
}

.metric-card {
  padding: 0.9rem 1rem;
  border-radius: 16px;
  background: rgba(15, 23, 42, 0.72);
  border: 1px solid rgba(148, 163, 184, 0.14);
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("BeTwin-AI Engine Digital Twin Dashboard")
st.write("Interactive health monitoring for NASA C-MAPSS FD001 engine data")


@st.cache_data(show_spinner=False)
def load_data():
    columns = ["engine_id", "cycle", "op1", "op2", "op3"]
    columns.extend(f"sensor_{index}" for index in range(1, 22))

    data = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None)
    data = data.dropna(axis=1)
    data.columns = columns
    return data


def build_engine_frame(data, engine_id, sensor_name, stress_factor):
    engine_frame = data.loc[data.engine_id == engine_id].copy()
    engine_frame.sort_values("cycle", inplace=True)
    engine_frame.reset_index(drop=True, inplace=True)

    engine_frame["stress_value"] = engine_frame[sensor_name] * stress_factor
    engine_frame["rul"] = int(engine_frame["cycle"].max()) - engine_frame["cycle"]

    window = min(8, max(3, len(engine_frame) // 10 or 3))
    rolling_mean = engine_frame[sensor_name].rolling(window=window, min_periods=1).mean()
    rolling_std = engine_frame[sensor_name].rolling(window=window, min_periods=1).std().fillna(0)

    engine_frame["sensor_roll"] = rolling_mean
    engine_frame["upper_band"] = rolling_mean + (2 * rolling_std)
    engine_frame["lower_band"] = rolling_mean - (2 * rolling_std)
    engine_frame["z_score"] = (engine_frame[sensor_name] - engine_frame[sensor_name].mean()) / max(
        engine_frame[sensor_name].std(ddof=0),
        1e-9,
    )
    engine_frame["anomaly"] = engine_frame["z_score"].abs() > 2.5
    return engine_frame


def build_forecast(engine_frame, sensor_name):
    x_values = engine_frame["cycle"].to_numpy()
    y_values = engine_frame[sensor_name].to_numpy()

    if len(x_values) < 3:
        return None

    span = max(5, len(x_values) // 3)
    recent_x = x_values[-span:]
    recent_y = y_values[-span:]

    slope, intercept = np.polyfit(recent_x, recent_y, 1)
    fitted = slope * recent_x + intercept
    residuals = recent_y - fitted
    spread = residuals.std(ddof=0) if len(residuals) > 1 else 0.0

    future_cycles = np.arange(int(x_values.max()) + 1, int(x_values.max()) + 21)
    forecast = slope * future_cycles + intercept
    confidence = 1.96 * spread

    return future_cycles, forecast, forecast - confidence, forecast + confidence


df = load_data()
sensor_columns = [column for column in df.columns if column.startswith("sensor_")]

with st.sidebar:
    st.title("Engine Controls")

    engine = st.selectbox("Select Engine", df.engine_id.unique())
    sensor = st.selectbox("Select Sensor", sensor_columns)
    stress = st.slider("Stress Simulation", 0.5, 2.0, 1.0, 0.05)

engine_df = build_engine_frame(df, engine, sensor, stress)
forecast_data = build_forecast(engine_df, sensor)

latest_value = float(engine_df[sensor].iloc[-1])
fleet_percentile = float((df[sensor] < latest_value).mean() * 100)
health_score = max(0.0, 100.0 - float(engine_df["rul"].iloc[-1]) / max(float(engine_df["cycle"].max()), 1.0) * 100.0)
anomaly_count = int(engine_df["anomaly"].sum())

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Selected Engine", f"#{engine}")
metric_col2.metric("Health Score", f"{health_score:.1f}%")
metric_col3.metric("Current Sensor", f"{latest_value:.3f}")
metric_col4.metric("Fleet Percentile", f"{fleet_percentile:.0f}th")

st.divider()

timeline_fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.72, 0.28],
    subplot_titles=("Sensor Trend and Stress Loading", "Deviation Score"),
)

timeline_fig.add_trace(
    go.Scatter(
        x=engine_df["cycle"],
        y=engine_df[sensor],
        mode="lines",
        name="Actual sensor",
        line=dict(color="#38bdf8", width=2),
    ),
    row=1,
    col=1,
)

timeline_fig.add_trace(
    go.Scatter(
        x=engine_df["cycle"],
        y=engine_df["sensor_roll"],
        mode="lines",
        name="Rolling average",
        line=dict(color="#f59e0b", width=3),
    ),
    row=1,
    col=1,
)

timeline_fig.add_trace(
    go.Scatter(
        x=engine_df["cycle"],
        y=engine_df["stress_value"],
        mode="lines",
        name="Stress adjusted",
        line=dict(color="#a78bfa", width=2, dash="dot"),
    ),
    row=1,
    col=1,
)

timeline_fig.add_trace(
    go.Scatter(
        x=engine_df["cycle"],
        y=engine_df["upper_band"],
        mode="lines",
        name="Upper band",
        line=dict(color="rgba(148,163,184,0.25)", width=0),
        showlegend=False,
    ),
    row=1,
    col=1,
)

timeline_fig.add_trace(
    go.Scatter(
        x=engine_df["cycle"],
        y=engine_df["lower_band"],
        mode="lines",
        name="Expected range",
        line=dict(color="rgba(148,163,184,0.25)", width=0),
        fill="tonexty",
        fillcolor="rgba(56,189,248,0.10)",
        showlegend=True,
    ),
    row=1,
    col=1,
)

timeline_fig.add_trace(
    go.Scatter(
        x=engine_df.loc[engine_df["anomaly"], "cycle"],
        y=engine_df.loc[engine_df["anomaly"], sensor],
        mode="markers",
        name="Anomaly",
        marker=dict(color="#f87171", size=10, symbol="x"),
    ),
    row=1,
    col=1,
)

timeline_fig.add_trace(
    go.Bar(
        x=engine_df["cycle"],
        y=engine_df["z_score"],
        name="Z-score",
        marker=dict(color=np.where(engine_df["anomaly"], "#f87171", "#60a5fa")),
        opacity=0.85,
    ),
    row=2,
    col=1,
)

timeline_fig.add_hline(y=0, line_width=1, line_color="rgba(226,232,240,0.35)", row=2, col=1)
timeline_fig.update_layout(
    template="plotly_dark",
    height=760,
    margin=dict(l=20, r=20, t=70, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
timeline_fig.update_xaxes(title_text="Cycle", row=2, col=1)
timeline_fig.update_yaxes(title_text="Sensor Value", row=1, col=1)
timeline_fig.update_yaxes(title_text="Z-score", row=2, col=1)

st.subheader("Engine Health Timeline")
st.plotly_chart(timeline_fig, use_container_width=True)

rul_fig = go.Figure()
rul_fig.add_trace(
    go.Scatter(
        x=engine_df["cycle"],
        y=engine_df["rul"],
        mode="lines",
        name="Actual RUL",
        line=dict(color="#22c55e", width=3),
    )
)
rul_fig.add_hrect(y0=0, y1=10, fillcolor="rgba(248,113,113,0.22)", line_width=0)
rul_fig.add_hrect(y0=10, y1=25, fillcolor="rgba(245,158,11,0.18)", line_width=0)
rul_fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(248,113,113,0.55)")
rul_fig.update_layout(
    template="plotly_dark",
    height=430,
    margin=dict(l=20, r=20, t=50, b=20),
    yaxis_title="Remaining Useful Life",
    xaxis_title="Cycle",
)

st.subheader("Remaining Useful Life")
st.plotly_chart(rul_fig, use_container_width=True)

correlation_fig = go.Figure(
    data=go.Heatmap(
        z=engine_df[sensor_columns].corr().values,
        x=sensor_columns,
        y=sensor_columns,
        colorscale="Viridis",
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Corr"),
    )
)
correlation_fig.update_layout(
    template="plotly_dark",
    height=560,
    margin=dict(l=20, r=20, t=50, b=20),
)

st.subheader("Sensor Correlation Map")
st.plotly_chart(correlation_fig, use_container_width=True)

recent_window = engine_df[sensor_columns].tail(min(len(engine_df), 40)).copy()
recent_window.index = engine_df["cycle"].tail(len(recent_window)).astype(int)
recent_window = (recent_window - recent_window.mean()) / recent_window.std(ddof=0).replace(0, 1)
recent_window = recent_window.fillna(0)

operational_fig = go.Figure(
    data=go.Heatmap(
        z=recent_window.T.values,
        x=recent_window.index.astype(str),
        y=sensor_columns,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="Std"),
    )
)
operational_fig.update_layout(
    template="plotly_dark",
    height=560,
    margin=dict(l=20, r=20, t=50, b=20),
)

st.subheader("Recent Operational Signature")
st.plotly_chart(operational_fig, use_container_width=True)

if forecast_data is not None:
    future_cycles, forecast, lower_bound, upper_bound = forecast_data

    prediction_fig = go.Figure()
    prediction_fig.add_trace(
        go.Scatter(
            x=engine_df["cycle"],
            y=engine_df[sensor],
            mode="lines+markers",
            name="Observed",
            line=dict(color="#38bdf8", width=2),
        )
    )
    prediction_fig.add_trace(
        go.Scatter(
            x=future_cycles,
            y=forecast,
            mode="lines",
            name="Projected",
            line=dict(color="#f59e0b", width=3, dash="dash"),
        )
    )
    prediction_fig.add_trace(
        go.Scatter(
            x=np.concatenate([future_cycles, future_cycles[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill="toself",
            fillcolor="rgba(245,158,11,0.16)",
            line=dict(color="rgba(245,158,11,0)"),
            name="Confidence band",
            showlegend=True,
        )
    )
    prediction_fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Cycle",
        yaxis_title=sensor,
    )

    st.subheader("Future Sensor Projection")
    st.plotly_chart(prediction_fig, use_container_width=True)

st.info(f"Detected {anomaly_count} likely anomaly points in the selected engine profile.")
st.success("Digital Twin Simulation Running Successfully")