import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page configuration
st.set_page_config(page_title="BeTwin-AI Monitor", layout="wide")

# -------------------------
# Dark Theme Styling
# -------------------------
st.markdown("""
<style>

.stApp{
  background-color:#0b0f1a;
  color:white;
}

.block-container{
  padding-top:2rem;
}

.css-1d391kg{
  background-color:#121826;
}

</style>
""", unsafe_allow_html=True)

st.title("⚙ BeTwin-AI Engine Digital Twin Dashboard")
st.write("Aircraft engine monitoring using NASA Turbofan FD001 dataset")

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():

  cols = ['engine_id','cycle','op1','op2','op3']

  for i in range(1,22):
    cols.append(f'sensor_{i}')

  df = pd.read_csv("data/train_FD001.txt", sep=" ", header=None)

  df = df.dropna(axis=1)

  df.columns = cols

  return df


df = load_data()

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.title("Engine Controls")

engine = st.sidebar.selectbox(
  "Select Engine",
  df.engine_id.unique()
)

sensor = st.sidebar.selectbox(
  "Select Sensor",
  [f"sensor_{i}" for i in range(1,22)]
)

stress = st.sidebar.slider(
  "Stress Simulation",
  0.5,
  2.0,
  1.0
)

engine_df = df[df.engine_id == engine]

engine_df["stress_value"] = engine_df[sensor] * stress

# -------------------------
# KPI Metrics
# -------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
  "Engine ID",
  engine
)

col2.metric(
  "Total Cycles",
  int(engine_df.cycle.max())
)

col3.metric(
  "Average Sensor",
  round(engine_df[sensor].mean(),2)
)

col4.metric(
  "Max Sensor",
  round(engine_df[sensor].max(),2)
)

st.divider()

# -------------------------
# Sensor Behaviour Graph
# -------------------------
st.subheader("Sensor Behaviour")

fig = px.line(
  engine_df,
  x="cycle",
  y=[sensor,"stress_value"],
  template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Remaining Useful Life
# -------------------------
st.subheader("Remaining Useful Life")

max_cycle = engine_df.cycle.max()

engine_df["RUL"] = max_cycle - engine_df["cycle"]

fig2 = px.line(
  engine_df,
  x="cycle",
  y="RUL",
  template="plotly_dark"
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Anomaly Detection
# -------------------------
st.subheader("Anomaly Detection")

mean = engine_df[sensor].mean()
std = engine_df[sensor].std()

engine_df["anomaly"] = engine_df[sensor] > mean + 2 * std

fig3 = px.scatter(
  engine_df,
  x="cycle",
  y=sensor,
  color="anomaly",
  template="plotly_dark"
)

st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Future Prediction
# -------------------------
st.subheader("Future Prediction")

x = engine_df["cycle"].values
y = engine_df[sensor].values

coef = np.polyfit(x, y, 1)
poly = np.poly1d(coef)

future_cycles = np.arange(max(x), max(x) + 20)

prediction = poly(future_cycles)

fig4 = px.line(template="plotly_dark")

fig4.add_scatter(
  x=x,
  y=y,
  mode="lines",
  name="Actual"
)

fig4.add_scatter(
  x=future_cycles,
  y=prediction,
  mode="lines",
  name="Prediction"
)

st.plotly_chart(fig4, use_container_width=True)

st.success("Digital Twin Simulation Running Successfully")