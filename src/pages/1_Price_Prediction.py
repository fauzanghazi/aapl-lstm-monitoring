# =========================================================
# Imports & setup
# =========================================================
import os
import uuid
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import json
import plotly.graph_objects as go

from log_utils import log_prediction

# Ensure logs directory exists (safe locally)
os.makedirs("logs", exist_ok=True)

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(page_title="AAPL Price Prediction", layout="centered")

st.title("AAPL Next-Day Price Prediction")
st.write("Comparison between baseline LSTM (v1) and improved LSTM (v2)")

# =========================================================
# Session state initialization (CRITICAL)
# =========================================================
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "last_date" not in st.session_state:
    st.session_state.last_date = None

if "session_hash" not in st.session_state:
    st.session_state.session_hash = uuid.uuid4().hex[:8]

if "run_id" not in st.session_state:
    st.session_state.run_id = None

# 🔥 LIVE MONITORING STORE
if "monitoring_logs" not in st.session_state:
    st.session_state.monitoring_logs = []


# =========================================================
# Data loading
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/aapl_2000_2025.csv", parse_dates=["Date"])
    return df.sort_values("Date")


df = load_data()


# =========================================================
# Model loading
# =========================================================
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


model_v1 = load_model("models/model_v1.pkl")
model_v2 = load_model("models/model_v2.pkl")


# =========================================================
# Metrics loading
# =========================================================
@st.cache_data
def load_metrics():
    path = "metrics/model_metrics_2025.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


metrics = load_metrics()


# =========================================================
# Feature preparation
# =========================================================
def prepare_input_v1(df, date, model_dict):
    window = model_dict["window_size"]
    scaler = model_dict["scaler"]
    data = df[df["Date"] <= pd.to_datetime(date)]["Close"].values[-window:]
    scaled = scaler.transform(data.reshape(-1, 1))
    return scaled.reshape(1, window, 1)


def prepare_input_v2(df, date, model_dict):
    window = model_dict["window_size"]
    scaler = model_dict["scaler"]
    temp = df.copy()
    temp["Return"] = temp["Close"].pct_change()
    temp = temp.dropna()
    features = temp[temp["Date"] <= pd.to_datetime(date)][
        ["Close", "Return", "Volume"]
    ].values[-window:]
    scaled = scaler.transform(features)
    return scaled.reshape(1, window, scaled.shape[1])


# =========================================================
# Rolling next-day forecast (RESTORED)
# =========================================================
def rolling_next_day_forecast(df, end_date, model_dict, version, lookback=60):
    forecasts = []
    dates = []

    sub_df = df[df["Date"] <= pd.to_datetime(end_date)].tail(lookback + 1)

    for i in range(len(sub_df) - 1):
        current_date = sub_df.iloc[i]["Date"]

        if version == "v1":
            X = prepare_input_v1(df, current_date, model_dict)
            pred_scaled = model_dict["model"].predict(X, verbose=0)
            pred = model_dict["scaler"].inverse_transform(pred_scaled)[0][0]
        else:
            X = prepare_input_v2(df, current_date, model_dict)
            pred_scaled = model_dict["model"].predict(X, verbose=0)
            pred = model_dict["scaler"].inverse_transform(
                np.hstack([pred_scaled, np.zeros((1, 2))])
            )[0][0]

        forecasts.append(pred)
        dates.append(sub_df.iloc[i + 1]["Date"])

    return pd.DataFrame({"Date": dates, "Forecast": forecasts})


# =========================================================
# User input
# =========================================================
st.subheader("Input Parameters")

selected_date = st.date_input("Select last available date", value=df["Date"].max())

if st.session_state.last_date != selected_date:
    st.session_state.prediction_done = False

valid_dates = df["Date"].dt.date.values
if selected_date not in valid_dates:
    adjusted_date = df[df["Date"].dt.date < selected_date]["Date"].dt.date.max()
    st.warning(
        f"Selected date is not a trading day. "
        f"Adjusted to nearest available date: {adjusted_date}"
    )
    selected_date = adjusted_date

# =========================================================
# Run prediction
# =========================================================
if st.button("Run Prediction", key="run_prediction_btn"):
    st.session_state.prediction_done = True
    st.session_state.last_date = selected_date
    st.session_state.run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Model v1
    start = time.time()
    X1 = prepare_input_v1(df, selected_date, model_v1)
    p1_scaled = model_v1["model"].predict(X1, verbose=0)
    st.session_state.latency_v1 = (time.time() - start) * 1000
    st.session_state.pred_v1 = model_v1["scaler"].inverse_transform(p1_scaled)[0][0]

    # Model v2
    start = time.time()
    X2 = prepare_input_v2(df, selected_date, model_v2)
    p2_scaled = model_v2["model"].predict(X2, verbose=0)
    st.session_state.latency_v2 = (time.time() - start) * 1000
    st.session_state.pred_v2 = model_v2["scaler"].inverse_transform(
        np.hstack([p2_scaled, np.zeros((1, 2))])
    )[0][0]

# =========================================================
# Prediction results
# =========================================================
if st.session_state.prediction_done:
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model v1 Prediction",
            f"${st.session_state.pred_v1:.2f}",
            help=f"Latency: {st.session_state.latency_v1:.2f} ms",
        )

    with col2:
        st.metric(
            "Model v2 Prediction",
            f"${st.session_state.pred_v2:.2f}",
            help=f"Latency: {st.session_state.latency_v2:.2f} ms",
        )

# =========================================================
# Actual vs Forecast Line Chart (RESTORED)
# =========================================================
if st.session_state.prediction_done:
    st.subheader("Actual vs Forecast (Weekly, Rolling Next-Day)")

    lookback_days = 60

    history = (
        df[df["Date"] <= pd.to_datetime(selected_date)]
        .tail(lookback_days + 1)
        .set_index("Date")
        .resample("W")
        .last()
    )

    df_v1 = (
        rolling_next_day_forecast(df, selected_date, model_v1, "v1", lookback_days)
        .set_index("Date")
        .resample("W")
        .last()
        .rename(columns={"Forecast": "Model v1 Forecast"})
    )

    df_v2 = (
        rolling_next_day_forecast(df, selected_date, model_v2, "v2", lookback_days)
        .set_index("Date")
        .resample("W")
        .last()
        .rename(columns={"Forecast": "Model v2 Forecast"})
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=history.index, y=history["Close"], name="Actual"))
    fig.add_trace(
        go.Scatter(
            x=df_v1.index,
            y=df_v1["Model v1 Forecast"],
            name="Model v1 Forecast",
            line=dict(dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_v2.index,
            y=df_v2["Model v2 Forecast"],
            name="Model v2 Forecast",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Week",
        yaxis_title="Price",
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Feedback logging (CSV + LIVE MEMORY)
# =========================================================
if st.session_state.prediction_done:
    give_feedback = st.checkbox("Give feedback on this prediction")

    if give_feedback:
        with st.form("feedback_form", clear_on_submit=True):
            score = st.slider("Prediction usefulness", 1, 5, 3)
            comment = st.text_area("Comments")
            submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            log_prediction(
                run_id=st.session_state.run_id,
                session_hash=st.session_state.session_hash,
                model_version="comparison",
                prediction_v1=float(st.session_state.pred_v1),
                prediction_v2=float(st.session_state.pred_v2),
                latency_v1_ms=float(st.session_state.latency_v1),
                latency_v2_ms=float(st.session_state.latency_v2),
                feedback_score=int(score),
                feedback_comment=comment,
            )

            st.session_state.monitoring_logs.append(
                {
                    "timestamp": pd.Timestamp.utcnow(),
                    "run_id": st.session_state.run_id,
                    "session_hash": st.session_state.session_hash,
                    "prediction_v1": float(st.session_state.pred_v1),
                    "prediction_v2": float(st.session_state.pred_v2),
                    "latency_v1_ms": float(st.session_state.latency_v1),
                    "latency_v2_ms": float(st.session_state.latency_v2),
                    "feedback_score": int(score),
                    "feedback_comment": comment,
                }
            )

            st.success("Feedback submitted successfully")
