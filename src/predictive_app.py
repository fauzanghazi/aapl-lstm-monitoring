import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import json
import plotly.graph_objects as go
import os

from log_utils import log_prediction

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="AAPL Price Prediction", layout="centered")

st.title("AAPL Next-Day Price Prediction")
st.write("Compare baseline LSTM (v1) and improved LSTM (v2)")

# -----------------------------
# Session state
# -----------------------------
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "last_date" not in st.session_state:
    st.session_state.last_date = None


# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/aapl_2000_2025.csv", parse_dates=["Date"])
    return df.sort_values("Date")


df = load_data()


# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


model_v1 = load_model("models/model_v1.pkl")
model_v2 = load_model("models/model_v2.pkl")


# -----------------------------
# Load metrics
# -----------------------------
@st.cache_data
def load_metrics():
    path = "metrics/model_metrics_2025.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


metrics = load_metrics()

# -----------------------------
# User input
# -----------------------------
st.subheader("Input Parameters")

selected_date = st.date_input("Select last available date", value=df["Date"].max())

# Reset prediction if date changes
if st.session_state.last_date != selected_date:
    st.session_state.prediction_done = False

if selected_date not in df["Date"].dt.date.values:
    st.error("Selected date not found in dataset")
    st.stop()


# -----------------------------
# Input preparation
# -----------------------------
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


def rolling_next_day_forecast(df, end_date, model_dict, version, lookback=60):
    preds = []
    dates = []

    sub_df = df[df["Date"] <= pd.to_datetime(end_date)].tail(lookback + 1)

    for i in range(len(sub_df) - 1):
        current_date = sub_df.iloc[i]["Date"]

        if version == "v1":
            X = prepare_input_v1(df, current_date, model_dict)
            p_scaled = model_dict["model"].predict(X, verbose=0)
            p = model_dict["scaler"].inverse_transform(p_scaled)[0][0]
        else:
            X = prepare_input_v2(df, current_date, model_dict)
            p_scaled = model_dict["model"].predict(X, verbose=0)
            p = model_dict["scaler"].inverse_transform(
                np.hstack([p_scaled, np.zeros((1, 2))])
            )[0][0]

        preds.append(p)
        dates.append(sub_df.iloc[i + 1]["Date"])

    return pd.DataFrame({"Date": dates, "Forecast": preds})


# -----------------------------
# Run prediction
# -----------------------------
if st.button("Run Prediction"):
    st.session_state.prediction_done = True
    st.session_state.last_date = selected_date

    # Model v1
    start_v1 = time.time()
    X1 = prepare_input_v1(df, selected_date, model_v1)
    p1_scaled = model_v1["model"].predict(X1, verbose=0)
    st.session_state.latency_v1 = (time.time() - start_v1) * 1000
    st.session_state.pred_v1 = model_v1["scaler"].inverse_transform(p1_scaled)[0][0]

    # Model v2
    start_v2 = time.time()
    X2 = prepare_input_v2(df, selected_date, model_v2)
    p2_scaled = model_v2["model"].predict(X2, verbose=0)
    st.session_state.latency_v2 = (time.time() - start_v2) * 1000
    st.session_state.pred_v2 = model_v2["scaler"].inverse_transform(
        np.hstack([p2_scaled, np.zeros((1, 2))])
    )[0][0]

# -----------------------------
# Display results
# -----------------------------
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

    # -----------------------------
    # Weekly Actual vs Forecast chart
    # -----------------------------
    st.subheader("Actual vs Forecast (Weekly, Rolling Next-Day)")

    lookback_days = 90
    history = df[df["Date"] <= pd.to_datetime(selected_date)].tail(lookback_days)

    v1_df = rolling_next_day_forecast(df, selected_date, model_v1, "v1", lookback_days)
    v2_df = rolling_next_day_forecast(df, selected_date, model_v2, "v2", lookback_days)

    actual_weekly = history.set_index("Date")["Close"].resample("W").mean()
    v1_weekly = v1_df.set_index("Date")["Forecast"].resample("W").mean()
    v2_weekly = v2_df.set_index("Date")["Forecast"].resample("W").mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual_weekly.index,
            y=actual_weekly.values,
            mode="lines",
            name="Actual",
            line=dict(color="#E0E0E0", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=v1_weekly.index,
            y=v1_weekly.values,
            mode="lines",
            name="Model v1 Forecast",
            line=dict(color="#FFA500", dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=v2_weekly.index,
            y=v2_weekly.values,
            mode="lines",
            name="Model v2 Forecast",
            line=dict(color="#00E5FF", dash="dash"),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Week",
        yaxis_title="Price",
        legend_title="Series",
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Model v1 reacts faster to price changes but is more volatile. "
        "Model v2 produces smoother forecasts but lags during trend shifts."
    )

    # -----------------------------
    # Metrics (collapsed)
    # -----------------------------
    if metrics:
        with st.expander("Model Accuracy Metrics (2025 Holdout)", expanded=False):
            st.table(pd.DataFrame(metrics).T)

    # -----------------------------
    # User feedback (collapsed)
    # -----------------------------
    with st.expander("User Feedback", expanded=False):
        feedback_score = st.slider(
            "Prediction usefulness (1 = poor, 5 = excellent)", 1, 5, 3
        )

        feedback_comment = st.text_area("Comments")

        if st.button("Submit Feedback"):
            log_prediction(
                model_version="v1",
                prediction=st.session_state.pred_v1,
                latency_ms=st.session_state.latency_v1,
                feedback_score=feedback_score,
                feedback_comment=feedback_comment,
            )

            log_prediction(
                model_version="v2",
                prediction=st.session_state.pred_v2,
                latency_ms=st.session_state.latency_v2,
                feedback_score=feedback_score,
                feedback_comment=feedback_comment,
            )

            st.success("Feedback logged successfully")
