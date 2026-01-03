import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import json

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
    with open("metrics/model_metrics_2025.json", "r") as f:
        return json.load(f)


metrics = load_metrics()

# -----------------------------
# User input
# -----------------------------
st.subheader("Input Parameters")

selected_date = st.date_input("Select last available date", df["Date"].max())

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


# -----------------------------
# Run prediction
# -----------------------------
if st.button("Run Prediction"):
    st.session_state.prediction_done = True
    st.session_state.last_date = selected_date

    # Model v1
    start_v1 = time.time()
    X_v1 = prepare_input_v1(df, selected_date, model_v1)
    pred_v1_scaled = model_v1["model"].predict(X_v1, verbose=0)
    st.session_state.latency_v1 = (time.time() - start_v1) * 1000
    st.session_state.pred_v1 = model_v1["scaler"].inverse_transform(pred_v1_scaled)[0][
        0
    ]

    # Model v2
    start_v2 = time.time()
    X_v2 = prepare_input_v2(df, selected_date, model_v2)
    pred_v2_scaled = model_v2["model"].predict(X_v2, verbose=0)
    st.session_state.latency_v2 = (time.time() - start_v2) * 1000

    st.session_state.pred_v2 = model_v2["scaler"].inverse_transform(
        np.hstack([pred_v2_scaled, np.zeros((1, 2))])
    )[0][0]

# -----------------------------
# Display results
# -----------------------------
if st.session_state.prediction_done:
    st.subheader("Prediction Results")

    st.metric("Model v1 Prediction", f"${st.session_state.pred_v1:.2f}")
    st.metric("Model v2 Prediction", f"${st.session_state.pred_v2:.2f}")

    st.write(f"v1 latency: {st.session_state.latency_v1:.2f} ms")
    st.write(f"v2 latency: {st.session_state.latency_v2:.2f} ms")

    # -----------------------------
    # Price trend
    # -----------------------------
    st.subheader("Recent Price Trend")

    lookback_days = 90
    history = df[df["Date"] <= pd.to_datetime(selected_date)].tail(lookback_days)
    chart_df = history[["Date", "Close"]].set_index("Date")

    st.line_chart(chart_df)

    st.caption(
        "Line chart shows the last 90 trading days of AAPL closing prices. "
        "Predictions represent the estimated next-day closing price."
    )

    # -----------------------------
    # Accuracy metrics
    # -----------------------------
    st.subheader("Model Accuracy Comparison (2025 Holdout)")

    metrics_df = pd.DataFrame({"Model v1": metrics["v1"], "Model v2": metrics["v2"]})

    st.table(metrics_df)

    st.caption(
        "Accuracy metrics are computed using rolling next-day predictions "
        "on the 2025 holdout period. Lower values indicate better performance."
    )

    # -----------------------------
    # User feedback
    # -----------------------------
    st.subheader("User Feedback")

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
