import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")

st.title("AAPL Model Monitoring Dashboard")
st.write("Operational monitoring for deployed LSTM models")

LOG_FILE = "logs/monitoring_logs.csv"


# -----------------------------
# Load logs
# -----------------------------
@st.cache_data
def load_logs():
    try:
        df = pd.read_csv(LOG_FILE, parse_dates=["timestamp"])
        return df
    except FileNotFoundError:
        return pd.DataFrame()


logs = load_logs()

if logs.empty:
    st.warning("No monitoring logs found. Run predictions first.")
    st.stop()

# -----------------------------
# Summary metrics
# -----------------------------
st.subheader("Summary Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    avg_latency = logs.groupby("model_version")["latency_ms"].mean()
    st.metric("Avg Latency v1", f"{avg_latency.get('v1', 0):.2f} ms")
    st.metric("Avg Latency v2", f"{avg_latency.get('v2', 0):.2f} ms")

with col2:
    avg_feedback = logs.groupby("model_version")["feedback_score"].mean()
    st.metric("Avg Feedback v1", f"{avg_feedback.get('v1', 0):.2f}")
    st.metric("Avg Feedback v2", f"{avg_feedback.get('v2', 0):.2f}")

with col3:
    st.metric("Total Predictions", len(logs))

# -----------------------------
# Latency comparison
# -----------------------------
st.subheader("Latency Comparison")

latency_df = logs[["model_version", "latency_ms"]]
st.bar_chart(latency_df.groupby("model_version").mean())

# -----------------------------
# Feedback distribution
# -----------------------------
st.subheader("Feedback Score Distribution")

feedback_df = logs.dropna(subset=["feedback_score"])
st.bar_chart(
    feedback_df.groupby(["model_version", "feedback_score"])
    .size()
    .unstack(fill_value=0)
)

# -----------------------------
# Recent comments
# -----------------------------
st.subheader("Recent User Comments")

comments = logs[
    logs["feedback_comment"].notna() & (logs["feedback_comment"].str.strip() != "")
]

st.dataframe(
    comments.sort_values("timestamp", ascending=False)[
        ["timestamp", "model_version", "feedback_score", "feedback_comment"]
    ].head(10),
    use_container_width=True,
)

# -----------------------------
# Raw logs
# -----------------------------
st.subheader("Raw Monitoring Logs")
st.dataframe(logs, use_container_width=True)
