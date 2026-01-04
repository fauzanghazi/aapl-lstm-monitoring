# =========================================================
# Model Monitoring Dashboard
# =========================================================
import streamlit as st
import pandas as pd

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="AAPL Model Monitoring Dashboard",
    layout="wide",
)

st.title("AAPL Model Monitoring Dashboard")
st.write("Operational monitoring for deployed LSTM models")

# =========================================================
# Load monitoring data (LIVE FIRST, CSV FALLBACK)
# =========================================================

# 1️⃣ LIVE in-memory logs (from Prediction page)
if "monitoring_logs" in st.session_state and st.session_state.monitoring_logs:
    logs = pd.DataFrame(st.session_state.monitoring_logs)
    st.caption("Showing **live session monitoring data**")
else:
    # 2️⃣ Fallback to CSV (for evidence / screenshots)
    LOG_FILE = "logs/monitoring_logs.csv"
    try:
        logs = pd.read_csv(LOG_FILE, parse_dates=["timestamp"])
        st.caption("Showing **persisted monitoring logs (CSV)**")
    except FileNotFoundError:
        logs = pd.DataFrame()

if logs.empty:
    st.warning(
        "No monitoring data available. Run predictions and submit feedback first."
    )
    st.stop()

# =========================================================
# Summary Metrics
# =========================================================
st.subheader("Summary Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    avg_latency_v1 = logs["latency_v1_ms"].mean()
    avg_latency_v2 = logs["latency_v2_ms"].mean()

    st.metric("Avg Latency v1", f"{avg_latency_v1:.2f} ms")
    st.metric("Avg Latency v2", f"{avg_latency_v2:.2f} ms")

with col2:
    avg_feedback = logs["feedback_score"].mean()
    st.metric("Avg Feedback Score", f"{avg_feedback:.2f}")

with col3:
    st.metric("Total Prediction Runs", len(logs))

# =========================================================
# Latency Comparison Visualization
# =========================================================
st.subheader("Latency Comparison Between Models")

latency_compare = pd.DataFrame(
    {
        "Model": ["Baseline LSTM (v1)", "Improved LSTM (v2)"],
        "Average Latency (ms)": [
            logs["latency_v1_ms"].mean(),
            logs["latency_v2_ms"].mean(),
        ],
    }
)

st.bar_chart(latency_compare.set_index("Model"))

# =========================================================
# Feedback Score Distribution
# =========================================================
st.subheader("Feedback Score Distribution")

feedback_df = logs.dropna(subset=["feedback_score"])

if not feedback_df.empty:
    feedback_counts = (
        feedback_df["feedback_score"].value_counts().sort_index().reset_index()
    )
    feedback_counts.columns = ["Feedback Score", "Count"]

    st.bar_chart(feedback_counts.set_index("Feedback Score"))
else:
    st.info("No feedback scores submitted yet.")

# =========================================================
# Recent User Comments
# =========================================================
st.subheader("Recent User Comments")

comments = logs[
    logs["feedback_comment"].notna() & (logs["feedback_comment"].str.strip() != "")
]

if comments.empty:
    st.info("No user comments submitted yet.")
else:
    st.dataframe(
        comments.sort_values("timestamp", ascending=False)[
            ["timestamp", "feedback_score", "feedback_comment"]
        ].head(10),
        use_container_width=True,
    )

# =========================================================
# Raw Monitoring Logs
# =========================================================
st.subheader("Raw Monitoring Logs")
st.dataframe(logs, use_container_width=True)

# =========================================================
# Interpretation
# =========================================================
st.subheader("Interpretation")

st.markdown(
    """
    **Model Behaviour Insights:**

    - The dashboard compares prediction latency between the baseline and improved LSTM models.
    - Average latency values highlight computational efficiency differences across iterations.
    - User feedback scores provide qualitative assessment of prediction usefulness.
    - Logged comments reveal usability issues and guide future improvements.

    This monitoring view supports **Agile-based iterative refinement**, enabling rapid feedback
    loops and evidence-driven model improvement decisions.
    """
)
