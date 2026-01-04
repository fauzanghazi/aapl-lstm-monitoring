import streamlit as st

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="AAPL LSTM System Overview",
    layout="wide",
)

# =========================================================
# Header section
# =========================================================
col1, col2 = st.columns([1, 5])

with col1:
    st.markdown(
        """
        <img src="https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg"
             style="width:80px; filter: invert(65%) grayscale(100%); opacity:0.85;">
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.title("AAPL LSTM Prediction & Monitoring System")
    st.caption(
        "End-to-end data science application using Agile and iterative development principles"
    )

st.markdown("---")

# =========================================================
# System Overview
# =========================================================
st.subheader("System Overview")

st.write(
    """
    This application demonstrates a **complete end-to-end data science lifecycle**
    built and deployed using **Agile-based iterative development**.

    The system showcases how predictive models evolve across iterations and how
    operational monitoring supports continuous improvement.
    """
)

st.markdown(
    """
    **Core Components:**
    - Baseline LSTM model (Model v1)
    - Improved LSTM model (Model v2)
    - Interactive Streamlit prediction interface
    - Latency measurement for model inference
    - User feedback collection and logging
    - Monitoring dashboard for deployed model behaviour
    """
)

# =========================================================
# Key Capabilities
# =========================================================
st.subheader("Key Capabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔮 Prediction")
    st.write(
        """
        - Next-day AAPL stock price prediction  
        - Side-by-side comparison of Model v1 and Model v2  
        - Real-time inference latency measurement
        """
    )

with col2:
    st.markdown("### 📝 Feedback & Logging")
    st.write(
        """
        - User usefulness rating (1–5 scale)  
        - Free-text qualitative feedback  
        - Persistent CSV-based logging for audit and evidence
        """
    )

with col3:
    st.markdown("### 📊 Monitoring")
    st.write(
        """
        - Latency comparison across model versions  
        - Feedback score analysis  
        - Recent user comments  
        - Full operational log inspection
        """
    )

st.markdown("---")

# =========================================================
# Live Market Context
# =========================================================
st.subheader("Live Market Context (AAPL)")

st.write(
    """
    The live market chart below provides real-world context for interpreting
    model predictions relative to recent AAPL price movements.
    """
)

tradingview_html = """
<iframe
    src="https://s.tradingview.com/widgetembed/?symbol=NASDAQ:AAPL
    &interval=D
    &theme=dark
    &style=1
    &locale=en
    &hide_side_toolbar=true
    &allow_symbol_change=false"
    style="width:100%; height:520px;"
    frameborder="0"
    allowtransparency="true"
    scrolling="no">
</iframe>
"""

st.components.v1.html(tradingview_html, height=540)

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption(
    "Academic project for MRTB2173 Agile Data Science demonstrating iterative model development, deployment, and monitoring."
)
