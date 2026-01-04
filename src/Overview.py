import streamlit as st

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="AAPL LSTM System",
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
        "End-to-end data science application with iterative model development, deployment, and monitoring"
    )


st.markdown("---")

# =========================================================
# Overview
# =========================================================
st.subheader("System Overview")

st.write(
    """
    This application demonstrates a **complete data science lifecycle** implemented using
    **Agile and iterative development principles**.

    The system includes:
    - A baseline LSTM model (Model v1)
    - An improved LSTM model (Model v2)
    - A Streamlit prediction interface with latency measurement
    - User feedback collection
    - Centralized logging
    - A monitoring dashboard for operational insights

    Use the **sidebar navigation** to explore each component of the system.
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
        - Next-day AAPL price prediction  
        - Side-by-side comparison of Model v1 and Model v2  
        - Latency measurement for each inference
        """
    )

with col2:
    st.markdown("### 📝 Feedback & Logging")
    st.write(
        """
        - User usefulness rating  
        - Free-text feedback  
        - Persistent logging of predictions, latency, and comments
        """
    )

with col3:
    st.markdown("### 📊 Monitoring")
    st.write(
        """
        - Latency comparison between models  
        - Feedback score analysis  
        - Recent user comments  
        - Raw operational logs
        """
    )

st.markdown("---")

# =========================================================
# Live Market Context (TradingView)
# =========================================================
st.subheader("Live Market Context")

st.write(
    """
    The chart below provides real-world market context for the prediction task,
    allowing users to interpret model outputs relative to recent AAPL price movements.
    """
)

tradingview_html = """
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js" async>
  {
    "symbols": [
      ["Apple", "NASDAQ:AAPL|1D"]
    ],
    "chartOnly": false,
    "width": "100%",
    "height": 500,
    "locale": "en",
    "colorTheme": "dark",
    "autosize": false,
    "showVolume": true,
    "showMA": false,
    "hideDateRanges": false,
    "hideMarketStatus": false,
    "hideSymbolLogo": false,
    "scalePosition": "right",
    "scaleMode": "Normal",
    "fontFamily": "Trebuchet MS, sans-serif",
    "fontSize": "10",
    "noTimeScale": false,
    "valuesTracking": "1",
    "changeMode": "price-and-percent",
    "chartType": "candlesticks"
  }
  </script>
</div>
"""

st.components.v1.html(tradingview_html, height=520)


# =========================================================
# Footer note
# =========================================================
st.markdown("---")
st.caption(
    "Academic project demonstrating Agile-based data science system construction using Streamlit and LSTM models."
)
