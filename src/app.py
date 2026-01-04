import streamlit as st

st.set_page_config(page_title="AAPL LSTM System", layout="wide")

st.title("AAPL LSTM Prediction & Monitoring System")
st.write(
    """
    This application demonstrates an end-to-end data science workflow,
    including model prediction, user feedback, logging, and monitoring.
    
    Use the sidebar to navigate between Prediction and Monitoring pages.
    """
)
