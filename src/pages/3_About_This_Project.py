import streamlit as st

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="About This Project",
    layout="wide",
)

# =========================================================
# Header section
# =========================================================
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.markdown(
        """
        <img src="https://www.svgrepo.com/show/530405/line-graph.svg"
             style="width:120px; filter: grayscale(0%) brightness(100%); opacity:1;">
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.title("Enhancing AAPL Prediction with LSTM")
    st.caption(
        "End-to-end data science application with iterative model development, deployment, and monitoring"
    )

with col3:
    st.write("")  # spacer

st.markdown("---")


# =========================================================
# Course & context
# =========================================================
st.subheader("Course Context")

st.write(
    """
    **Course:** MRTB2173 – Agile Data Science  
    **Lecturer:** Dr Fiza Abdul Rahim  

    This project is developed as part of the course *Agile Data Science*, focusing on the
    construction of a **minimum viable, end-to-end data science system** using
    **iterative development and deployment principles**.
    """
)

st.markdown("---")

# =========================================================
# CLO
# =========================================================
st.subheader("Course Learning Outcome (CLO2)")

st.info(
    """
    **CLO2:** Construct a data science application using iterative development and
    deployment following Agile principles.
    """
)

# =========================================================
# Project description
# =========================================================
st.subheader("Project Description")

st.write(
    """
    This case study demonstrates the practical application of **Agile Data Science**
    by designing, implementing, and deploying an **LSTM-based stock price prediction system**
    for Apple Inc. (AAPL).

    The system was built incrementally through multiple iterations, starting with a
    baseline model (Model v1) and subsequently improved into Model v2 based on
    feature enhancement, performance evaluation, and system feedback.

    Beyond model development, the project emphasizes **operationalization**, including:
    - Real-time prediction via Streamlit
    - Latency measurement
    - User feedback collection
    - Centralized logging
    - Monitoring and performance analysis
    """
)

# =========================================================
# Agile principles applied
# =========================================================
st.subheader("Agile Principles Applied")

st.write(
    """
    The project follows Agile principles through:
    - Incremental model development (v1 → v2)
    - Continuous feedback integration
    - Clear separation of prediction and monitoring components
    - Rapid iteration and deployment using Streamlit
    - Evidence-based evaluation through logged metrics
    """
)

# =========================================================
# Team information
# =========================================================
st.subheader("Project Team")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        **Fauzan Ghazi**  
        Matric No: MRT234008  

        Role:
        - Model development
        - Streamlit application design
        - Monitoring & logging implementation
        """
    )

with col2:
    st.markdown(
        """
        **Grace Chong**  
        Matric No: MRT231051  

        Role:
        - Agile documentation
        - Case study reporting
        - User feedback analysis
        """
    )

st.markdown("---")

# =========================================================
# Closing note
# =========================================================
st.caption(
    "This project serves as an academic demonstration of Agile-based data science system construction using Streamlit and LSTM models."
)
