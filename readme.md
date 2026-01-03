# AAPL LSTM Price Prediction with Monitoring and Feedback

## Project Overview

This project demonstrates a **minimum viable end-to-end data science application** using historical Apple Inc. (AAPL) stock data.  
The focus is not only on model accuracy, but also on **deployment readiness, monitoring, iteration, and human feedback**, following modern MLOps and Agile principles.

The system compares two LSTM models:

- **Model v1**: Baseline implementation
- **Model v2**: Improved iteration with enhanced features and architecture

Both models are deployed through a Streamlit application, monitored via structured logging, and evaluated using a separate monitoring dashboard.

---

## Objectives

The objectives of this project are to:

- Build and compare two versions of an LSTM time series model  
- Demonstrate clear **model iteration and improvement**  
- Deploy models through a functional prediction interface  
- Capture prediction latency and user feedback  
- Monitor deployed model behaviour using real logs  
- Present results in a transparent and auditable manner  

---

## Dataset

- Asset: AAPL stock  
- Frequency: Daily  
- Period:
  - Training data: 2000–2024  
  - Holdout period: 2025  
- Source: Public historical market data  

The target variable is the **next-day closing price**.

---

## Project Structure

aapl-lstm-monitoring/
│
├── data/
│   ├── raw/
│   │   └── aapl_2000_2025.csv
│
├── models/
│   ├── model_v1.pkl
│   └── model_v2.pkl
│
├── logs/
│   └── monitoring_logs.csv
│
├── src/
│   ├── train_model_v1.py
│   ├── train_model_v2.py
│   ├── log_utils.py
│   ├── predictive_app.py
│   └── monitor_dashboard.py
│
├── requirements.txt
├── README.md
└── .gitignore

---

## Model Design and Iteration

### Model v1: Baseline LSTM

- Feature set: Closing price only  
- Lookback window: 30 days  
- Architecture: Single LSTM layer  
- Purpose: Establish baseline performance  

### Model v2: Improved LSTM

- Feature set:
  - Closing price  
  - Daily return  
  - Trading volume  
- Lookback window: 60 days  
- Architecture:
  - Two LSTM layers  
  - Dropout for regularisation  
- Training improvements:
  - Validation split  
  - Early stopping on validation loss  

The changes from v1 to v2 are intentional and documented to demonstrate **iterative model improvement**.

---

## Streamlit Applications

### 1. Prediction Application (`predictive_app.py`)

The prediction app allows users to:

- Select a date from the dataset  
- Generate next-day price predictions from both models  
- Compare prediction latency between model versions  
- Provide feedback using a rating score and comments  

All prediction events are logged automatically.

### 2. Monitoring Dashboard (`monitor_dashboard.py`)

The monitoring dashboard provides:

- Average latency comparison between models  
- Average user feedback score per model  
- Distribution of feedback ratings  
- Recent qualitative user comments  
- Full raw monitoring logs for auditability  

This dashboard enables post-deployment visibility into model behaviour.

---

## Logging and Monitoring

All predictions are logged to `monitoring_logs.csv` with the following fields:

- Timestamp  
- Model version  
- Prediction value  
- Latency in milliseconds  
- User feedback score  
- User comments  

Logging is handled centrally through `log_utils.py` to ensure consistency and reusability.

---

## Agile and Iterative Development Evidence

This repository demonstrates Agile practices through:

- Clear separation of baseline and improved models  
- Incremental commits showing refinement  
- Explicit documentation of changes between iterations  
- Monitoring and feedback loops enabling continuous improvement  

---

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
