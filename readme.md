# AAPL LSTM Price Prediction (v1 vs v2)

## Overview

This repository contains a **minimal end-to-end machine learning application** for next-day stock price forecasting using Apple Inc. (AAPL) data.

The project focuses on **model iteration, deployment readiness, monitoring, and feedback**, rather than pure predictive accuracy.

Two LSTM models are compared:

- **Model v1**: Baseline LSTM  
- **Model v2**: Feature-enhanced LSTM iteration  

Both models are deployed via Streamlit with logging, evaluation, and monitoring.

## Key Features

- Two versioned LSTM models (v1 and v2)
- Next-day price prediction
- Rolling forecast visualisation (weekly)
- Latency measurement per model
- Offline evaluation on a 2025 holdout set
- User feedback and monitoring logs
- Monitoring dashboard for deployed behaviour

## Dataset

- Asset: AAPL  
- Frequency: Daily  
- Period:
  - Training: 2000–2024  
  - Holdout evaluation: 2025  
- Target: Next-day closing price  

## Project Structure

```text
aapl-lstm-monitoring/
├── data/
│   └── raw/aapl_2000_2025.csv
│
├── models/
│   ├── model_v1.pkl
│   └── model_v2.pkl
│
├── metrics/
│   └── model_metrics_2025.json
│
├── logs/
│   └── monitoring_logs.csv
│
├── src/
│   ├── train_model_v1.py
│   ├── train_model_v2.py
│   ├── evaluate_models_2025.py
│   ├── predictive_app.py
│   ├── monitor_dashboard.py
│   └── log_utils.py
│
├── requirements.txt
├── README.md
└── .gitignore

```

## Models

### Model v1 (Baseline)

- Input: Closing price  
- Lookback window: 30 days  
- Architecture: Single LSTM layer  

Purpose: establish a simple baseline for comparison.

---

### Model v2 (Iteration)

- Inputs:
  - Closing price  
  - Daily return  
  - Trading volume  
- Lookback window: 60 days  
- Architecture:
  - Deeper LSTM  
  - Regularisation  

Purpose: demonstrate **intentional model iteration**, not guaranteed improvement.

## What This Project Demonstrates

- Model versioning and comparison
- Realistic deployment behaviour using rolling forecasts
- Monitoring and feedback loops
- Separation of training, inference, evaluation, and monitoring
- Practical MLOps and Agile mindset in a compact project

## Notes

This project is intended as a **portfolio, teaching, and assessment-ready example**, not a production trading or investment system.
