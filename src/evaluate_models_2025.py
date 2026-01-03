import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/raw/aapl_2000_2025.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

train_df = df[df["Date"].dt.year <= 2024]
test_df = df[df["Date"].dt.year == 2025]

# -----------------------------
# Load models
# -----------------------------
with open("models/model_v1.pkl", "rb") as f:
    model_v1 = pickle.load(f)

with open("models/model_v2.pkl", "rb") as f:
    model_v2 = pickle.load(f)


# -----------------------------
# Helper functions
# -----------------------------
def prepare_input_v1(df, end_date, model_dict):
    window = model_dict["window_size"]
    scaler = model_dict["scaler"]

    data = df[df["Date"] <= end_date]["Close"].values[-window:]
    scaled = scaler.transform(data.reshape(-1, 1))
    return scaled.reshape(1, window, 1)


def prepare_input_v2(df, end_date, model_dict):
    window = model_dict["window_size"]
    scaler = model_dict["scaler"]

    temp = df.copy()
    temp["Return"] = temp["Close"].pct_change()
    temp = temp.dropna()

    features = temp[temp["Date"] <= end_date][["Close", "Return", "Volume"]].values[
        -window:
    ]

    scaled = scaler.transform(features)
    return scaled.reshape(1, window, scaled.shape[1])


# -----------------------------
# Rolling evaluation on 2025
# -----------------------------
y_true = []
y_pred_v1 = []
y_pred_v2 = []

for i in range(len(test_df) - 1):
    current_date = test_df.iloc[i]["Date"]
    next_day_price = test_df.iloc[i + 1]["Close"]

    # Model v1
    X1 = prepare_input_v1(df, current_date, model_v1)
    p1_scaled = model_v1["model"].predict(X1, verbose=0)
    p1 = model_v1["scaler"].inverse_transform(p1_scaled)[0][0]

    # Model v2
    X2 = prepare_input_v2(df, current_date, model_v2)
    p2_scaled = model_v2["model"].predict(X2, verbose=0)
    p2 = model_v2["scaler"].inverse_transform(np.hstack([p2_scaled, np.zeros((1, 2))]))[
        0
    ][0]

    y_true.append(next_day_price)
    y_pred_v1.append(p1)
    y_pred_v2.append(p2)

# -----------------------------
# Metrics (version-safe)
# -----------------------------
metrics = {
    "v1": {
        "mae": float(round(mean_absolute_error(y_true, y_pred_v1), 2)),
        "rmse": float(round(np.sqrt(mean_squared_error(y_true, y_pred_v1)), 2)),
    },
    "v2": {
        "mae": float(round(mean_absolute_error(y_true, y_pred_v2), 2)),
        "rmse": float(round(np.sqrt(mean_squared_error(y_true, y_pred_v2)), 2)),
    },
}

# -----------------------------
# Save metrics
# -----------------------------
output_path = "metrics/model_metrics_2025.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("2025 evaluation complete.")
print(metrics)
