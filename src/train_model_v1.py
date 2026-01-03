import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("data/raw/aapl_2000_2025.csv", parse_dates=["Date"])
df = df.sort_values("Date")

# Train on 2000–2024 only
train_df = df[df["Date"].dt.year <= 2024]

prices = train_df["Close"].values.reshape(-1, 1)

# -----------------------------
# 2. Scale data
# -----------------------------
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

# -----------------------------
# 3. Create sequences
# -----------------------------
def create_sequences(data, window=30):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

WINDOW_SIZE = 30
X_train, y_train = create_sequences(scaled_prices, WINDOW_SIZE)

# -----------------------------
# 4. Build LSTM (Baseline)
# -----------------------------
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

# -----------------------------
# 5. Train model
# -----------------------------
early_stop = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 6. Save model + scaler
# -----------------------------
with open("models/model_v1.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "scaler": scaler,
            "window_size": WINDOW_SIZE
        },
        f
    )

print("Model v1 training complete and saved.")
