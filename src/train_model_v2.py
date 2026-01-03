import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("data/raw/aapl_2000_2025.csv", parse_dates=["Date"])
df = df.sort_values("Date")

# Train on 2000–2024 only
train_df = df[df["Date"].dt.year <= 2024].copy()

# -----------------------------
# 2. Feature engineering
# -----------------------------
train_df["Return"] = train_df["Close"].pct_change()
train_df = train_df.dropna()

features = train_df[["Close", "Return", "Volume"]].values

# -----------------------------
# 3. Scale features
# -----------------------------
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# -----------------------------
# 4. Create sequences
# -----------------------------
def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i, 0])  # predict Close price
    return np.array(X), np.array(y)

WINDOW_SIZE = 60
X_train, y_train = create_sequences(scaled_features, WINDOW_SIZE)

# -----------------------------
# 5. Build improved LSTM
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

# -----------------------------
# 6. Train model
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 7. Save model + scaler
# -----------------------------
with open("models/model_v2.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "scaler": scaler,
            "window_size": WINDOW_SIZE,
            "feature_names": ["Close", "Return", "Volume"]
        },
        f
    )

print("Model v2 training complete and saved.")
