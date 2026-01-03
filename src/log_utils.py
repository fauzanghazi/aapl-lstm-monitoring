import os
import csv
from datetime import datetime

# =========================================================
# Log file configuration
# =========================================================
LOG_FILE = "logs/monitoring_logs.csv"

LOG_COLUMNS = [
    "timestamp",
    "run_id",
    "session_hash",
    "model_version",
    "prediction_v1",
    "prediction_v2",
    "latency_v1_ms",
    "latency_v2_ms",
    "feedback_score",
    "feedback_comment",
]


# =========================================================
# Initialize log file
# =========================================================
def init_log_file():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_COLUMNS)


# =========================================================
# Log prediction event
# =========================================================
def log_prediction(
    run_id,
    session_hash,
    model_version,
    prediction_v1,
    prediction_v2,
    latency_v1_ms,
    latency_v2_ms,
    feedback_score,
    feedback_comment,
):
    init_log_file()

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                run_id,
                session_hash,
                model_version,
                round(float(prediction_v1), 4),
                round(float(prediction_v2), 4),
                round(float(latency_v1_ms), 2),
                round(float(latency_v2_ms), 2),
                feedback_score,
                feedback_comment,
            ]
        )
