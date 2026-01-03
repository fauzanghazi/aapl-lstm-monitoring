import os
import csv
from datetime import datetime

LOG_FILE = "logs/monitoring_logs.csv"

LOG_COLUMNS = [
    "timestamp",
    "model_version",
    "prediction",
    "latency_ms",
    "feedback_score",
    "feedback_comment"
]


def init_log_file():
    """
    Create monitoring log file with headers if it does not exist.
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_COLUMNS)


def log_prediction(
    model_version,
    prediction,
    latency_ms,
    feedback_score=None,
    feedback_comment=""
):
    """
    Append a prediction event to monitoring_logs.csv
    """
    init_log_file()

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            model_version,
            round(float(prediction), 4),
            round(float(latency_ms), 2),
            feedback_score,
            feedback_comment
        ])
