import csv
import os
from datetime import datetime

LOG_FILE = "chat_logs.csv"

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "participant_id",
                "condition",
                "session",
                "role",
                "message"
            ])


def log_message(participant_id, condition, session, role, message):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            participant_id,
            condition,
            session,
            role,
            message
        ])