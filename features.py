"""Convert one Ubuntu host journal into five-minute feature windows."""

from collections import Counter
import json

import pandas as pd

COLUMNS = ["events", "warnings", "errors", "failed_auth"]
WINDOW_US = 300_000_000


def journal_features(path):
    windows = {}
    skipped = 0
    with open(path, encoding="utf-8") as journal:
        for line in journal:
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Expected an object")
                timestamp = int(record["__REALTIME_TIMESTAMP"])
                priority = int(record.get("PRIORITY", 6))
                if timestamp < 0 or not 0 <= priority <= 7:
                    raise ValueError("Invalid timestamp or priority")
            except (ValueError, TypeError, KeyError, OverflowError):
                skipped += 1
                continue

            window = windows.setdefault(timestamp // WINDOW_US, Counter())
            window["events"] += 1
            window["warnings"] += priority == 4
            window["errors"] += priority <= 3
            message = str(record.get("MESSAGE", "")).lower()
            service = record.get("SYSLOG_IDENTIFIER", "")
            window["failed_auth"] += service in ("sshd", "sshd-session") and (
                "failed password" in message or "authentication failure" in message
            )

    if not windows:
        raise ValueError(f"No valid journal entries in {path}")
    keys = sorted(windows)
    index = pd.to_datetime([key * 300 for key in keys], unit="s", utc=True)
    frame = pd.DataFrame([windows[key] for key in keys], index=index, columns=COLUMNS)
    frame.index.name = "window_utc"
    return frame, skipped
