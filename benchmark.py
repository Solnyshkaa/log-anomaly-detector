import csv
import re
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from detector import fit, score

BLOCK = re.compile(r"\bblk_-?\d+\b")
LEVEL_INDEX = {"INFO": 1, "WARN": 2, "ERROR": 3, "FATAL": 3}
FEATURES = ["lines", "info", "warnings", "errors"]

def hdfs_features(log_path, labels_path):
    labels = {}
    with open(labels_path, encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            block, label = row["BlockId"], row["Label"].strip().lower()
            if block in labels or label not in ("normal", "anomaly"):
                raise ValueError(f"Duplicate block or unexpected label for {block}")
            labels[block] = int(label == "anomaly")

    counts = {}
    with open(log_path, encoding="utf-8", errors="replace") as log:
        for line in log:
            fields = line.split(maxsplit=5)
            if len(fields) < 6:
                continue
            ids = set(BLOCK.findall(fields[5]))
            level = LEVEL_INDEX.get(fields[3])
            for block in ids.intersection(labels):
                vector = counts.setdefault(block, [0, 0, 0, 0])
                vector[0] += 1
                if level is not None:
                    vector[level] += 1

    if not counts:
        raise ValueError("No labeled block IDs matched the HDFS log")
    frame = pd.DataFrame.from_dict(counts, orient="index", columns=FEATURES).sort_index()
    target = pd.Series(labels).reindex(frame.index)
    return frame, target, len(labels) - len(counts)

def evaluate(log_path, labels_path, alert_rate):
    features, labels, unmatched = hdfs_features(log_path, labels_path)
    normal = features.loc[labels.eq(0)]
    anomalies = features.loc[labels.eq(1)]
    if len(normal) < 30 or anomalies.empty:
        raise ValueError("Benchmark requires at least 30 normal blocks and one anomaly")
    train, remaining = train_test_split(normal, test_size=0.4, random_state=42)
    validation, test_normal = train_test_split(remaining, test_size=0.5, random_state=42)
    test = pd.concat([test_normal, anomalies])
    truth = labels.reindex(test.index)
    model = fit(train)
    results = {}
    for name, valid_scores, test_scores in [
        ("isolation_forest", score(model, validation), score(model, test)),
        ("warning_error_count", validation["warnings"] + validation["errors"],
         test["warnings"] + test["errors"]),
    ]:
        threshold = float(valid_scores.quantile(1 - alert_rate))
        predictions = test_scores.gt(threshold)
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predictions, average="binary", zero_division=0
        )
        results[name] = {
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "average_precision": float(average_precision_score(truth, test_scores)),
            "threshold": threshold, "validation_alert_rate": float(valid_scores.gt(threshold).mean()),
            "confusion_matrix_TN_FP_FN_TP": confusion_matrix(truth, predictions, labels=[0, 1]).ravel().tolist(),
        }

    return {
        "train_normal": len(train), "validation_normal": len(validation),
        "test_normal": len(test_normal), "test_anomaly": len(anomalies),
        "labels_without_log_records": unmatched,
        "models": results,
    }
