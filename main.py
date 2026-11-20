"""Analyze a system journal or evaluate detectors against Loghub labels."""

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmark import evaluate
from detector import fit, score
from features import journal_features


def analyze(args):
    reference, skipped_reference = journal_features(args.reference)
    current, skipped_current = journal_features(args.input)
    if reference.index.max() >= current.index.min():
        raise ValueError("The input windows must start after the reference windows end")
    if len(reference) < 30:
        raise ValueError("Export a longer reference with at least 30 populated windows")

    boundary = int(len(reference) * 0.8)
    model = fit(reference.iloc[:boundary])
    calibration = score(model, reference.iloc[boundary:])
    scores = score(model, current)
    threshold = calibration.quantile(1 - args.alert_rate)
    ml_alert = scores.gt(threshold)
    auth_alert = current["failed_auth"].ge(5)
    report = current.assign(
        anomaly_score=scores,
        threshold=threshold,
        ml_alert=ml_alert,
        auth_alert=auth_alert,
        is_anomaly=ml_alert | auth_alert,
    )
    print(f"Skipped records: reference={skipped_reference}, input={skipped_current}")
    print(f"Alerts: {report.is_anomaly.sum()} / {len(report)} windows")
    return report


def demo(output):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    output.mkdir(parents=True, exist_ok=True)
    for name, offset, windows in [("reference", 0, 80), ("monitor", 80, 12)]:
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as file:
            for index in range(windows):
                anomaly = name == "monitor" and index == 7
                timestamp = (start + pd.Timedelta(minutes=5 * (offset + index))).value // 1000
                for event in range(24 if anomaly else 4 + index % 5):
                    failed = anomaly and event < 16
                    record = {
                        "__REALTIME_TIMESTAMP": str(timestamp),
                        "PRIORITY": "3" if failed else ("4" if event == 0 and index % 6 == 0 else "6"),
                        "SYSLOG_IDENTIFIER": "sshd" if failed else "synthetic-demo",
                        "MESSAGE": "Failed password [synthetic]" if failed else "Synthetic event",
                    }
                    print(json.dumps(record), file=file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo_parser = commands.add_parser("demo", help="Generate synthetic journal samples")
    demo_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze", help="Score chronological journal windows")
    analyze_parser.add_argument("--reference", type=Path, required=True)
    analyze_parser.add_argument("--input", type=Path, required=True)
    benchmark_parser = commands.add_parser("benchmark", help="Evaluate with Loghub block labels")
    benchmark_parser.add_argument("--log", type=Path, required=True)
    benchmark_parser.add_argument("--labels", type=Path, required=True)
    for command in (analyze_parser, benchmark_parser):
        command.add_argument("--alert-rate", type=float, default=0.05)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "analyze" and args.output.suffix.lower() != ".csv":
        parser.error("analyze --output requires a .csv path")

    try:
        if args.command == "demo":
            demo(args.output)
            return
        if not 0 < args.alert_rate < 0.5:
            parser.error("--alert-rate must be between 0 and 0.5")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.command == "analyze":
            report = analyze(args)
            report.to_csv(args.output)
            report.to_html(args.output.with_suffix(".html"), border=0)
        else:
            result = evaluate(args.log, args.labels, args.alert_rate)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.output}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
