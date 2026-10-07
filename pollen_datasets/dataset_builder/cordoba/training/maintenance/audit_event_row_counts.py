"""Report event IDs that do not occur exactly twice within each dataset CSV."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

from .._internal.event_validation import count_csv_events


def audit(folder, output):
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSVs found in {folder}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "events_not_two_rows.csv").open("w", newline="", encoding="utf-8") as details:
        writer = csv.writer(details)
        writer.writerow(["dataset", "event_id", "row_count", "issue"])
        summaries = []
        for path in files:
            print(f"Checking {path.name}", flush=True)
            counts, missing = count_csv_events(path)
            invalid = {event: count for event, count in counts.items() if count != 2}
            for event, count in sorted(invalid.items()):
                writer.writerow([path.name, event, count, "row_count_not_two"])
            if missing:
                writer.writerow([path.name, "", missing, "missing_event_id"])
            distribution = Counter(counts.values())
            summary = {
                "dataset": path.name, "rows": sum(counts.values()) + missing,
                "events": len(counts), "events_with_two_rows": distribution[2],
                "events_not_two_rows": len(invalid), "missing_event_id_rows": missing,
                "row_count_distribution": "; ".join(f"{n} rows: {c} events" for n, c in sorted(distribution.items())),
            }
            summaries.append(summary)
            details.flush()
            print(summary, flush=True)
    pd.DataFrame(summaries).to_csv(output / "dataset_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    audit(args.folder, args.output_dir)
