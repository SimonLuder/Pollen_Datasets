"""Require exactly two rows per event ID, across all species and datasets."""

from pathlib import Path
import csv
import posixpath
from collections import Counter

import pandas as pd


def count_event_ids(events):
    """Count nonblank event IDs and report missing rows for a series or chunk."""
    missing = events.isna() | events.astype(str).str.strip().eq("")
    return Counter(events.loc[~missing].value_counts().to_dict()), int(missing.sum())


def count_csv_events(path):
    counts, missing = Counter(), 0
    for chunk in pd.read_csv(path, usecols=["event_id"], dtype=str,
                             keep_default_na=False, chunksize=100000):
        current, blank = count_event_ids(chunk.event_id)
        counts.update(current)
        missing += blank
    return counts, missing


def require_event_pairs(counts, *, context, report_path, missing_rows=0):
    issues = [{"event_id": event, "row_count": int(count), "issue": "row_count_not_two"}
              for event, count in counts.items() if count != 2]
    if missing_rows:
        issues.append({"event_id": "", "row_count": int(missing_rows), "issue": "missing_event_id"})
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(issues, columns=["event_id", "row_count", "issue"]).to_csv(report_path, index=False)
    if issues:
        raise ValueError(
            f"{context}: every event_id must have exactly two rows. "
            f"Found {len(issues) - bool(missing_rows)} invalid event IDs and "
            f"{missing_rows} rows with missing IDs. See {report_path}. "
            "No rows were automatically removed."
        )


def validate_frame_events(rows, *, context, report_path):
    counts, missing = count_event_ids(rows["event_id"])
    require_event_pairs(counts,
                        context=context, report_path=report_path,
                        missing_rows=missing)


def require_distinct_images(records, columns, *, context, report_path):
    """Check two distinct nonblank image references, not image file contents."""
    references = [name for name in ("img_path", "filename", "rec_path") if name in columns]
    if not references:
        raise ValueError(f"{context}: need img_path, filename, or rec_path to verify image pairs")
    images = {}
    issues = []
    for row in records:
        event = str(row["event_id"])
        # Prefer the complete path; a dataset scopes a bare reconstructed filename.
        selected, reference = next(((name, str(row[name]).strip()) for name in references
                                   if row[name] is not None and str(row[name]).strip()), ("", ""))
        if not reference:
            issues.append({"event_id": event, "issue": "missing_image_reference"})
            continue
        reference = posixpath.normpath(reference.replace("\\", "/"))
        key = (str(row.get("dataset_id", "")) if selected == "rec_path" else "", reference)
        if key in images.setdefault(event, set()):
            issues.append({"event_id": event, "issue": "duplicate_image_reference"})
        images[event].add(key)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(issues, columns=["event_id", "issue"]).to_csv(report_path, index=False)
    if issues:
        raise ValueError(f"{context}: every event must reference two distinct images. See {report_path}")


def validate_csv_images(path, *, report_path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_distinct_images(reader, reader.fieldnames, context=Path(path).name,
                                report_path=report_path)


def validate_split_isolation(files, *, report_path):
    """Globally check train/val/test roles across dataset variants and subsets."""
    assignments = {}
    issues = []
    for name, path in files:
        role = name.split("_", 1)[1].removesuffix(".csv").split("_", 1)[0]
        for chunk in pd.read_csv(path, usecols=["event_id"], dtype=str,
                                 keep_default_na=False, chunksize=100000):
            for event in chunk.event_id.unique():
                previous = assignments.get(event)
                if previous is not None and previous[0] != role:
                    issues.append({"event_id": event, "first_dataset": previous[1],
                                   "conflicting_dataset": name})
                else:
                    assignments[event] = (role, name)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(issues, columns=["event_id", "first_dataset", "conflicting_dataset"]).to_csv(report_path, index=False)
    if issues:
        raise ValueError(f"Event IDs overlap between train/val/test. See {report_path}")
