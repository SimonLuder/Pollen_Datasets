"""Remove every row for audited invalid event IDs, backing up original CSVs."""

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


from ._file_operations import replace_verified
from .._internal.event_validation import count_csv_events


def clean(folder, audit_dir, staging):
    folder = folder.resolve()
    staging.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "events_not_two_rows.csv").open(newline="", encoding="utf-8") as handle:
        invalid = {}
        for row in csv.DictReader(handle):
            if row["issue"] != "row_count_not_two":
                raise ValueError("Missing event IDs require separate handling")
            invalid.setdefault(row["dataset"], {})[row["event_id"]] = int(row["row_count"])
    with (audit_dir / "dataset_summary.csv").open(newline="", encoding="utf-8") as handle:
        expected = {row["dataset"]: row for row in csv.DictReader(handle)}
    sources = sorted(folder.glob("*.csv"))
    if {path.name for path in sources} != set(expected):
        raise ValueError("Dataset filenames have changed since the audit; re-audit first")
    backup = folder / "archive" / ("before_event_cleanup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup.mkdir(parents=True, exist_ok=False)
    for name in ("merge_summary.json",):
        if (folder / name).exists():
            shutil.copy2(folder / name, backup / name)
    shutil.copy2(audit_dir / "events_not_two_rows.csv", backup)
    reports = []
    for source in sources:
        if source.resolve().parent != folder:
            raise ValueError(f"Source outside target directory: {source}")
        print(f"Filtering {source.name}", flush=True)
        before = source.stat()
        bad = invalid.get(source.name, {})
        original_counts = Counter()
        kept_counts = Counter()
        local = staging / source.name
        with source.open(newline="", encoding="utf-8-sig") as reader_handle, local.open("w", newline="", encoding="utf-8") as writer_handle:
            reader = csv.reader(reader_handle)
            writer = csv.writer(writer_handle)
            header = next(reader)
            position = header.index("event_id")
            writer.writerow(header)
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"Malformed row in {source}")
                event = row[position]
                if not event.strip():
                    raise ValueError(f"Missing event ID in {source}")
                original_counts[event] += 1
                if event not in bad:
                    writer.writerow(row)
                    kept_counts[event] += 1
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed while reading: {source}")
        actual_bad = {event: count for event, count in original_counts.items() if count != 2}
        if actual_bad != bad or sum(original_counts.values()) != int(expected[source.name]["rows"]):
            raise ValueError(f"Audit no longer matches {source}; no replacement performed")
        if any(count != 2 for count in kept_counts.values()):
            raise ValueError(f"Invalid retained events in {source}")
        # Check the serialized file, not just the in-memory filtering counts.
        with local.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            if next(reader) != header:
                raise ValueError("Output header mismatch")
            if any(len(row) != len(header) for row in reader):
                raise ValueError("Output row width mismatch")
        persisted, missing = count_csv_events(local)
        if missing or persisted != kept_counts:
            raise ValueError("Output verification failed")
        print(f"Verified {source.name}; copying cleaned file to NAS", flush=True)
        remote = folder / (source.name + ".cleaned.partial")
        saved = backup / source.name
        checksum = replace_verified(source, local, partial=remote, backup=saved,
                                    original_stat=before, root=folder)
        report = {"dataset": source.name, "removed_events": len(bad),
                  "removed_rows": sum(bad.values()), "remaining_rows": sum(kept_counts.values()),
                  "remaining_events": len(kept_counts), "all_events_have_two_rows": True,
                  "sha256": checksum, "backup": str(saved)}
        reports.append(report)
        text = json.dumps(reports, indent=2)
        (folder / "event_cleanup_summary.json").write_text(text, encoding="utf-8")
        (staging / "event_cleanup_summary.json").write_text(text, encoding="utf-8")
        print(json.dumps(report), flush=True)
    print(f"Completed {len(reports)} files. Original files: {backup}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--staging-dir", required=True, type=Path)
    args = parser.parse_args()
    clean(args.folder, args.audit_dir, args.staging_dir)
