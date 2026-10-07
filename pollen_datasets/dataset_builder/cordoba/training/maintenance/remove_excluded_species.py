"""Remove excluded species from CSVs directly in a folder and its finetuning folder."""
import argparse
import json
from collections import Counter
from pathlib import Path
import uuid

import pandas as pd

from .._internal.species_exclusions import exclusion_names, filter_species_frame, filter_species_csv


from ._file_operations import replace_verified
from .._internal.event_validation import count_event_ids


def inspect(path, excluded, check_events=False):
    header = pd.read_csv(path, nrows=0).columns
    labels = [c for c in ("species", "species_norm") if c in header]
    # Reports without species labels cannot contain labeled dataset samples.
    if not labels:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        names = {s.casefold() for s in excluded}
        if any(frame[c].str.strip().str.casefold().isin(names).any() for c in frame):
            raise ValueError(f"Excluded name in unrecognized report schema: {path}")
        return {"rows": len(frame), "excluded_rows": 0, "species_counts": {}, "no_species_columns": True}
    columns = labels + (["event_id"] if check_events and "event_id" in header else [])
    total, excluded_count = 0, 0
    counts, events = Counter(), Counter()
    for chunk in pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False, chunksize=100000):
        retained = filter_species_frame(chunk, excluded)
        total += len(chunk)
        excluded_count += len(chunk) - len(retained)
        counts.update(retained[labels[0]].value_counts().to_dict())
        if "event_id" in columns:
            current, missing = count_event_ids(retained.event_id)
            if missing:
                raise ValueError(f"Missing event ID: {path}")
            events.update(current)
    if events and any(n != 2 for n in events.values()):
        raise ValueError(f"Retained finetuning events do not all have two rows: {path}")
    return {"rows": total, "excluded_rows": excluded_count, "species_counts": dict(counts)}


def run(folder, staging, workbook):
    folder = folder.resolve()
    excluded = exclusion_names(workbook)
    print("Excluding: " + ", ".join(sorted(excluded)), flush=True)
    folders = [folder, folder / "finetuning"]
    files = [(directory, path) for directory in folders for path in sorted(directory.glob("*.csv"))]
    staging.mkdir(parents=True, exist_ok=True)
    backup_name = "before_species_exclusion_" + uuid.uuid4().hex[:12]
    report = {"workbook": str(workbook), "excluded_species": sorted(excluded), "files": []}
    for index, (directory, path) in enumerate(files):
        if path.resolve().parent != directory.resolve():
            raise ValueError("CSV outside requested directory")
        print(f"Inspecting {path}", flush=True)
        stamp = path.stat()
        before = inspect(path, excluded, check_events=directory.name == "finetuning")
        entry = {"file": str(path), "original_rows": before["rows"],
                 "removed_rows": before["excluded_rows"], "species_counts": before["species_counts"]}
        if before["excluded_rows"]:
            local = staging / f"{index}_{path.name}"
            kept, removed = filter_species_csv(path, local, excluded)
            verified = inspect(local, excluded, check_events=directory.name == "finetuning")
            if verified["excluded_rows"] or verified["rows"] != before["rows"] - before["excluded_rows"] or kept != before["species_counts"]:
                raise ValueError(f"Filtered file validation failed: {path}")
            remote = path.with_suffix(".csv.species_exclusion.partial")
            saved = directory / "archive" / backup_name / path.name
            replace_verified(path, local, partial=remote, backup=saved,
                             original_stat=stamp, root=folder)
            entry.update(backup=str(saved), removed_by_species=removed)
            local.unlink()
        entry["remaining_rows"] = before["rows"] - before["excluded_rows"]
        entry["excluded_species_remaining"] = 0
        report["files"].append(entry)
        text = json.dumps(report, indent=2)
        (staging / "species_exclusion_summary.json").write_text(text, encoding="utf-8")
        (folder / "species_exclusion_summary.json").write_text(text, encoding="utf-8")
        print(f"Complete: {path.name}, removed {entry['removed_rows']}, remaining {entry['remaining_rows']}", flush=True)
    print(f"Verified {len(files)} CSVs; removed {sum(r['removed_rows'] for r in report['files'])} rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--species-workbook", type=Path)
    args = parser.parse_args()
    run(args.folder, args.staging_dir, args.species_workbook or args.folder / "poleno_monthly_species.xlsx")
