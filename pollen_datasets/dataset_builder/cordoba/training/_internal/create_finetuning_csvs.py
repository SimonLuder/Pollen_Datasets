"""Append missing species from matching left-out splits to basic/combined CSVs."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .event_validation import count_event_ids, require_event_pairs, validate_csv_images, validate_split_isolation
from .species_exclusions import exclusion_names, filter_species_frame, filter_species_csv


def species_counts(path, *, event_report=None):
    counts = {}
    events = Counter()
    missing_events = 0
    columns = ["species", "event_id"] if event_report is not None else ["species"]
    for chunk in pd.read_csv(path, usecols=columns, dtype=str,
                             keep_default_na=False, chunksize=100000):
        if event_report is not None:
            current, missing = count_event_ids(chunk.event_id)
            missing_events += missing
            events.update(current)
        for species, count in chunk.species.value_counts().items():
            if not species.strip():
                raise ValueError(f"Missing species in {path}")
            counts[species] = counts.get(species, 0) + int(count)
    if event_report is not None:
        require_event_pairs(events, context=path.name, report_path=event_report,
                            missing_rows=missing_events)
    return counts


def build(folder, workbook=None, *, left_folder=None, output=None):
    left_folder = folder if left_folder is None else Path(left_folder)
    if workbook is None and (folder / "poleno_monthly_species.xlsx").exists():
        workbook = folder / "poleno_monthly_species.xlsx"
    if workbook is None:
        raise ValueError("A species workbook is required to exclude deliberately unknown species")
    excluded = exclusion_names(workbook)
    sources = sorted(p for p in folder.glob("*.csv")
                     if p.name.startswith(("basic", "combined")))
    if not sources:
        raise ValueError(f"No basic/combined CSVs in {folder}")
    pairs = []
    for source in sources:
        suffix = source.stem.split("_", 1)[1]
        if suffix not in {"train", "val", "test", "val_20", "test_20"}:
            raise ValueError(f"Cannot determine corresponding split: {source.name}")
        left = left_folder / f"left_out_poleno_{suffix}.csv"
        if not left.is_file():
            raise FileNotFoundError(left)
        pairs.append((source, left))
    out = folder / "finetuning" if output is None else Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for source, _ in pairs:
        if (out / source.name).exists():
            raise FileExistsError(out / source.name)
    report = {}
    staged = []
    for source, left in pairs:
        print(f"Inspecting {source.name}", flush=True)
        target = out / source.name
        temporary = target.with_suffix(".csv.partial")
        original_counts, removed = filter_species_csv(source, temporary, excluded)
        extra = filter_species_frame(pd.read_csv(left, dtype=str, keep_default_na=False), excluded)
        missing = sorted(set(extra.species) - set(original_counts))
        extra = extra.loc[extra.species.isin(missing)]
        with source.open(encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle))
        # Some original splits omit the optional local event enumeration.
        if set(extra.columns) - set(columns) - {"event_id_enum"}:
            raise ValueError(f"Left-out columns missing from {source.name}")
        extra.reindex(columns=columns, fill_value="").to_csv(
            temporary, mode="a", header=False, index=False)
        expected = dict(original_counts)
        expected.update({name: int(count) for name, count in extra.species.value_counts().items()})
        # Count across the entire merged file, including IDs shared by labels.
        actual = species_counts(temporary, event_report=
                                out / "validation" / f"{source.stem}_invalid_events.csv")
        if actual != expected:
            raise ValueError(f"Output count verification failed: {source.name}")
        validate_csv_images(temporary, report_path=
                            out / "validation" / f"{source.stem}_invalid_images.csv")
        staged.append((source.name, temporary))
        report[source.name] = {
            "left_out_source": left.name, "added_species": missing,
            "original_rows": sum(original_counts.values()) + sum(removed.values()),
            "retained_original_rows": sum(original_counts.values()), "added_rows": len(extra),
            "output_rows": sum(actual.values()), "species_counts": actual,
            "all_events_have_two_rows": True,
            "all_events_have_distinct_images": True,
            "excluded_species": sorted(excluded), "removed_original_rows_by_species": removed,
        }
        print(f"Validated {target.name}: +{len(extra):,} rows, +{len(missing)} species; "
              f"{sum(actual.values()):,} total rows", flush=True)
    validate_split_isolation(staged, report_path=out / "validation" / "split_overlap.csv")
    for name, temporary in staged:
        temporary.rename(out / name)
        report[name]["split_isolation_verified"] = True
    (out / "merge_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--species-workbook", type=Path)
    args = parser.parse_args()
    build(args.folder, args.species_workbook)
