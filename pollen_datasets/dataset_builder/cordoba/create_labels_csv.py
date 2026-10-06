"""Index reconstruction images using the poleno48_filter_events.csv columns.

Run from the repository root or with pollen_datasets installed:
    python -m pollen_datasets.dataset_builder.cordoba.create_labels_csv

Paths use forward slashes and are relative to --root (the events folder).
Only filenames ending in .computed_data.holography.image_pairs.N.N.rec_mag.png
are included; JSON metadata and other images are ignored.
"""

import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import re
import tempfile


DEFAULT_ROOT = Path(r"Z:\marvel\marvel-fhnw\data\Cordoba\events")
COLUMNS = [
    "label", "date", "event_id", "split", "filter", "rec_path", "filename",
    "root", "image_nr", "intermediate_path", "img_path",
]
IMAGE_PATTERN = re.compile(
    r"(?P<event_id>.+_(?P<date>\d{4}-\d{2}-\d{2})_.+_ev)"
    r"\.computed_data\.holography\.image_pairs\.\d+\."
    r"(?P<image_nr>\d+)\.rec_mag\.png"
)


def raise_walk_error(error):
    """Do not silently omit folders that cannot be read."""
    raise error


def iter_rows(root, split="test"):
    """Yield one row per reconstruction image, in deterministic path order."""
    for directory, subdirectories, filenames in os.walk(
        root, onerror=raise_walk_error
    ):
        subdirectories.sort()
        for name in sorted(filenames):
            match = IMAGE_PATTERN.fullmatch(name)
            if match is None:
                continue
            relative_path = (Path(directory) / name).relative_to(root).as_posix()
            yield {
                "label": "",
                "date": match["date"],
                "event_id": match["event_id"],
                "split": split,
                "filter": "",
                "rec_path": name,
                "filename": relative_path,
                "root": ".",
                "image_nr": match["image_nr"],
                "intermediate_path": ".",
                "img_path": relative_path,
            }


def write_validated_csv(root, output, split="test"):
    """Keep events with exactly two entries, both pointing to nonempty files.

    Counts span all subfolders. Rows are spooled to disk to limit memory usage.
    Validity means size > 0 bytes; image decoding is not checked.
    """
    counts = Counter()
    valid_counts = Counter()
    empty = unreadable = scanned = kept = 0
    with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as pending:
        writer = csv.DictWriter(pending, fieldnames=COLUMNS)
        writer.writeheader()
        for row in iter_rows(root, split):
            scanned += 1
            event_id = row["event_id"]
            counts[event_id] += 1
            try:
                size = (root / row["img_path"]).stat().st_size
            except OSError:
                unreadable += 1
            else:
                if size > 0:
                    valid_counts[event_id] += 1
                    writer.writerow(row)
                else:
                    empty += 1
            if scanned % 100_000 == 0:
                print(f"Checked {scanned:,} images...", flush=True)

        print("Validating event counts and writing final CSV...", flush=True)
        pending.seek(0)
        with output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            for row in csv.DictReader(pending):
                event_id = row["event_id"]
                if counts[event_id] == 2 and valid_counts[event_id] == 2:
                    writer.writerow(row)
                    kept += 1

    wrong_count = sum(count != 2 for count in counts.values())
    invalid_pairs = sum(
        count == 2 and valid_counts[event_id] != 2
        for event_id, count in counts.items()
    )
    print(f"Scanned {scanned:,} image entries across {len(counts):,} events.")
    print(f"Found {empty:,} zero-byte files and {unreadable:,} inaccessible files.")
    print(f"Dropped {wrong_count:,} events with entry count != 2 and "
          f"{invalid_pairs:,} two-entry events with invalid files.")
    print(f"Dropped {scanned - kept:,} rows; kept {kept // 2:,} complete events.")
    print(f"Wrote {kept:,} image rows to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output", type=Path, default=Path("cordoba_filter_events.csv")
    )
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Events folder is missing or inaccessible: {root}")

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    write_validated_csv(root, args.output, args.split)
    print(f"Image paths are relative to {root}")


if __name__ == "__main__":
    main()
