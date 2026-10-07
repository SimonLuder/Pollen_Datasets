"""Create a CSV showing which species occur in each CSV dataset."""

import argparse
import os
from pathlib import Path

import pandas as pd


def inventory_csv_species(root, output_csv, *, species_column="species", chunksize=100_000):
    """Scan recursively, pruning archive directories; return the presence table.

    Species columns contain 1 (present), 0 (absent), or blank (file could not
    be evaluated). Species names are stripped of surrounding whitespace but
    retain their case and spelling. Missing/empty labels are ignored.
    """
    root = Path(root).resolve()
    output = Path(output_csv).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")

    files = []
    def traversal_error(error):
        raise error

    for directory, subdirectories, filenames in os.walk(root, onerror=traversal_error):
        subdirectories[:] = sorted(d for d in subdirectories if d.casefold() != "archive")
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.suffix.casefold() == ".csv" and path.resolve() != output:
                files.append(path)
    if not files:
        raise ValueError(f"No CSV datasets found outside archive directories in {root}.")

    records = []
    all_species = set()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        record = {"dataset_path": relative, "row_count": None, "species_count": None,
                  "status": "ok", "error": ""}
        names = set()
        try:
            columns = pd.read_csv(path, nrows=0).columns
            if species_column not in columns:
                record["status"] = "missing_species_column"
                record["error"] = f"No {species_column!r} column"
            else:
                count = 0
                with pd.read_csv(path, usecols=[species_column], dtype=str,
                                 keep_default_na=False, chunksize=chunksize) as reader:
                    for chunk in reader:
                        values = chunk[species_column].str.strip()
                        names.update(values.loc[values.ne("")].unique())
                        count += len(chunk)
                record["row_count"] = count
                record["species_count"] = len(names)
                all_species.update(names)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            record["status"] = "read_error"
            record["error"] = str(error)
            names = set()
        print(f"{relative}: {len(names)} species ({record['status']})", flush=True)
        records.append((record, names))

    species_names = sorted(all_species, key=lambda name: (name.casefold(), name))
    rows = []
    for record, names in records:
        # Prefix prevents collisions with metadata columns such as 'status'.
        record.update({f"species::{name}": int(name in names) if record["status"] == "ok" else None
                       for name in species_names})
        rows.append(record)
    result = pd.DataFrame(rows)
    for column in ["row_count", "species_count"] + [f"species::{name}" for name in species_names]:
        result[column] = result[column].astype("Int64")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result)} datasets and {len(species_names)} distinct species to {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path,
                        help="Report path; excluded from the scan when inside root. Replaced on reruns.")
    parser.add_argument("--species-column", default="species")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    inventory_csv_species(args.root, args.output_csv,
                          species_column=args.species_column, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
