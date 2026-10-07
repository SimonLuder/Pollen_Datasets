"""Species held out from training and finetuning datasets."""
import csv
from collections import Counter
from ..._internal.species_workbook import normalize_species, read_species_workbook


def exclusion_names(workbook):
    if workbook is None:
        raise ValueError("A species workbook is required to define exclusions")
    spellings = {}
    for row in read_species_workbook(workbook):
        if not row["known"]:
            spellings.setdefault(normalize_species(row["species"]), row["species"])
    return set(spellings.values())


def filter_species_frame(frame, excluded=()):
    names = {normalize_species(name) for name in excluded}
    keep = None
    for column in ("species", "species_norm"):
        if column in frame:
            allowed = ~frame[column].astype(str).str.strip().str.casefold().isin(names)
            keep = allowed if keep is None else keep & allowed
    if keep is None:
        raise ValueError("No species or species_norm column")
    return frame.loc[keep].copy()


def filter_species_csv(source, target, excluded=()):
    """Preserve retained cell strings, column order and row order."""
    names = {normalize_species(name) for name in excluded}
    kept, removed = Counter(), Counter()
    with source.open(encoding="utf-8-sig", newline="") as src, target.open("w", encoding="utf-8", newline="") as dst:
        reader, writer = csv.reader(src), csv.writer(dst)
        header = next(reader)
        indexes = [header.index(c) for c in ("species", "species_norm") if c in header]
        if not indexes:
            raise ValueError(f"No species column: {source}")
        label_index = indexes[0]
        writer.writerow(header)
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Malformed CSV row: {source}")
            if any(row[i].strip().casefold() in names for i in indexes):
                removed[row[label_index]] += 1
            else:
                writer.writerow(row)
                kept[row[label_index]] += 1
    return dict(kept), dict(removed)
