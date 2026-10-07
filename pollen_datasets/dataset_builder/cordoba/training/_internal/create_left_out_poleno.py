"""Build a Poleno25 subset with image paths and labels, without particle features."""

import json
import os
from pathlib import Path

import pandas as pd

from .event_splitting import split_by_species_events, sample_species_events

from .event_validation import validate_frame_events, require_distinct_images
from .species_exclusions import exclusion_names, filter_species_frame


SPECIES = [
    "Cupressus sp.", "Cynosurus cristatus",
    "Poaceae sp.", "Populus sp.", "Quercus sp.", "Trisetum sp.", "Ulmus sp.",
]

# Explicit schema keeps index artifacts and particle measurements out of exports.
OUTPUT_COLUMNS = [
    "event_id", "dataset_id", "rec_path", "filename", "root",
    "species", "genus", "image_nr", "intermediate_path", "img_path",
    "species_norm", "dataset_id_enum", "species_norm_enum", "genus_enum",
    "event_id_enum",
]


def export_event_subsets(splits, out, random_state=42):
    """Write 20-event validation/test subsets sampled from their parent splits."""
    subsets = {name: sample_species_events(splits[name], random_state=random_state)
               for name in ("val", "test")}
    for name, frame in subsets.items():
        validate_frame_events(frame, context=f"left_out_poleno_{name}_20",
                              report_path=out / f"left_out_poleno_{name}_20_invalid_events.csv")
    counts = {}
    for name, frame in subsets.items():
        path = out / f"left_out_poleno_{name}_20.csv"
        frame.to_csv(path, index=False)
        counts[name] = {"events_per_species": 20, "images": len(frame),
                        "images_by_species": frame.groupby("species").size().to_dict()}
        print(f"Saved {len(frame):,} images (20 events per species) to {path}", flush=True)
    return {"random_state": random_state, "counts": counts}


def export_splits(rows, out, random_state=42):
    validate_frame_events(rows, context="left-out split input",
                          report_path=out / "left_out_poleno_split_input_invalid_events.csv")
    splits, counts = split_by_species_events(rows, random_state)
    for name, frame in splits.items():
        validate_frame_events(frame, context=f"left_out_poleno_{name}",
                              report_path=out / f"left_out_poleno_{name}_invalid_events.csv")
    for name, frame in splits.items():
        path = out / f"left_out_poleno_{name}.csv"
        frame.to_csv(path, index=False)
        print(f"Saved {len(frame):,} images / {frame.event_id.nunique():,} events to {path}", flush=True)
    report = pd.DataFrame.from_dict(counts, orient="index").rename_axis("species").reset_index()
    for name in ("train", "val", "test"):
        report[name + "_percent"] = 100 * report[name + "_events"] / report["total_events"]
    report.to_csv(out / "left_out_poleno_split_counts.csv", index=False)
    return {"random_state": random_state, "unit": "event_id", "minimum_test_events": 80,
            "minimum_val_events": 20, "target_ratio": {"train": 0.85, "val": 0.05, "test": 0.10},
            "counts_by_species": counts,
            "subsets": export_event_subsets(splits, out, random_state)}


def create(images, labels, out, *, workbook=None):
    if workbook is None:
        raise ValueError("A species workbook is required to exclude deliberately unknown species")
    excluded = exclusion_names(workbook)
    images, labels, out = Path(images), Path(labels), Path(out)
    # Reserve a fresh destination before writing any generated files.
    out.mkdir(parents=True, exist_ok=False)
    mapping = json.loads((images / "dataset_ids.json").read_text(encoding="utf-8"))
    schema = OUTPUT_COLUMNS
    manifest_path = out / "left_out_poleno_image_manifest.json"
    records = []
    def traversal_error(error):
        raise error
    for species in SPECIES:
        before = len(records)
        for dataset, name in mapping.items():
            if name != species:
                continue
            folder = images / dataset
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            for directory, _, files in os.walk(folder, onerror=traversal_error):
                for filename in files:
                    for number in (0, 1):
                        suffix = f".computed_data.holography.image_pairs.0.{number}.rec_mag.png"
                        if filename.endswith(suffix):
                            records.append({"dataset_id": dataset, "rec_path": filename,
                                            "event_id": filename[:-len(suffix)],
                                            "filename": (Path(directory) / filename).relative_to(images).as_posix(),
                                            "species": species, "image_nr": number})
        print(f"{species}: {len(records) - before} actual images", flush=True)
    manifest_path.write_text(json.dumps(records), encoding="utf-8")
    actual = pd.DataFrame(records)
    actual = filter_species_frame(actual, excluded)
    keys = ["dataset_id", "rec_path"]
    if actual.duplicated(keys).any():
        raise ValueError("Duplicate image keys in physical inventory")
    # All retained fields come from the image inventory and label mappings.
    result = actual.sort_values(["species", "dataset_id", "event_id", "image_nr"]).reset_index(drop=True)
    result["root"] = "Poleno25"
    result["intermediate_path"] = "Poleno25"
    result["img_path"] = "Poleno25/" + result["filename"]
    result["species_norm"] = result["species"]
    result["genus"] = result["species"].str.split().str[0]
    enums = json.loads((labels / "label_mappings.json").read_text(encoding="utf-8"))
    additions = {}
    for column in ("dataset_id", "species_norm", "genus"):
        lookup = enums[column]
        additions[column] = {}
        for value in sorted(set(result[column]) - set(lookup)):
            new_id = max(lookup.values(), default=-1) + 1
            lookup[value] = new_id
            additions[column][value] = new_id
        result[column + "_enum"] = result[column].map(lookup)
    # Event enumeration is local to this generated dataset.
    result["event_id_enum"] = pd.factorize(result["event_id"], sort=True)[0]
    missing_columns = set(schema) - set(result.columns)
    if missing_columns:
        raise ValueError(f"Missing output columns: {missing_columns}")
    result = result[schema]
    excluded_normalized = {name.strip().casefold() for name in excluded}
    expected_species = {name for name in SPECIES if name.strip().casefold() not in excluded_normalized}
    if set(result.species) != expected_species:
        raise ValueError("Output species mismatch")
    validate_frame_events(result, context="left_out_poleno_combined",
                          report_path=out / "left_out_poleno_combined_invalid_events.csv")
    require_distinct_images(result.to_dict("records"), result.columns,
                            context="left_out_poleno_combined",
                            report_path=out / "left_out_poleno_combined_invalid_images.csv")
    target = out / "left_out_poleno_combined.csv"
    result.to_csv(target, index=False)
    saved = pd.read_csv(target, low_memory=False)
    assert saved.columns.tolist() == schema
    assert len(saved) == len(actual)
    assert not saved.duplicated(keys).any()
    assert saved[["dataset_id_enum", "species_norm_enum", "genus_enum", "event_id_enum"]].notna().all().all()
    (out / "left_out_poleno_combined_label_mappings.json").write_text(json.dumps(enums, indent=2), encoding="utf-8")
    summary = {"rows": len(result), "species": result.species.nunique(), "columns": len(schema),
               "excluded_species": sorted(excluded),
               "column_names": schema,
               "counts": result.groupby("species").size().to_dict(),
               "new_enum_values": additions, "event_id_enum": "Local, sorted event IDs starting at zero",
               "source_images": str(images), "label_mappings_source": str(labels / "label_mappings.json")}
    summary["splits"] = export_splits(saved, out)
    (out / "left_out_poleno_combined_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
