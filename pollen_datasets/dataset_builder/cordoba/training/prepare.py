"""Prepare Poleno finetuning CSVs without copying images or training models."""

import argparse
from pathlib import Path

from ._internal.create_left_out_poleno import create
from ._internal.create_finetuning_csvs import build
from ._internal.inventory_csv_species import inventory_csv_species


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="Report species present in existing CSVs")
    inventory.add_argument("--datasets", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    command = commands.add_parser(
        "all", help="Create missing-species splits and merge them into existing datasets")
    command.add_argument("--datasets", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True,
                         help="Fresh output folder (must not exist)")
    command.add_argument("--species-workbook", type=Path)
    command.add_argument("--images", type=Path, required=True,
                         help="Poleno25 folder containing dataset_ids.json")
    command.add_argument("--labels", type=Path,
                         help="Folder containing label_mappings.json; defaults to datasets' parent")
    args = parser.parse_args(argv)
    if args.command == "inventory":
        inventory_csv_species(args.datasets, args.output)
        return
    workbook = args.species_workbook
    if workbook is None:
        candidate = args.datasets / "poleno_monthly_species.xlsx"
        if candidate.exists():
            workbook = candidate
    if workbook is None:
        raise ValueError("A species workbook is required to exclude deliberately unknown species. "
                         "Supply --species-workbook PATH or place poleno_monthly_species.xlsx "
                         "in --datasets.")
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    create(args.images, args.labels or args.datasets.parent, args.output,
           workbook=workbook)
    build(args.datasets, workbook, left_folder=args.output,
          output=args.output / "finetuning")


if __name__ == "__main__":
    main()
