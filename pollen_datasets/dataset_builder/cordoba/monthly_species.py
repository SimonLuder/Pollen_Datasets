"""Export the explicit active/known flags in a monthly species workbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._internal.species_workbook import MONTHS, read_species_workbook


def load_monthly_species(
    workbook_path: str | Path,
    *,
    known_only: bool = True,
    sheet_name: str = "Monthly labels",
) -> dict[str, list[str]]:
    """Return active species by month, optionally requiring known == yes.

    Read month/species/active/known columns. Flags must be yes/no (ignoring
    case and surrounding spaces). Preserve species spelling, remove duplicate
    entries, and sort each list. All twelve English month names are returned;
    months without selected species have empty lists. No genus matching or
    inference from the workbook's other sheets is performed.
    """
    if not isinstance(known_only, bool):
        raise TypeError("known_only must be a bool.")
    selected = {month: set() for month in MONTHS}
    for row in read_species_workbook(workbook_path, monthly=True, sheet_name=sheet_name):
        if row["active"] and (not known_only or row["known"]):
            selected[row["month"]].add(row["species"])
    return {month: sorted(names) for month, names in selected.items()}


def export_monthly_species(
    workbook_path: str | Path,
    output_path: str | Path,
    *,
    known_only: bool = True,
    sheet_name: str = "Monthly labels",
) -> dict[str, list[str]]:
    """Read the workbook, write a month-to-species JSON object, and return it.

    Existing output JSON is replaced so exports can be regenerated after edits.
    """
    output = Path(output_path)
    if output.suffix.lower() != ".json":
        raise ValueError("output_path must have a .json extension.")
    if output.resolve() == Path(workbook_path).resolve():
        raise ValueError("Output must differ from the input workbook.")
    result = load_monthly_species(workbook_path, known_only=known_only, sheet_name=sheet_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-unknown", action="store_true",
                        help="Include all active species, regardless of their known flag.")
    parser.add_argument("--sheet", default="Monthly labels")
    args = parser.parse_args(argv)
    result = export_monthly_species(
        args.workbook, args.output, known_only=not args.include_unknown, sheet_name=args.sheet,
    )
    print(f"Saved {sum(map(len, result.values()))} species-month entries to {args.output}")


if __name__ == "__main__":
    main()
