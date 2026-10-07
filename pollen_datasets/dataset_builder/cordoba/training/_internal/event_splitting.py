"""Split and sample whole events while preserving their image rows."""

import math

import numpy as np

import pandas as pd


def split_by_species_events(rows, random_state=42):
    """Split unique events per species, keeping all their images together.

    Test receives max(80, ceil(10%)); validation receives max(20, ceil(5%)).
    Training receives the remainder. Fail if no training event can remain.
    Shared event IDs across species/datasets retain the same assignment.
    Smaller species are assigned first; remaining events fill each target.
    Raise if overlapping labels make these target counts incompatible with
    prior assignments. The input schema and original label IDs are preserved.
    """
    if rows.empty:
        raise ValueError("Cannot split an empty dataset.")
    if rows[["species", "event_id"]].isna().any().any():
        raise ValueError("Species and event IDs must not be missing.")
    event_labels = rows[["event_id", "species"]].drop_duplicates()
    rng = np.random.default_rng(random_state)
    assignments = {}
    counts = {}
    sizes = event_labels.groupby("species").size()
    for species in sorted(sizes.index, key=lambda name: (sizes[name], name)):
        events = np.array(sorted(event_labels.loc[event_labels["species"].eq(species), "event_id"]))
        total = len(events)
        n_test = max(80, math.ceil(total * 0.10))
        n_val = max(20, math.ceil(total * 0.05))
        n_train = total - n_test - n_val
        if n_train < 1:
            raise ValueError(f"{species}: {total} events cannot provide {n_test} test, {n_val} validation, and at least 1 training event.")
        remaining = np.array([event for event in events if event not in assignments])
        rng.shuffle(remaining)
        targets = {"train": n_train, "val": n_val, "test": n_test}
        deficits = {split: target - sum(assignments.get(event) == split for event in events)
                    for split, target in targets.items()}
        if min(deficits.values()) < 0:
            raise ValueError(f"{species}: shared event assignments exceed a split target; cannot meet exact targets without leaking events.")
        test_stop = deficits["test"]
        val_stop = test_stop + deficits["val"]
        for split, selected in (
            ("test", remaining[:test_stop]),
            ("val", remaining[test_stop:val_stop]),
            ("train", remaining[val_stop:]),
        ):
            assignments.update(dict.fromkeys(selected, split))
        counts[species] = {"total_events": total, "train_events": n_train,
                           "val_events": n_val, "test_events": n_test}
    roles = rows["event_id"].map(assignments)
    splits = {name: rows.loc[roles.eq(name)].copy() for name in ("train", "val", "test")}
    assert sum(len(frame) for frame in splits.values()) == len(rows)
    for species, expected in counts.items():
        for name, frame in splits.items():
            assert frame.loc[frame.species.eq(species), "event_id"].nunique() == expected[name + "_events"]
    for name, frame in splits.items():
        for species, count in frame.groupby("species").size().items():
            counts[species][name + "_images"] = int(count)
    return splits, counts


def sample_species_events(rows, events_per_species=20, random_state=42):
    """Select events within each species, retaining every image of each event."""
    if rows.empty or rows[["species", "event_id"]].isna().any().any():
        raise ValueError("Subset input must contain non-missing species and event IDs.")
    if events_per_species < 1:
        raise ValueError("events_per_species must be positive.")
    rng = np.random.default_rng(random_state)
    selected = pd.Series(False, index=rows.index)
    for species in sorted(rows.species.unique()):
        mask = rows.species.eq(species)
        events = np.array(sorted(rows.loc[mask, "event_id"].unique()))
        if len(events) < events_per_species:
            raise ValueError(f"{species}: only {len(events)} events; need {events_per_species} for the subset.")
        chosen = rng.choice(events, size=events_per_species, replace=False)
        # Match species too: event IDs may be shared by different species.
        selected |= mask & rows.event_id.isin(chosen)
    return rows.loc[selected].copy()


