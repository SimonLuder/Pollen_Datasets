import os
import argparse
import pandas as pd

def drop_events_with_more_than_two(df, id_col="event_id"):
    """
    Drop entire events that have more than two rows (images).
    """
    # Count how many images each event_id has
    event_counts = df.groupby(id_col).size()

    # Keep only event_ids that have <= 2 entries
    valid_event_ids = event_counts[event_counts <= 2].index

    # Filter the DataFrame
    df_filtered = df[df[id_col].isin(valid_event_ids)].copy()

    return df_filtered

def remove_duplicate_event_copies(df):
    """
    Remove redundant duplicate event copies while keeping one complete (0 & 1) pair per event_id.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: ['root', 'dataset_id', 'event_id', 'image_nr']

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame where for each event_id only one (root, dataset_id) pair remains,
        containing both image_nr 0 and 1 if available.
    """

    df = df.copy()

    # Identify valid (root, dataset_id) groups that have both 0 and 1 images
    group_cols = ["root", "dataset_id", "event_id"]
    valid_pairs = (
        df.groupby(group_cols)["image_nr"]
        .apply(lambda x: set(x) == {0, 1})
        .reset_index(name="has_both")
    )

    # Keep only groups that contain both 0 and 1
    df = df.merge(valid_pairs, on=group_cols, how="left")
    df = df[df["has_both"]]

    # For events that appear in multiple dataset_ids (duplicate folders)
    # → keep only the first valid (root, dataset_id) group per event_id
    first_valid = (
        df.drop_duplicates(subset=group_cols)
        .groupby("event_id")
        .first()
        .reset_index()[group_cols]
    )

    # Merge back to retain only the selected groups
    df_cleaned = df.merge(first_valid, on=group_cols, how="inner")

    # Optionally drop helper column
    df_cleaned = df_cleaned.drop(columns="has_both", errors="ignore")

    # Drop event_ids with more than two entries
    df_cleaned = drop_events_with_more_than_two(df_cleaned)

    return df_cleaned


def cleanup_columns(df):
    df["species"] = df["species"].fillna(df["label"]) # species
    df["genus"] = df["genus"].fillna(df["species"].str.split().str[0]) # genus
    df["filename"] = df["filename"].fillna(df.apply(lambda x: os.path.join(x["dataset_id"], x["rec_path"]), axis=1)) # filename
    df = df.drop(columns="label") # drop labels column
    return df


def combine(df1, df2, save_as=None):

    combined = pd.concat([df1, df2], ignore_index=True)

    # Remove duplicate rows
    combined = remove_duplicate_event_copies(combined)

    combined = cleanup_columns(combined)

    if save_as is not None:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        combined.to_csv(save_as)

    return combined


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for dataset combinatiion')
    # Old poleno labels
    parser.add_argument('--file1', default='./data/processed/poleno/computed_data_full_re.csv', type=str)
    parser.add_argument('--root1', default='Z:/marvel/marvel-fhnw/data/Poleno', type=str)
    # New poleno labels
    parser.add_argument('--file2', default='./data/processed/poleno_25/poleno_25_labels.csv', type=str)
    parser.add_argument('--root2', default='Z:/marvel/marvel-fhnw/data/Poleno25', type=str)
    # Output file
    parser.add_argument('--save_as', default='data/final/poleno/poleno_labels_clean.csv', type=str)

    args = parser.parse_args()
    
    df1 = pd.read_csv(args.file1)
    df1["root"] = args.root1

    df2 = pd.read_csv(args.file2)
    df2["root"] = args.root2

    combine(df1, df2, save_as=args.save_as)


