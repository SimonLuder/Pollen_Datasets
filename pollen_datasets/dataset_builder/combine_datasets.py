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

    df = df.dropna(subset=["image_nr"])

    # Identify valid (root, dataset_id) groups that have both 0 and 1 images
    group_cols = ["root", "dataset_id", "event_id"]
    valid_pairs = (
        df.groupby(group_cols)["image_nr"]
        .apply(lambda x: set(x) == {0, 1})
        .reset_index(name="has_both")
    )

    # Keep only groups that contain both 0 and 1
    df = df.merge(valid_pairs, on=group_cols, how="left")
    wtftest = df[~df["has_both"]]
    print("Test1", wtftest.loc[wtftest["dataset_id"].isin(isolated_ids)].value_counts("dataset_id") / 2)
    df = df[df["has_both"]]

    print("Test1", df.loc[df["dataset_id"].isin(isolated_ids)].value_counts("dataset_id") / 2)

    # For events that appear in multiple dataset_ids (duplicate folders)
    # → keep only the first valid (root, dataset_id) group per event_id
    first_valid = (
        df.drop_duplicates(subset=group_cols)
        .groupby("event_id")
        .first()
        .reset_index()[group_cols]
    )

    print("test2", df.loc[df["dataset_id"].isin(isolated_ids)].value_counts("dataset_id") / 2)

    # Merge back to retain only the selected groups
    df_cleaned = df.merge(first_valid, on=group_cols, how="inner")

    # Optionally drop helper column
    df_cleaned = df_cleaned.drop(columns="has_both", errors="ignore")

    # Drop event_ids with more than two entries
    df_cleaned = drop_events_with_more_than_two(df_cleaned)

    return df_cleaned


def cleanup_columns(df):
    # df["species"] = df["species"].fillna(df["label"]) # species
    df["genus"] = df["genus"].fillna(df["species"].str.split().str[0]) # genus
    df["filename"] = df["filename"].fillna(df.apply(lambda x: os.path.join(x["dataset_id"], x["rec_path"]), axis=1)) # filename
    # df = df.drop(columns="label") # drop labels column
    return df


def combine(df1, df2, save_as=None):

    # print("df1 len:", df1)
    print("df2 len:", df2)

    # combined = pd.concat([df1, df2], ignore_index=True)
    combined = df2.copy()

    print(combined.loc[combined["dataset_id"].isin(isolated_ids)].value_counts("dataset_id") / 2)

    # Remove duplicate rows
    combined = remove_duplicate_event_copies(combined)

    combined = cleanup_columns(combined)

    if save_as is not None:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        combined.to_csv(save_as)

    print("combined len:", combined)

    return combined


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for dataset combinatiion')
    # Old poleno labels
    # parser.add_argument('--file1', default='./data/processed/poleno/computed_data_full_re.csv', type=str)
    parser.add_argument('--root1', default='Z:/marvel/marvel-fhnw/data/Poleno', type=str)
    # New poleno labels
    parser.add_argument('--file2', default='Z:/simon_luder/Data_Setup/Pollen_Datasets/data/processed/Poleno_25/poleno_25_labels.csv', type=str)
    # Output file
    parser.add_argument('--save_as', default='data/final/poleno/poleno_labels_clean_why_me.csv', type=str)

    args = parser.parse_args()
    
    # df1 = pd.read_csv(args.file1)
    # df1["root"] = args.root1

    df2 = pd.read_csv(args.file2)

    # print(len(df1))
    # df1 = df1[~df1["rec_path"].isin(df2["rec_path"])] # drop_all in df1 that are in df2
    # print(len(df1))

    isolated_ids = {
    "11ed827a-01e5-3372-88b0-66f2ec8a65cb",
    "11f03faf-063f-ed0a-8380-1e119433b62f",
    "11f0b0b7-42aa-4ec4-a17f-1e119433b62f",
    "11f07e8c-1270-c8d0-92f5-1e119433b62f",
    "11f0c084-3a8d-4ebc-a344-1e119433b62f",
    "11f07dc3-dd16-7e28-a558-1e119433b62f",
    "11f04c4a-1031-482a-86d8-1e119433b62f",
    "11f04110-1c47-3e02-b5f1-1e119433b62f",
    "11f0c088-4571-4320-bf3f-1e119433b62f",
    "11f0c088-8e66-6858-8271-1e119433b62f",
    "11f03626-645b-b9a6-abc6-1e119433b62f",
    "11f037dc-955c-2678-89fb-1e119433b62f",
    }


    combine(df1=None, df2=df2, save_as=args.save_as)