import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from pollen_datasets.poleno import DataSetup

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from pollen_datasets.holographic_features import recalculate_holographic_features


def get_datasets_as_df(folder):
    """Get all .csv  files in a given folder and combine them to a single dataframe"""
    if os.path.exists(folder):
        # Get list of all CSV files in the directory
        csv_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".csv")
        ]

        if not csv_files:
            print("No CSV files found.")
        else:
            print(f"Found {len(csv_files)} CSV files. Combining...")

            # Read and combine all CSVs
            df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
            return df
        
def has_columns(df, required_cols):
    """
    Checks if a DataFrame contains all specified columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame to check.
    - required_cols (list or set): Column names to verify.

    Returns:
    - bool: True if all required columns are present, False otherwise.
    - list: A list of missing columns (empty if none are missing).
    """
    missing = [col for col in required_cols if col not in df.columns]
    return len(missing) == 0, missing
        

def generate_labels(images_root, labels_folder, filename, force_recalc):

    feature_cols = [
        "area","bbox_area","convex_area","eccentricity","equivalent_diameter","feret_diameter_max",
        "major_axis_length","minor_axis_length","max_intensity","min_intensity","mean_intensity",
        "orientation","perimeter","perimeter_crofton","solidity"
        ]

    # Get current databasis
    df = get_datasets_as_df(labels_folder)
    if df is not None:
        already_searched = list(set(df["dataset_id"]))
    else:
        already_searched = []
    
    ignore = already_searched 

    # Search missing images
    setup = DataSetup()
    setup.search_images_in_folder(
        root=images_root, 
        ignore=already_searched,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setup.save_as_csv(os.path.join(labels_folder, f"collection_{timestamp}.csv"))

    # Get all csv files
    all_csv = [os.path.join(labels_folder, file) for file in os.listdir(labels_folder) if file.endswith(".csv")]
    print(all_csv)

    # Itterate over all csv files separately
    for csv in all_csv:

        df = pd.read_csv(csv)
        complete, _ = has_columns(df, feature_cols)

        if force_recalc or not complete:
            # Recalculate features for all images in the dataframe
            computed_data_full_recalc = recalculate_holographic_features(df, images_root)
            # Save the dataframe
            computed_data_full_recalc.to_csv(filename, index=False)

    # Get updated databasis
    df = get_datasets_as_df(labels_folder)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for resnet training')
    parser.add_argument('--labels', default="./data/processed/poleno_25/", type=str)
    parser.add_argument('--root', default='Z:/marvel/marvel-fhnw/data/Poleno_25', type=str)
    parser.add_argument('--filename', default='poleno_25_labels.csv', type=str)
    parser.add_argument('--force_recalc', default=True, type=bool)
    args = parser.parse_args()

    generate_labels(args.root, args.labels, args.filename, args.force_recalc)

