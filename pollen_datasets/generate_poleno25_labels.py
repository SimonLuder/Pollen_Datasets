import os
import re
import sys
import json
import logging
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
            logging.info("No CSV files found.")
        else:
            logging.info(f"Found {len(csv_files)} CSV files. Combining...")
            
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



# -------------------- Further data setup functions --------------------

def image_nr_from_rec_path(rec_path):
    """Get the image number from the rec_path"""
    pattern = re.compile(r'image_pairs\.\d+\.(\d+)\.rec_mag')
    return int(pattern.search(rec_path).group(1)) if pattern.search(rec_path) else None


def species_from_dataset_id(dataset_id, id_to_species):
    """Get the species from the dataset_id"""
    return id_to_species.get(dataset_id, None)


def genus_from_species(species):
    """Get the genus from the species"""
    return species.split(" ")[0] if species is not None else None


def add_annotations(df, dataset_ids_file):
    """Add rest of annotation columns to the dataset DataFrame."""
    # dataset_id to species mapper
    with open(dataset_ids_file, "r") as f:
        id_to_species = json.load(f)
    # Additional feature columns
    df["species"]  = df["dataset_id"].apply(species_from_dataset_id, id_to_species=id_to_species)
    df["genus"]    = df["species"].apply(genus_from_species)
    df["image_nr"] = df["rec_path"].apply(image_nr_from_rec_path)
    return df


def generate_labels(images_root, labels_folder, out_file, dataset_ids_file, force_recalc):

    regionprops_cols = [
        "area","bbox_area","convex_area","eccentricity","equivalent_diameter","feret_diameter_max",
        "major_axis_length","minor_axis_length","max_intensity","min_intensity","mean_intensity",
        "orientation","perimeter","perimeter_crofton","solidity"
        ]
    
    annotations_cols = [
        "image_nr", "species", "genus"
        ]
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("generate_poleno25_labels.log", mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Get current databasis
    df = get_datasets_as_df(labels_folder)
    if df is not None:
        already_searched = list(set(df["dataset_id"]))
    else:
        already_searched = []

    logging.info(f"Searching in {images_root}, ignore folders: {already_searched}")
    # Search missing images
    setup = DataSetup()
    setup.search_images_in_folder(
        root=images_root, 
        ignore=already_searched,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setup.save_as_csv(os.path.join(labels_folder, f"collection_{timestamp}.csv"))
    logging.info("Done")

    # Get all csv files
    all_csv = [os.path.join(labels_folder, file) for file in os.listdir(labels_folder) if file.endswith(".csv")]
    logging.info(f"Found csv's: {all_csv}")

    # Itterate over all csv files separately
    for csv in all_csv:

        df = pd.read_csv(csv)
        regionprops_complete, _ = has_columns(df, regionprops_cols)
        annotations_complete, _ = has_columns(df, annotations_cols)

        if force_recalc or not regionprops_complete:
            
            # Recalculate features for all images in the dataframe
            logging.info(f"Recalculate regionprops for {csv}")
            df = recalculate_holographic_features(df, images_root)

            # Save the dataframe
            df.to_csv(csv, index=False)

        if force_recalc or not annotations_complete:

            # Add image annotation labels
            logging.info(f"Adding annotations to {csv}")
            df = add_annotations(df, dataset_ids_file)

            # Save the dataframe
            df.to_csv(csv, index=False)
        
    # Get updated databasis and save as single csv
    df = get_datasets_as_df(labels_folder)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    logging.info(f"Created final csv for {out_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for resnet training')
    parser.add_argument('--labels', default="./data/processed/poleno_25/temp/", type=str)
    parser.add_argument('--root', default='Z:/marvel/marvel-fhnw/data/Poleno_25', type=str)
    parser.add_argument('--filename', default='data/processed/poleno_25/poleno_25_labels.csv', type=str)
    parser.add_argument('--force_recalc', default=False, type=bool)
    parser.add_argument('--dataset_ids', default="Z:\marvel\marvel-fhnw\data\Poleno_25\dataset_ids.json", type=str)
    args = parser.parse_args()

    generate_labels(
        args.root, 
        labels_folder=args.labels, 
        out_file=args.filename, 
        dataset_ids_file=args.dataset_ids, 
        force_recalc=args.force_recalc, 
        )

