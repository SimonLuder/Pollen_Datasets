import argparse
import logging
import pandas as pd
from pollen_datasets.dataset_builder.holographic_features import recalculate_holographic_features


REGIONPROPS_COLS = [
    "area","bbox_area","convex_area","eccentricity","equivalent_diameter",
    "feret_diameter_max","major_axis_length","minor_axis_length",
    "max_intensity","min_intensity","mean_intensity","orientation",
    "perimeter","perimeter_crofton","solidity"
]


def recalc_regionprops(input_csv, images_root, output_csv, intermediate_path=None):
    """
    Recalculate region properties while keeping all other dataframe columns unchanged.
    """

    logging.info(f"Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)

    logging.info("Recalculating holographic region properties...")
    
    # Compute features on a copy
    df_features = recalculate_holographic_features(df.copy(), images_root, intermediate_path)

    logging.info("Updating region property columns...")

    # Replace only regionprops columns
    for col in REGIONPROPS_COLS:
        if col in df_features.columns:
            df[col] = df_features[col]

    logging.info(f"Saving updated dataset → {output_csv}")
    df.to_csv(output_csv, index=False)

    logging.info("Finished successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Recalculate region properties in a Poleno dataset")

    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to existing dataset CSV"
    )

    parser.add_argument(
        "--images_root",
        required=True,
        help="Root folder containing image data"
    )

    parser.add_argument(
        "--output_csv",
        default="updated_regionprops.csv",
        help="Output CSV file"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    recalc_regionprops(
        args.input_csv,
        args.images_root,
        args.output_csv
    )