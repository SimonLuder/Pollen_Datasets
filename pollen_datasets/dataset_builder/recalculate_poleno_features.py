'''
This script is used to recalculate the visual features from the pollen objects in the holographic images. 

Two arguments are required for the recalculation:

    - database: path of the poleno_marvel.db file
    - image_folder: path to the folder containing the holographic images

The default path configuration is setup to run inside the singularity container. To run it localy the two precious paths need to be adapted.
'''

import os
import sys
import sqlite3
import logging
import argparse
import pandas as pd


# add base dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from pollen_datasets.dataset_builder.holographic_features import recalculate_holographic_features

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for poleno labels recalculation')
    parser.add_argument('--database', default='Z:/marvel/marvel-fhnw/data/Poleno/poleno_marvel_old.db', type=str)
    parser.add_argument('--image_folder', default='Z:/marvel/marvel-fhnw/data/Poleno', type=str)
    parser.add_argument('--logging', default=None, type=str)
    args = parser.parse_args()

    # Configure logging
    if args.logging is not None:
        handlers = [
                logging.FileHandler(args.logging, mode='a', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
    else:
        handlers = [
                logging.StreamHandler(sys.stdout),
            ]
   
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )

    logging.debug(f"Connecting to {args.database}")

    # Connect to the SQLite database
    conn = sqlite3.connect(args.database)

    # Get table computed_data_full as dataframe

    df_computed_data_full = pd.read_sql_query("SELECT * FROM computed_data_full", conn)

    # Close the database connection
    conn.close()

    logging.info(f"Loaded {len(df_computed_data_full)} entries from {args.database}")

    # Recalculate features for all images in the dataframe
    computed_data_full_recalc = recalculate_holographic_features(df_computed_data_full, args.image_folder)

    logging.info(f"Recalculated regionprops features.")

    # Save the dataframe
    computed_data_full_recalc.to_csv("computed_data_full_re.csv", index=False)