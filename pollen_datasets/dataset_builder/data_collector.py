import os
import pickle
import sqlite3
import pandas as pd
from pathlib import Path

class DataSetup:
    """
    Class to setup datasets by searching images in folders and downloading tables from a sqlite database.
    """
    def __init__(self):
        self.foldername_to_id = dict()
        self.samples = None


    def generate_dataset(self, root, save_as=None):
        self.search_images_in_folder(root, save_as)
        if save_as is not None and save_as.lower().endswith((".pickle", ".pkl")):
            self.save_as_pickle(save_as)
        if save_as is not None and save_as.lower().endswith(".csv"):
            self.save_as_csv(save_as)


    def search_images_in_folder(self, root, ignore=None):
        """
        Recursively search for images in a root folder and its subfolders.

        Args:
            root (str): Root folder path.
            ignore (list, optional): Folder names to ignore. Defaults to [].
        """
        if ignore is None:
            ignore = set()
        else:
            ignore = set(ignore)

        valid_exts = {".png", ".jpg", ".jpeg"}
        self.samples = []
        self.foldername_to_id.clear()
        index = 0

        # Use scandir-based recursion for high performance
        def _scan_dir(dirpath):
            nonlocal index
            try:
                with os.scandir(dirpath) as it:
                    for entry in it:
                        # Skip ignored folders early
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in ignore:
                                print(f"Skipping {entry.name}")
                                continue
                            print(f"Searching {entry.name}")
                            _scan_dir(entry.path)
                        elif entry.is_file():
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in valid_exts:
                                dataset_id = os.path.basename(os.path.dirname(entry.path))
                                if dataset_id not in self.foldername_to_id:
                                    self.foldername_to_id[dataset_id] = index
                                    index += 1
                                rec_path = entry.name
                                event_id = self.event_id_from_rec_path(rec_path)
                                rel_path = os.path.relpath(entry.path, root)
                                self.samples.append((event_id, dataset_id, rec_path, rel_path))

            except PermissionError:
                # Skip folders without access
                pass

        _scan_dir(root)


    def event_id_from_rec_path(self, rec_path):
        """Extract event_id from the rec_path"""
        idx = rec_path.find("_ev")
        event_id = rec_path[:idx + len("_ev")]
        return event_id


    def save_as_pickle(self, save_as):
        """Save the searched samples as pickle file"""
        Path(save_as).parent.mkdir(parents=True, exist_ok=True)
        with open(save_as, 'wb') as f:
            pickle.dump(self.samples, f)


    def save_as_csv(self, save_as):
        """Save the searched samples as csv file"""
        if len(self.samples) > 0:
            Path(save_as).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.samples).to_csv(save_as, index=False, header=["event_id", "dataset_id", "rec_path", "filename"])
            print(f"Saved {len(self.samples)} new entries as {save_as}.")
        else:
            print("No new entries found to save.")


    def download_tables_from_db(self, db_path, csv_dir):
        Path(csv_dir).mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        # Get all tables names
        all_table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        # Download tables
        for _, table in all_table_names.iterrows():
            if not os.path.isfile(os.path.join(csv_dir, f"{table['name']}.csv")):
                print(f"Downloading table: {table['name']}")
                df = pd.read_sql_query(f"SELECT * FROM {table['name']}", conn)
                df.to_csv(os.path.join(csv_dir, f"{table['name']}.csv"), index=False)
                pass
        conn.close()

    def preprocess(self, df):
        df = self.remove_invalid_entries(df)
        df = self.create_addidional_labels(df)
        return df
    

    def create_addidional_labels(self, df):
        df["filenames"] = df["dataset_id"] + "/" + df["rec_path"]
        return df


    def normalize(self, df):
        return (df-df.mean())/df.std()


    def remove_invalid_entries(self, df):
        # Remove rows with NaN values
        df = df.dropna()

        # Keep only event_ids with exactly two entries
        event_id_count = df["event_id"].value_counts()
        valid_samples = event_id_count[event_id_count == 2].index
        df = df.loc[df["event_id"].isin(valid_samples)]
        return df

