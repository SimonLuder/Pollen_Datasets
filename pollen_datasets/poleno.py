import os
import re
import torch
import pickle
import sqlite3
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Optional, Any

class BaseHolographyImageFolder(torch.utils.data.Dataset):
    
    def __init__(self, root, transform=None, labels: Optional[str]=None, config=None, verbose: bool=False):
        self.root = root
        self.transform = transform
        self.config = config
        self.labels = labels
        self.cond_imgs = None
        self.tabular_features = None
        self.class_labels = None
        self.samples = None

        if labels is not None and not isinstance(labels, str):
            raise TypeError("Expected a string or None, got type: {}".format(type(labels).__name__))
        
        if (labels is not None):
            if os.path.exists(labels):

                if labels.endswith(".csv"):
                    if verbose:
                        print("Loading dataset from csv")
                    self.load_annotations_from_csv()

                if labels.endswith(".pkl"):
                    if verbose:
                        print("Loading dataset from pickle")
                    self.load_annotations_from_pickle()       

            else:
                raise FileNotFoundError(f"Labels file {labels} does not exist.")


    def load_annotations_from_csv(self):
        raise NotImplementedError("Method not implemented for this dataset.")


    def load_annotations_from_pickle(self):
        raise NotImplementedError("Method not implemented for this dataset.")


    def search_images_in_folder(self):
        raise NotImplementedError("Method not implemented for this dataset.")


    def __len__(self):
        return len(self.samples)
    

    def _load_image(self, img_path):
        img = Image.open(os.path.join(self.root, img_path))
        
        if img.mode == 'I;16' or img.mode == 'I':
            # Convert to NumPy and scale from [0, 65535] to [0, 1]
            img = np.array(img).astype(np.float32) / 65535.0
        elif img.mode == 'L':
            # 8-bit grayscale, scale from [0, 255] to [0, 1]
            img = np.array(img).astype(np.float32) / 255.0

        return img


class HolographyImageFolder(BaseHolographyImageFolder):

    def __init__(self, root, transform=None, labels=None, config=None, verbose=False):
        """
        Here, `config` is the full config dict or at least:
          {
            "dataset": ...,
            "conditioning": ...
          }
        """
        self.full_config = config
        self.dataset_cfg = config["dataset"]
        self.cond_cfg = config.get("conditioning", {})
        super().__init__(root, transform, labels, config, verbose)


    def load_annotations_from_csv(self):
        df = pd.read_csv(self.labels)

        # image path + filename like before
        filename_column = self.dataset_cfg.get("filenames", "filename")
        image_folder_col = self.dataset_cfg.get("img_path", "img_path")
        self.samples = list(zip(df[image_folder_col], df[filename_column]))

        # generic condition storage
        self.conditions: dict[str, list] = {}

        enabled = self.cond_cfg.get("enabled", "") # enabled encoders
        enabled_names = [c for c in enabled.replace(" ", "").split("+") if c]

        enc_cfgs = self.cond_cfg.get("encoders", {})

        for name in enabled_names:
            if name not in enc_cfgs:
                raise KeyError(f"Condition '{name}' enabled but not defined in conditioning.encoders")
            cfg = enc_cfgs[name]
            cols = cfg["use_columns"]
            if isinstance(cols, str):
                cols = [cols]

            # For now, just store raw values; convert to tensors in __getitem__
            self.conditions[name] = df[cols].values.tolist()


    def __getitem__(self, idx):
        img_path, filename = self.samples[idx]
        img = self._load_image(img_path)

        if self.transform:
            img = self.transform(img)

        cond_dict: dict[str, Any] = {}
        if hasattr(self, "conditions"):
            for name, data_list in self.conditions.items():
                val = data_list[idx]
                # if it's a single column, collapse list -> scalar
                if isinstance(val, (list, np.ndarray)) and len(val) == 1:
                    val = val[0]
                cond_dict[name] = val

        return img, cond_dict, filename
    

class PairwiseHolographyImageFolder(BaseHolographyImageFolder):

    def __init__(self, root, transform=None, pair_transform=None, labels=None, config=None, verbose=False):
        """
        config must be the SAME full config dict used for the single-image dataset.
        """
        self.pair_transform = pair_transform
        self.full_config = config
        self.dataset_cfg = config["dataset"]
        self.cond_cfg = config.get("conditioning", {})
        super().__init__(root, transform, labels, config, verbose)


    def load_annotations_from_csv(self):
        df = pd.read_csv(self.labels)

        # Column names
        filename_col = self.dataset_cfg.get("filenames", "filename")
        imgpath_col = self.dataset_cfg.get("img_path", "img_path")
        particle_id_col = self.dataset_cfg.get("particle_id", "event_id")

        # Group by particle identifier to get pairs
        grouped = df.groupby(particle_id_col)

        self.samples = []             # list of ((img1_path, fn1), (img2_path, fn2))
        self.conditions = {}          # dict[str, list[tuple(val1, val2)]]

        # Determine which conditioning types are active
        enabled = self.cond_cfg.get("enabled", "")
        enabled_names = [c for c in enabled.replace(" ", "").split("+") if c]

        # Access encoder configs
        enc_cfgs = self.cond_cfg.get("encoders", {})

        # Initialize conditioning lists
        for name in enabled_names:
            if name not in enc_cfgs:
                raise KeyError(
                    f"Condition '{name}' is enabled but not defined under conditioning.encoders."
                )
            self.conditions[name] = []

        # Build paired samples
        for event_id, group in grouped:
            if len(group) != 2:
                raise ValueError(f"Event ID {event_id} does not have exactly 2 rows")

            group = group.sort_index().reset_index(drop=True)

            # Paired sample
            sample1 = (group.loc[0, imgpath_col], group.loc[0, filename_col])
            sample2 = (group.loc[1, imgpath_col], group.loc[1, filename_col])
            self.samples.append((sample1, sample2))

            # Add conditioning vals for each encoder
            for name in enabled_names:
                cfg = enc_cfgs[name]
                cols = cfg["use_columns"]
                if isinstance(cols, str):
                    cols = [cols]

                # Extract values for each row
                val1 = group.loc[0, cols].values.tolist() if len(cols) > 1 else group.loc[0, cols[0]]
                val2 = group.loc[1, cols].values.tolist() if len(cols) > 1 else group.loc[1, cols[0]]

                # Store pair (val1, val2)
                self.conditions[name].append((val1, val2))


    def __getitem__(self, idx):
        # image pair
        (path1, name1), (path2, name2) = self.samples[idx]

        img1 = self._load_image(path1)
        img2 = self._load_image(path2)

        # single-image transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # pair-dependent transform
        meta = {}
        if self.pair_transform:
            img1, img2, meta = self.pair_transform(img1, img2)

        # Build conditioning dict
        cond_dict = {}
        if hasattr(self, "conditions"):
            for name, pair_list in self.conditions.items():
                val1, val2 = pair_list[idx]

                if meta.get("swapped", False): # swap condition
                    val1, val2 = val2, val1

                cond_dict[name] = (val1, val2)

        if meta.get("swapped", False):
            name1, name2 = name2, name1

        # return consistent structure
        return (img1, img2), cond_dict, (name1, name2)
    

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

