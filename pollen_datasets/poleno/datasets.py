import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional, Any

from .registry import get_condition_fn


class BaseHolographyImageFolder(torch.utils.data.Dataset):
    
    def __init__(self, root: Optional[str]=None, labels: Optional[str]=None, verbose: bool=False):
        self.root = root
        self.labels = labels
        self.samples = None

        if labels is not None and not isinstance(labels, str):
            raise TypeError("Expected a string or None, got type: {}".format(type(labels).__name__))
        
        if verbose and root is not None:
            print(f"Set image path root: {root}")
        
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
    

    # def _load_image(self, img_path):

    #     full_path = img_path if self.root is None else os.path.join(self.root, img_path)
 
    #     img = Image.open(full_path)
        
    #     if img.mode == 'I;16' or img.mode == 'I':
    #         # Convert to NumPy and scale from [0, 65535] to [0, 1]
    #         img = np.array(img).astype(np.float32)
    #         img /= 65535.0
    #     elif img.mode == 'L':
    #         # 8-bit grayscale, scale from [0, 255] to [0, 1]
    #         img = np.array(img).astype(np.float32)
    #         img /= 255.0
    #     elif img.mode == 'RGB':
    #         # RGB uint8
    #         img = np.array(img, dtype=np.uint8).astype(np.float32)
    #         img /= 255.0
    #     else:
    #         raise ValueError(f"Unsupported image mode: {img.mode}")

    #     return img
    
    def _load_image(self, img_path):

        full_path = img_path if self.root is None else os.path.join(self.root, img_path)

        with Image.open(full_path) as img:
            mode = img.mode
            img = np.array(img).astype(np.float32)


        if mode == 'I;16' or mode == 'I':
            img /= 65535.0
        elif mode == 'L':
            img /= 255.0
        elif mode == 'RGB':
            img /= 255.0
        else:
            raise ValueError(f"Unsupported image mode: {mode}")

        return img
    

    def _convert_condition_val(self, val):
        """
        Universal conversion:
          - ints → int64 tensor (categorical)
          - everything else → float32 tensor (numeric)
        """
        # collapse lists like [3] → 3
        if isinstance(val, (list, np.ndarray)) and len(val) == 1:
            val = val[0]
        # categorical
        if isinstance(val, (np.integer, int)):
            return torch.tensor(val, dtype=torch.int64)
        # numeric vector
        return torch.as_tensor(val, dtype=torch.float32).flatten()


class HolographyImageFolder(BaseHolographyImageFolder):

    def __init__(self, root=None, transform=None, labels=None, dataset_cfg={}, cond_cfg={}, verbose=False):
        self.dataset_cfg = dataset_cfg
        self.cond_cfg = cond_cfg
        self.transform = transform
        super().__init__(root, labels, verbose)


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

            # Add detault value if no columns specified
            if cols is None:
                default_val = cfg.get("default", [])
                self.conditions[name] = [default_val] * len(df)
                continue

            if isinstance(cols, str):
                cols = [cols]

            # For now, just store raw values; convert to tensors in __getitem__
            self.conditions[name] = df[cols].values.tolist()
    

    def __getitem__(self, idx):
        img_path, filename = self.samples[idx]
        img = self._load_image(img_path)

        if self.transform:
            img = self.transform(img)

        cond_dict = {}
        for name, data_list in self.conditions.items():
            raw = data_list[idx]

            # process raw values
            val = self._convert_condition_val(raw)

            cond_dict[name] = val

        return img, cond_dict, filename
    

class PairwiseHolographyImageFolder(BaseHolographyImageFolder):

    def __init__(self, root=None, transform=None, transform1=None, transform2=None, pair_transform=None, labels=None, dataset_cfg={}, cond_cfg={}, verbose=False):
        self.pair_transform = pair_transform
        self.dataset_cfg = dataset_cfg
        self.cond_cfg = cond_cfg
        self.transform1 = transform1 if transform1 else transform
        self.transform2 = transform2 if transform2 else transform
        super().__init__(root, labels, verbose)


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

                # Add detault value if no columns specified
                if cols is None:
                    default_val = cfg.get("default", [])
                    self.conditions[name].append((default_val, default_val))
                    continue

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
        if self.transform1:
            img1 = self.transform1(img1)
        if self.transform2:
            img2 = self.transform2(img2)

        # pair-dependent transform
        meta = {}
        if self.pair_transform:
            img1, img2, meta = self.pair_transform(img1, img2)

        # Build conditioning dicts
        cond_dict1, cond_dict2 = {}, {}
        cond_dict1["meta"] = meta
        cond_dict2["meta"] = meta

        if hasattr(self, "conditions"):
            for name, pair_list in self.conditions.items():
                raw1, raw2 = pair_list[idx]

                if meta.get("swapped", False): # swap condition
                    raw1, raw2 = raw2, raw1

                # process raw values
                val1 = self._convert_condition_val(raw1)
                val2 = self._convert_condition_val(raw2)

                # apply conditioning transform function (if exists)
                func_name = self.cond_cfg["encoders"][name].get("condition_fn", None)
                if func_name is not None:
                    fn = get_condition_fn(func_name)
                    val1, val2 = fn(val1, val2, meta)

                cond_dict1[name] = val1
                cond_dict2[name] = val2

        # If images are swapped also swapp filenames
        if meta.get("swapped", False):
            name1, name2 = name2, name1

        # return consistent structure
        return (img1, img2), (cond_dict1, cond_dict2), (name1, name2)