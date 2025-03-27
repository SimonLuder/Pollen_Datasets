import os
import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Optional

class BaseHolographyImageFolder(torch.utils.data.Dataset):
    
    def __init__(self, root, transform=None, labels: Optional[str]=None, config=None, conditioning=None, verbose: bool=False):
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
        
        if (labels is not None) and os.path.exists(labels):

            if labels.endswith(".csv"):
                if verbose:
                    print("Loading dataset from csv")
                self.load_annotations_from_csv()

            if labels.endswith(".pkl"):
                if verbose:
                    print("Loading dataset from pickle")
                self.load_annotations_from_pickle()

        if self.samples is None:
            if verbose:
                print("Search for image files in root and child folders. This might take a while...")
            self.search_images_in_folder()


    def load_annotations_from_csv(self):
        raise NotImplementedError("Method not implemented for this dataset.")


    def load_annotations_from_pickle(self):
        raise NotImplementedError("Method not implemented for this dataset.")


    def search_images_in_folder(self):
        raise NotImplementedError("Method not implemented for this dataset.")


    def __len__(self):
        return len(self.samples)


class HolographyImageFolder(BaseHolographyImageFolder):
    
    def __init__(self, root, transform=None, labels=None, config=None, conditioning=None, verbose=False):
        super().__init__(root, transform, labels, config, conditioning, verbose) 

    def search_images_in_folder(self):
        
        # Search the root dir if no valid labels file is given
        if self.labels is None or not os.path.exists(self.labels):
            search_root_dir = True
        else: 
            search_root_dir = False

        foldername_to_id = dict()
        index = 0
        if search_root_dir:
            # Get all images in root and subdirs of root
            self.samples = []
            for dirpath, _, filenames in os.walk(self.root):
                for filename in filenames:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(dirpath, filename)
                        relative_path = os.path.relpath(img_path, self.root)
                        folder_name = os.path.basename(os.path.dirname(img_path))
                        if folder_name not in foldername_to_id.keys():
                            foldername_to_id[folder_name] = index
                            index += 1
                        # self.samples.append((relative_path, filename, foldername_to_id[folder_name]))
                        self.samples.append((relative_path, filename))

            # Save the searched samples as pickle file
            if self.labels is not None and self.labels.lower().endswith((".pickle", ".pkl")):
                Path(os.path.dirname(os.path.split(self.labels)[-1])).mkdir(parents=True, exist_ok=True)
                with open(self.labels, 'wb') as f:
                    pickle.dump(self.samples, f)

            # Save the searched samples as csv file
            if self.labels is not None and self.labels.lower().endswith(".csv"):
                Path(os.path.dirname(os.path.split(self.labels)[-1])).mkdir(parents=True, exist_ok=True)
                pd.DataFrame(self.samples).to_csv(self.labels, index=False, header=["img_path", "filename"])
        

    def load_annotations_from_csv(self):
        df = pd.read_csv(self.labels)

        if self.config is not None:
            # Get abbreviation from config if they exists
            filename_column = self.config.get("filenames", "filename")
            image_folder = self.config.get("img_path", "img_path")
            class_cond_colunmn = self.config.get("classes", None) 
            feature_columns = self.config.get("features", None)
            cond_image_colunmn = self.config.get("cond_img_path", None)
        else:
            # Default names
            filename_column = "filename"
            image_folder = "img_path"
            class_cond_colunmn = None
            feature_columns = None
            cond_image_colunmn = None

        self.samples = list(zip(df[image_folder], df[filename_column]))

        # class features for conditioning
        if class_cond_colunmn is not None:
            self.class_labels = df[class_cond_colunmn].values.tolist()
        else:
            self.class_labels = None

        # tabular features for conditioning
        if feature_columns is not None:
            self.tabular_features = df[feature_columns].values.tolist()
        else:
            self.tabular_features = None
            
        # images for conditioning
        if cond_image_colunmn is not None:
            self.cond_imgs = df[cond_image_colunmn].values.tolist()
        else:
            self.cond_imgs = None
        

    def load_annotations_from_pickle(self):
        # load samples from pickle file if exists
        with open(self.labels, 'rb') as f:
            self.samples = pickle.load(f)


    def __getitem__(self, idx):

        # get image
        img_path, filename = self.samples[idx]
        img = Image.open(os.path.join(self.root, img_path))

        if img.mode == 'I':
            img = img.convert('I;16') 
            img = (np.array(img) / 256).astype('uint8')

        elif img.mode == 'L':
            img = np.array(img).astype('uint8')

        if self.transform:
            img = self.transform(img)

        # Conditioning
        condition = dict()

        if self.class_labels:
            condition["class"] = self.class_labels[idx]

        if self.tabular_features:
            tab_cond = self.tabular_features[idx]
            condition["tabular"] = torch.tensor(tab_cond)

        if self.cond_imgs: 
            condition["image"] = img

        return img, condition, filename


class PairwiseHolographyImageFolder(BaseHolographyImageFolder):
    
    def __init__(self, root, transform=None, labels=None, config=None, conditioning=None, verbose=False):
        super().__init__(root, transform, labels, config, conditioning, verbose) 


    def load_annotations_from_csv(self):
        df = pd.read_csv(self.labels)
        
        if self.config is not None:
            filename_column = self.config.get("filenames", "filename")
            image_folder = self.config.get("img_path", "img_path")
            class_cond_colunmn = self.config.get("classes", None) 
            feature_columns = self.config.get("features", None)
            cond_image_colunmn = self.config.get("cond_img_path", None)
            event_id_column = "event_id"
        else:
            # Default names
            filename_column = "filename"
            image_folder = "img_path"
            class_cond_colunmn = None
            feature_columns = None
            cond_image_colunmn = None
            event_id_column = "event_id"

        # Group by event_id
        grouped = df.groupby(event_id_column)
        self.samples = []
        self.class_labels = []
        self.tabular_features = []
        self.cond_imgs = []

        for event_id, group in grouped:
            if len(group) != 2:
                raise ValueError(f"Event ID {event_id} does not have exactly two rows.")
            
            filenames = group[filename_column].tolist()
            img_paths = [os.path.join(image_folder, filename) for filename in filenames]
            self.samples.append((img_paths, filenames))

            if class_cond_colunmn is not None:
                self.class_labels.append(group[class_cond_colunmn].tolist())
            if feature_columns is not None:
                self.tabular_features.append(group[feature_columns].values.tolist())
            if cond_image_colunmn is not None:
                self.cond_imgs.append(group[cond_image_colunmn].tolist())


    def __getitem__(self, idx):
        img_paths, filenames = self.samples[idx]
        imgs = [Image.open(img_path) for img_path in img_paths]

        for i, img in enumerate(imgs):
            if img.mode == 'I':
                img = img.convert('I;16') 
                img = (np.array(img) / 256).astype('uint8')
            elif img.mode == 'L':
                img = np.array(img).astype('uint8')

            if self.transform:
                imgs[i] = self.transform(img)   

        # Conditioning
        condition = dict()

        if self.class_labels:
            condition["class"] = self.class_labels[idx]

        if self.tabular_features:
            tab_cond = self.tabular_features[idx]
            condition["tabular"] = torch.tensor(tab_cond)

        if self.cond_imgs:
            condition["image"] = imgs

        return imgs, condition, filenames

