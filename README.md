# Pollen Datasets

This repository contains code to access data from the SwissPoleno project. 

*Note: This only works if you have the correct access rights. If you don't know what this dataset is, you probably don't have access rights.*


**Installation:**
```
pip install git+https://github.com/SimonLuder/Pollen_Datasets.git
```

**How to use:**

Single images
```python
from pollen_datasets.poleno import HolographyImageFolder

dataset = HolographyImageFolder(
    root="YourImagesRootFolder",
    labels="YourLabelsFilePath.csv",
    dataset_cfg=dataset_cfg,
    transform=transform, 
)
```

Pairwise images
```python
from pollen_datasets.poleno import HolographyImageFolder

dataset = PairwiseHolographyImageFolder(
    root="YourImagesRootFolder",
    labels="YourLabelsFilePath.csv",
    dataset_cfg=dataset_cfg,
    transform=transform,
    pair_transform=pair_transform
)
```