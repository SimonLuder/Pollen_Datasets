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
    labels="YourLabelsFile.csv",
    dataset_cfg=dataset_cfg,
    transform=transform, 
) 
# Returns: 
#   (image, condition, filename)
```

Pairwise images
```python
from pollen_datasets.poleno import HolographyImageFolder

dataset = PairwiseHolographyImageFolder(
    root="YourImagesRootFolder",
    labels="YourLabelsFile.csv",
    dataset_cfg=dataset_cfg,
    transform=transform,
    pair_transform=pair_transform
) 
# Returns: 
#   ((image1, image2), (condition1, condition2), (filename1, filename2))
```

Stacked pairs of images
```python
from pollen_datasets.poleno import HolographyImageFolder

dataset = StackedPairwiseHolographyImageFolder(
    root="YourImagesRootFolder",
    labels="YourLabelsFile.csv",
    dataset_cfg=dataset_cfg,
    transform=transform,
    pair_transform=pair_transform
    merge_conditions=False,
) 
# Returns:
#   (stacked_images, (condition1, condition2), (filename1, filename2))  if merge_conditions=False
#   (stacked_images, stacked_condition, (filename1, filename2))         if merge_conditions=True

```