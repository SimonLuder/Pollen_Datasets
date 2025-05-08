# Pollen Datasets

This reporitory contains code to access data from the SwissPoleno project. 

*Note: This only works if you have the correct access rights. If you don't know what this dataset is, you probably don't have access rights.*


**Installation:**
```
pip install git+https://github.com/SimonLuder/Pollen_Datasets.git
```

**How to use:**
```python
from pollen_datasets.poleno import PairwiseHolographyImageFolder

dataset = PairwiseHolographyImageFolder(
    root="YourImagesRootFolder",
    transform=transform, 
    labels="YourLabelsFilePath.csv"
)
```
