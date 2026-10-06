# Córdoba dataset preparation

Create an events CSV with one row per reconstruction image and empty labels:

```powershell
python -m pollen_datasets.dataset_builder.cordoba.create_labels_csv `
    --root "Z:\marvel\marvel-fhnw\data\Cordoba\events" `
    --output "cordoba_filter_events.csv"
```

Image paths are relative to `--root`. The default split is `test`; override it
with `--split`. Only events with exactly two matching image entries, both
pointing to nonempty files, are retained. Image decoding is not checked.
Existing output files are never overwritten.

Add future Córdoba-specific preparation scripts to this package. When multiple
subset scripts are needed, group them in a `subsets` subpackage and select whole
events so both image rows stay together.
