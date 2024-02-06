
Workflow for generating datasets:

## Download MWE Datasets

Generate the list of filenames to download
1. Clone the Model-Written Evals github repo and cd into it
2. Run this command: 
```
ls -a persona | grep jsonl | sed 's/^/persona\//' > anthropic_dataset_names.txt
```
3. Move the file back to this directory

Download files in the list of filenames and preprocess them to our own dataset format
```
python download_datasets.py
```

Inspect the datasets using `inspect_datasets.ipynb`

## Extract Concept Vectors

Extract concept vectors using: 
```
python extract_concept_vectors.py
```

Inspect concept vectors using `inspect_concept_vectors.ipynb` 
