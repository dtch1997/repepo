
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

The workflow to extract and evaluate concept vectors is in a bash script:
```bash
bash repepo/experiments_2/run.sh 
```

By default, we only use 6 out of 136 possible datasets. For debugging purposes, the datasets used by default are `train-dev` and `val-dev`, each comprising about 1% of the full dataset. 

To run on the full `train` and `val` split, we can use the bash script like this: 
```bash
bash repepo/experiments_2/run.sh dev train val
```

The intermediate artefacts can be inspected with the `inspect_xxx` notebooks. 
