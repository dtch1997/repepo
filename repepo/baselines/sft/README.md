# Supervised Fine-Tuning

## Quickstart

First, generate the dataset by running `truthful_qa_ift.ipynb`. 

This should result in a local file `truthful_qa.json`, which looks like this:
```
[
    {
        'instruction': <question>
        'input': ''
        'output': <answer>
    },
    ...
]
```

This can be used to train a model as follows:
```bash
python alpaca_train.py \
    --model_name_or_path EleutherAI/pythia-70m \ 
    --data_path ./truthful_qa.json \
    --cache_dir . \ # Replace with your cache dir
    --output_dir ./output \
    --num_train_epochs 30
```