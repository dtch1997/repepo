
# Running Baselines

The following code assumes you have set up dependencies. 

For detailed instructions, refer to README.md in top-level directory. 


## Generate datasets

```bash
cd repepo/data
make datasets
```

Currently, we have implemented `stereoset` and `truthful_qa`. 

The datasets are stored in `datasets` in top-level directory. 

## Supervised Fine-tuning

Our implementation is based off Stanford Alpaca. 

```bash
python repepo/baselines/sft/train.py --num_train_epochs 30
# --dataset_name truthful_qa (truthful_qa, steroset)
```

Watch the loss go down in WandB! 