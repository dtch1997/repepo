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

```bash
python repepo/baselines/sft/train_simple.py --num_train_epochs 30
# --dataset_name truthful_qa (truthful_qa, steroset)
# --model_name EleutherAI/Pythia70m (any HF model)
```
