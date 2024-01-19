This directory contains scripts to reproduce the results in [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)

To produce figures with Llama-2-hf-chat-7b.

```bash
# Run from project root
python repepo/experiments/caa_repro/generate_vectors.py --dataset_spec.name sycophancy --dataset_spec.split 0:1%
python repepo/experiments/caa_repro/prompting_with_steering.py --settings.type in_distribution --layers [15] --multipliers [-1 0 1] --settings.model_size 7b --settings.dataset_spec.split 1:2%
python repepo/experiments/caa_repro/plot_results.py --settings.type in_distribution --layers [15,20] --multipliers [-1.5,0,1.5] --settings.model_size 7b --settings.dataset_spec.split 1:2%
```

Results will be saved in `experiments/caa_repro/analysis`
