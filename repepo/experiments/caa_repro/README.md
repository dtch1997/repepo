# Contrastive Activation Addition

This directory contains scripts to reproduce the results in [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)

## Reproducing figures

Figure 6
```bash
# Run from project root
python repepo/experiments/caa_repro/generate_vectors.py --model_size "13b"
python repepo/experiments/caa_repro/prompting_with_steering.py --layers [15,16,17,18,19,20] --multipliers [-1.5,-1,-0.5,0,0.5,1,1.5] --settings.model_size "13b"
python repepo/experiments/caa_repro/plot_results.py --settings.type in_distribution --layers [15,20] --multipliers [-1.5,0,1.5] --settings.model_size "13b"
# plot saved in `repepo/experiments/caa_repro/analysis`
```

TODO: Figure 11
