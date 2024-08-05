
# Llama70b experiments

```bash
# From the root directory
pip install -e . 
mkdir experiments
python -m repepo.experiments.sweep_llama3_70b_layers --output_dir experiments
# Then zip 'experiments' dir and send it
```