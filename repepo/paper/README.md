
# Usage

## Data

We expect that appropriate data is located in the ProjectDir at `experiments`:
- `experiments/persona_generalization_llama7b` for llama7b
- `experiments/persona_generalization_qwen` for Qwen

These should be directories which contain many `PersonaCrossSteeringExperimentResult`s from doing a sweep over datasets

## Run the scripts

```bash
python repepo/paper/preprocess_results.py --model qwen
python repepo/paper/preprocess_results.py --model llama7b
python repepo/paper/make_figures_steering_id.py
python repepo/paper/make_figures_steering_ood.py
python repepo/paper/make_figures_misc.py
```
