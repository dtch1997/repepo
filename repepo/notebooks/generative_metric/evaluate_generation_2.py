# flake8: noqa
# %%

import os

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

import pandas as pd
import torch
import itertools
import pathlib
import logging
from contextlib import ExitStack
from repepo.variables import Environ
from repepo.core.pipeline import Pipeline
from repepo.core.types import Example, Completion, Dataset
from repepo.core.evaluate import (
    update_completion_template_at_eval,
    update_system_prompt_at_eval,
    select_repe_layer_at_eval,
    set_repe_direction_multiplier_at_eval,
    get_eval_hooks,
    get_prediction,
    EvalHook,
    SteeringHook,
    Evaluator,
    LogitDifferenceEvaluator,
    EvalPrediction,
)
from steering_vectors import SteeringVector, guess_and_enhance_layer_config
from repepo.data.multiple_choice.make_mwe_xrisk import make_mwe
from repepo.data.multiple_choice.make_mwe_persona import make_mwe_personas_caa
from repepo.experiments.persona_generalization import persona_pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from repepo.steering.utils.helpers import make_dataset
from repepo.experiments.persona_generalization import (
    PersonaCrossSteeringExperimentResult,
)

import json
from typing import cast

from repepo.data.make_dataset import (
    build_dataset_filename,
    get_dataset_dir,
    get_raw_dataset_dir,
)
from repepo.data.io import jdump
from repepo.core.types import Dataset, Example, Completion
from repepo.translation.constants import LangOrStyleCode
from repepo.translation.load_translation import load_translation, TS
from repepo.data.multiple_choice.make_mwe_persona import preprocess_ab_dataset


def get_completion_template(
    use_caa: bool = False,
):
    if use_caa:
        return "{prompt} ({response}"
    else:
        return "{prompt} {response}"


def preprocess_example(
    example: Example,
    use_caa: bool = False,
) -> Example:
    if not use_caa:
        return example

    # strip the leading brackets
    pos_response = example.positive.response[1:]
    neg_response = example.negative.response[1:]

    return Example(
        positive=Completion(prompt=example.positive.prompt, response=pos_response),
        negative=Completion(prompt=example.negative.prompt, response=neg_response),
        meta=example.meta,
    )


def get_steering_vector_result_df(
    pipeline: Pipeline,
    steering_vector: SteeringVector,
    dataset: Dataset,
    layers: list[int],
    multipliers: list[float],
    n_generation: int = 1,
    use_caa: bool = False,
) -> pd.DataFrame:
    rows = []
    for layer, multiplier in itertools.product(layers, multipliers):
        eval_hooks = get_eval_hooks(
            layer_id=layer,
            multiplier=multiplier,
            completion_template=get_completion_template(use_caa),
            system_prompt=None,
        )

        steering_hook = SteeringHook(
            steering_vector=steering_vector,
            direction_multiplier=0,
            patch_generation_tokens_only=True,
            skip_first_n_generation_tokens=0,
            layer_config=guess_and_enhance_layer_config(pipeline.model),
        )

        pipeline.hooks.clear()
        pipeline.hooks.append(steering_hook)

        with ExitStack() as stack:
            for eval_hook in eval_hooks:
                stack.enter_context(eval_hook(pipeline))

            for i, example in enumerate(dataset):
                example = preprocess_example(example, use_caa)
                eval_prediction = get_prediction(
                    pipeline, example, n_generation=n_generation
                )
                logit_diff = eval_prediction.metrics["logit_diff"]
                predicted_token = (
                    example.positive.response[0]
                    if logit_diff > 0
                    else example.negative.response[0]
                )
                for generation in eval_prediction.generations:
                    row_dict = {
                        "prompt": generation.prompt,
                        "response": generation.response,
                        "layer": layer,
                        "multiplier": multiplier,
                        "logit_diff": logit_diff,
                        "predicted_token": predicted_token,
                        # metadata
                        "prompt_idx": i,
                        "raw_prompt": example.positive.prompt,
                    }
                    row_dict.update(eval_prediction.metrics)
                    rows.append(row_dict)

        pipeline.hooks.clear()
    df = pd.DataFrame(rows)
    return df


# %%

torch.set_grad_enabled(False)

model_name: str = "meta-llama/Llama-2-7b-chat-hf"
dataset_name: str = "openness"
persona: str = "baseline"
device: str = "cuda"

# Load the pipeline
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map=0
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = persona_pipeline(model, tokenizer, persona, dataset_name, False)

# Load steering vector
result_path = (
    pathlib.Path(Environ.ProjectDir) / "experiments" / "persona_generalization"
)
result = torch.load(result_path / f"{dataset_name}.pt")
steering_vector: SteeringVector = result.steering_vectors["baseline"]
print(steering_vector)

# %%

# Load a dataset with and without prompts
# And then visualize

import pandas as pd

pd.set_option("display.max_colwidth", None)


def get_pos_completions(dataset: Dataset) -> list[Completion]:
    completions = []
    for example in dataset:
        completions.append(example.positive)
    return completions


def make_completions_df(completions: list[Completion]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "prompt": [completion.prompt for completion in completions],
            "response": [completion.response for completion in completions],
        }
    )
    return df


dataset_gen = make_dataset(dataset_name + "_gen")

gen_df = make_completions_df(get_pos_completions(dataset_gen))
gen_df.head()


# %%
dataset_caa = make_dataset(dataset_name)
caa_df = make_completions_df(get_pos_completions(dataset_caa))
caa_df.head()
# %%

# Evaluate pipeline and steering vector on CAA-formatted dataset --> get logit diff
gen_result_df = get_steering_vector_result_df(
    pipeline,
    steering_vector,
    dataset_gen[:1],
    layers=[13],
    multipliers=[-1, 0, 1],
    n_generation=1,
    use_caa=False,
)
print(len(gen_result_df))
gen_result_df = gen_result_df[
    ["prompt", "response", "prompt_idx", "layer", "multiplier"]
]
gen_result_df.head()
# %%
# Evaluate pipeline and steering vector on Gen-formatted dataset --> get generations
# Evaluate pipeline and steering vector on CAA-formatted dataset --> get logit diff
caa_result_df = get_steering_vector_result_df(
    pipeline,
    steering_vector,
    dataset_caa[:1],
    layers=[13],
    multipliers=[-1, 0, 1],
    n_generation=1,
    use_caa=True,
)
print(len(caa_result_df))
caa_result_df = caa_result_df[
    ["prompt", "prompt_idx", "logit_diff", "layer", "multiplier"]
]
caa_result_df.head()
# %%

df = pd.merge(
    gen_result_df,
    caa_result_df,
    on=["prompt_idx", "layer", "multiplier"],
    suffixes=("_gen", "_caa"),
)
df.head()
# %%

# Plot logits
import seaborn as sns  # noqa

sns.set_theme()
df = caa_result_df.copy()
df["prompt_idx"] = df["prompt"].astype("category").cat.codes.astype("category")
sns.lineplot(df, x="multiplier", y="logit_diff", hue="prompt_idx")

# %%


def get_combined_results_df(
    pipeline,
    steering_vector,
    dataset_name: str,
    layers: list[int],
    multipliers: list[float],
    n_generation: int = 1,
    n_examples: int = 1,
    use_caa: bool = False,
):
    dataset_gen = make_dataset(dataset_name + "_gen")
    dataset_caa = make_dataset(dataset_name)

    caa_result_df = get_steering_vector_result_df(
        pipeline,
        steering_vector,
        dataset_caa[:n_examples],
        layers=[13],
        multipliers=[-1, 0, 1],
        n_generation=1,
        use_caa=True,
    )
    caa_result_df = caa_result_df[
        ["prompt", "raw_prompt", "prompt_idx", "logit_diff", "layer", "multiplier"]
    ]

    gen_result_df = get_steering_vector_result_df(
        pipeline,
        steering_vector,
        dataset_gen[:n_examples],
        layers=[13],
        multipliers=[-1, 0, 1],
        n_generation=1,
        use_caa=False,
    )
    gen_result_df = gen_result_df[
        ["prompt", "raw_prompt", "response", "prompt_idx", "layer", "multiplier"]
    ]
    df = pd.merge(
        gen_result_df,
        caa_result_df,
        on=["prompt_idx", "layer", "multiplier"],
        suffixes=("_gen", "_caa"),
    )

    return df


# %%
df = get_combined_results_df(
    pipeline,
    steering_vector,
    dataset_name,
    layers=[13],
    multipliers=[-1, 0, 1],
    n_generation=1,
    n_examples=10,
    use_caa=False,
)
# %%
sns.set_theme()
sns.lineplot(df, x="multiplier", y="logit_diff", hue="prompt_idx")
# %%
from IPython.display import HTML  # noqa

pd.set_option("display.max_colwidth", None)
HTML(df.to_html(index=False))
# %%
df.to_csv("openness-steering.csv")
# %%
