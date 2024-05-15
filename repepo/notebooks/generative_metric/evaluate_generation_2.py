# flake8: noqa
# %%
import pandas as pd
import torch
import itertools
import pathlib
import logging
from IPython.display import HTML
from transformers.generation import GenerationConfig
from contextlib import ExitStack
from repepo.core.format import LlamaChatFormatter
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
from repepo.steering.utils.helpers import get_model_and_tokenizer
import pandas as pd 

def get_generation(
    pipeline: Pipeline,
    completion: Completion,
    steering_vector: SteeringVector,
    layer: int,
    multiplier: float,
    generation_config: GenerationConfig | None = None,
) -> Completion:
    eval_hooks = get_eval_hooks(
        layer_id=layer,
        multiplier=multiplier,
        # completion_template=get_completion_template(use_caa),
        completion_template=None,
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
        gen_prompt = pipeline.build_generation_prompt(completion)
        gen_response = pipeline.generate(completion, generation_config= generation_config)

    pipeline.hooks.clear()
    return Completion(prompt=gen_prompt, response=gen_response)

def get_logit_diff(
    pipeline: Pipeline,
    example: Example,
    steering_vector: SteeringVector,
    layer: int,
    multiplier: float,
) -> float:
    eval_hooks = get_eval_hooks(
        layer_id=layer,
        multiplier=multiplier,
        # completion_template=get_completion_template(use_caa),
        completion_template=None,
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
        eval_prediction = get_prediction(pipeline, example)

    pipeline.hooks.clear()
    return eval_prediction.metrics["logit_diff"]

# %%
# Load persona dataset

# dataset_name: str = "anti-LGBTQ-rights"
# raw_dataset_filepath = get_raw_dataset_dir() / "mwe" / "persona" / f"{dataset_name}.jsonl"
# raw_df = pd.read_json(raw_dataset_filepath, lines=True)
# raw_df.head()
# statements = raw_df['statement']
# print(statements[0])

# raw_df['question'][0]
# prompt = raw_df['question'][0].strip()
# answer_matching_behavior = raw_df['answer_matching_behavior'][0].strip()
# answer_not_matching_behavior = raw_df['answer_not_matching_behavior'][0].strip()
# caa_prompt = f"{prompt}\n\nChoices:\n(A) {answer_matching_behavior}\n(B) {answer_not_matching_behavior}"
# print(caa_prompt)

# prompt = raw_df['question'][0].strip()
# answer_matching_behavior = raw_df['answer_matching_behavior'][0].strip()
# answer_not_matching_behavior = raw_df['answer_not_matching_behavior'][0].strip()
# caa_prompt = f"{prompt}\n\nChoices:\n(A) {answer_matching_behavior}\n(B) {answer_not_matching_behavior}"
# print(caa_prompt)

# def make_caa_prompt_from_row(row):
#     prompt = row['question'].strip()
#     answer_matching_behavior = row['answer_matching_behavior'].strip()
#     answer_not_matching_behavior = row['answer_not_matching_behavior'].strip()
#     return f"{prompt}\n\nChoices:\n(A) {answer_matching_behavior}\n(B) {answer_not_matching_behavior}"

# raw_df['caa_prompt'] = raw_df.apply(make_caa_prompt_from_row, axis=1)
# raw_df.head()

# %% 
# Load xrisk dataset 

# dataset_name: str = "survival-instinct"
example_idx = 0
dataset_name: str = "corrigible-neutral-HHH"
raw_dataset_filepath = get_raw_dataset_dir() / "mwe" / "xrisk" / f"{dataset_name}.jsonl"
raw_df = pd.read_json(raw_dataset_filepath, lines=True)
raw_df.head()

def strip_choices(row):
    return row['question'].split("Choices")[0]

raw_df['statement'] = raw_df.apply(strip_choices, axis=1)
raw_df['caa_prompt'] = raw_df['question']
print(raw_df['caa_prompt'][example_idx])
print("Positive answer: ", raw_df['answer_matching_behavior'][example_idx])
print(raw_df['statement'][example_idx])
# %%
# Load pipeline
from repepo.paper.utils import get_persona_cross_steering_experiment_result_path, get_steering_vector, load_persona_cross_steering_experiment_result

torch.set_grad_enabled(False)

model_name: str = "meta-llama/Llama-2-7b-chat-hf"
persona: str = "baseline"
device: str = "cuda"

# Load the pipeline
model, tokenizer = get_model_and_tokenizer(model_name, load_in_8bit=True)
formatter = LlamaChatFormatter()
pipeline = Pipeline(model=model, tokenizer=tokenizer, formatter=formatter)

# %%
# Load steering vector
result = load_persona_cross_steering_experiment_result(dataset_name)
steering_vector = get_steering_vector(result, persona)
print(steering_vector)


# %%
# Calculate logit diff



# %%
# Calculate logit diff
caa_prompt = raw_df['caa_prompt'][example_idx]
answer_matching_behavior = raw_df['answer_matching_behavior'][example_idx]
answer_not_matching_behavior = raw_df['answer_not_matching_behavior'][example_idx]
caa_example = Example(
    positive=Completion(prompt=caa_prompt, response=answer_matching_behavior),
    negative=Completion(prompt=caa_prompt, response=answer_not_matching_behavior),
)
# multipliers = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
# multipliers = list(range(-10, 11))
multipliers = [-100, 0, 100]
for multiplier in multipliers:
    logit_diff = get_logit_diff(pipeline, caa_example, steering_vector, 13, multiplier)
    print(f"Multiplier: {multiplier}, Logit Diff: {logit_diff}")
# %%
# Generate completions
pd.set_option('display.max_colwidth', None)
gen_config = GenerationConfig(
    max_length=500,
)

rows = []

statement = raw_df['statement'][example_idx]
completion = Completion(
    prompt=statement,
    response=""
)

layer = 13
# multipliers = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
# multipliers = [-5, -3, 0]

multipliers = [-100, 0, 100]
for multiplier in multipliers:
    gen_completion = get_generation(
        pipeline, 
        completion, 
        steering_vector, 
        layer, 
        multiplier,
        generation_config=gen_config
    )
    rows.append({
        "multiplier": multiplier,
        "gen_prompt": gen_completion.prompt,
        "gen_response": gen_completion.response
    })

df = pd.DataFrame(rows)

HTML(df.to_html(index=False))
# %%
# In the case of CAA, evaluate the total probability mass on option tokens. 

A_token = 29909
B_token = 29933


for caa_prompt in raw_df['caa_prompt'][:5]:
    prompt = pipeline.build_generation_prompt(Completion(prompt=caa_prompt, response="")) + " ("
    print(prompt)
    outputs = pipeline(prompt)
    logits = outputs.loss['logits']
    print(f"A token: {A_token}, B token: {B_token}")


    A_logit = logits[0, -1, A_token].item()
    B_logit = logits[0, -1, B_token].item()
    last_token_logits = logits[0, -1, :]
    topk = torch.topk(last_token_logits, 5)
    print(topk)
    print(tokenizer.decode(topk.indices))
    print(f"A logit: {A_logit}, B logit: {B_logit}")
# %%
print(topk.indices)
print(tokenizer.decode(topk.indices[0]))
print(tokenizer.decode(topk.indices[1]))
print(tokenizer.decode(A_token))
print(tokenizer.decode(B_token))
# %%

print(tokenizer.encode("My A A"))
# %%
