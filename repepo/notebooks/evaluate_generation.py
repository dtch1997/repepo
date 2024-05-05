# %%
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
    evaluate,
    EvalHook,
    SteeringHook,
    Evaluator,
    MultipleChoiceAccuracyEvaluator,
    LogitDifferenceEvaluator,
    NormalizedPositiveProbabilityEvaluator,
    EvalResult,
    EvalPrediction,
)
from steering_vectors import SteeringVector, guess_and_enhance_layer_config
from repepo.data.multiple_choice.make_mwe_xrisk import make_mwe
from repepo.data.multiple_choice.make_mwe_persona import make_mwe_personas_caa
from repepo.experiments.persona_generalization import persona_pipeline, PersonaCrossSteeringExperimentResult
from transformers import AutoTokenizer, AutoModelForCausalLM
from repepo.steering.utils.helpers import make_dataset

def setup_logger(name: str, logging_level: int = logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger("evaluate_generation", logging.INFO)

# Function to apply appropriate hooks to pipeline. 
def get_eval_hooks(
    layer_id: int,
    multiplier: float = 0,
    completion_template: str | None = None,
    system_prompt: str | None = None
) -> list[EvalHook]:
    eval_hooks = [
        set_repe_direction_multiplier_at_eval(multiplier),
        select_repe_layer_at_eval(layer_id),
    ]
    if completion_template is not None:
        eval_hooks.append(
            update_completion_template_at_eval(completion_template)
        )
    if system_prompt is not None:
        eval_hooks.append(update_system_prompt_at_eval(system_prompt))
    return eval_hooks

# Function to get a single prediction
def get_prediction(
    pipeline: Pipeline,
    example: Example,
    evaluators: list[Evaluator] = [
        LogitDifferenceEvaluator()
    ],
    n_generation: int = 1,
    slim_results: bool = True
) -> EvalPrediction:
    logger.debug(f"Calculating output logprobs for example {example}")
    positive_probs = pipeline.calculate_output_logprobs(
        example.positive, slim_results=slim_results
    )
    negative_probs = pipeline.calculate_output_logprobs(
        example.negative, slim_results=slim_results
    )
    pred = EvalPrediction(
        positive_output_prob=positive_probs,
        negative_output_prob=negative_probs,
        metrics={},
    )
    example_metrics = {}
    logger.debug(f"Scoring prediction for example {example}")
    for evaluator in evaluators:
        example_metrics[evaluator.get_metric_name()] = (
            evaluator.score_prediction(pred)
        )
    logger.debug(f"Generating {n_generation} completions for example {example}")
    for _ in range(n_generation):
        prompt = pipeline.build_generation_prompt(example.positive)
        response = pipeline.generate(example.positive)
        pred.generations.append(Completion(prompt=prompt, response=response))
    pred.metrics = example_metrics
    return pred

def get_completion_template():
    return "{prompt} ({response}"

def preprocess_example(example: Example) -> Example:
    # strip the leading brackets
    pos_response = example.positive.response[1:]
    neg_response = example.negative.response[1:]
    
    return Example(
        positive = Completion(
            prompt = example.positive.prompt,
            response = pos_response
        ),
        negative = Completion(
            prompt = example.negative.prompt,
            response = neg_response
        ),
        meta = example.meta
    )

def get_steering_vector_result_df(
    pipeline: Pipeline,
    steering_vector: SteeringVector,
    dataset: Dataset,
    layers: list[int],
    multipliers: list[float],
    n_generation: int = 1    
) -> pd.DataFrame:



    rows = []
    for layer, multiplier in itertools.product(layers, multipliers):

        eval_hooks = get_eval_hooks(
            layer_id=layer,
            multiplier=multiplier,
            completion_template=get_completion_template(),
            system_prompt=None
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
                example = preprocess_example(example)
                eval_prediction = get_prediction(
                    pipeline,
                    example,
                    n_generation=n_generation   
                )
                logit_diff = eval_prediction.metrics["logit_diff"]
                predicted_token = example.positive.response[0] if logit_diff > 0 else example.negative.response[0]
                for generation in eval_prediction.generations:
                    row_dict = {
                        "prompt": generation.prompt,
                        "response": generation.response,
                        "layer": layer,
                        "multiplier": multiplier,                        
                        "logit_diff": logit_diff,
                        # Which token is predicted?
                        "predicted_token": predicted_token,
                    }
                    row_dict.update(eval_prediction.metrics)
                    rows.append(row_dict)
                
        pipeline.hooks.clear()
    df = pd.DataFrame(rows)
    return df

# Main
# %%
model_name: str = "meta-llama/Llama-2-7b-chat-hf"
dataset_name: str = "openness"
persona: str = "baseline"
device: str = "cuda"

make_mwe_personas_caa()
make_mwe()

# %%
torch.set_grad_enabled(False)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map=0
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)


# %%
dataset = make_dataset(dataset_name)
pipeline = persona_pipeline(model, tokenizer, persona, dataset_name, False)

# %%
# Load steering vector
result_path = pathlib.Path(Environ.ProjectDir) / "experiments" / "persona_generalization"
result = torch.load(result_path / f"{dataset_name}.pt")
steering_vector: SteeringVector = result.steering_vectors["baseline"]
print(steering_vector)

# %%

df = get_steering_vector_result_df(
    pipeline,
    steering_vector,
    dataset[:10],
    layers=[13],
    multipliers=[-1, 0, 1],
    n_generation=1
)
# %%
from IPython.display import HTML

pd.set_option("display.max_colwidth", None)
HTML(df.to_html(index=False))
# %%
import seaborn as sns
sns.set_theme()
df['prompt_idx'] = df['prompt'].astype('category').cat.codes.astype('category')
sns.lineplot(df, x="multiplier", y="logit_diff", hue="prompt_idx")

# %%
