from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
from statistics import mean
from typing import Literal, cast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import seaborn as sns
from simple_parsing import ArgumentParser
import matplotlib.pyplot as plt
from steering_vectors import SteeringVector
from repepo.core.format import (
    Llama3ChatFormatter,
    LlamaChatFormatter,
    QwenChatFormatter,
)
from repepo.core.pipeline import Pipeline
from repepo.core.types import Example, Model, Tokenizer
from repepo.steering.build_steering_training_data import (
    build_steering_vector_training_data,
)
from repepo.steering.utils.helpers import get_model_and_tokenizer
from repepo.steering.evaluate_cross_steering import (
    CrossSteeringResult,
    evaluate_cross_steering,
)
from repepo.steering.plot_steering_vector_cos_similarities import (
    plot_steering_vector_cos_similarities,
)
from repepo.steering.utils.helpers import make_dataset
from steering_vectors import train_steering_vector
from repepo.data.multiple_choice.make_mwe_xrisk import make_mwe as make_mwe_xrisk_caa
from repepo.data.multiple_choice.make_mwe_persona import make_mwe_personas_caa
from repepo.data.multiple_choice.make_caa_sycophancy import make_sycophancy_caa
from repepo.data.multiple_choice.make_caa_truthfulqa import make_truthfulqa_caa
from repepo.utils.stats import bernoulli_js_dist
from repepo.experiments.get_datasets import get_all_prompts


Persona = Literal["positive", "negative", "baseline"]
PERSONAS: list[Persona] = ["positive", "negative", "baseline"]

CONFIG_SAVE_PATH = "config.json"

DARK_TRIAD_PERSONAS = {
    "machiavellianism",
    "narcissism",
    "psychopathy",
}
PERSONALITY_PERSONAS = {
    "neuroticism",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness",
}
POLITICS_PERSONAS = {
    "politically-liberal",
    "politically-conservative",
    "believes-in-gun-rights",
    "believes-abortion-should-be-illegal",
    "anti-immigration",
    "anti-LGBTQ-rights",
}


def persona_pipeline(
    model: Model,
    tokenizer: Tokenizer,
    persona: Persona,
    formatter_name: Literal[
        "llama-chat-formatter", "qwen-chat-formatter", "llama3-chat-formatter"
    ],
    dataset_name: str,
    use_sys_prompt: bool,
) -> Pipeline:
    all_persona_prompts = get_all_prompts()
    if dataset_name not in all_persona_prompts:
        print(all_persona_prompts.keys())
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    positive_prompt, negative_prompt = all_persona_prompts[dataset_name]
    if formatter_name == "llama-chat-formatter":
        formatter_class = LlamaChatFormatter
    elif formatter_name == "qwen-chat-formatter":
        formatter_class = QwenChatFormatter
    elif formatter_name == "llama3-chat-formatter":
        formatter_class = Llama3ChatFormatter
    else:
        raise ValueError(f"Invalid formatter name: {formatter_name}")
    formatter = formatter_class()
    if persona == "positive":
        if use_sys_prompt:
            formatter = formatter_class(system_prompt=positive_prompt)
        else:
            formatter = formatter_class(prompt_prefix=positive_prompt + "\n\n")
    elif persona == "negative":
        if use_sys_prompt:
            formatter = formatter_class(system_prompt=negative_prompt)
        else:
            formatter = formatter_class(prompt_prefix=negative_prompt + "\n\n")
    return Pipeline(model, tokenizer, formatter=formatter)


def prefix_example(example: Example, prefix: str) -> Example:
    return Example(
        positive=replace(
            example.positive,
            prompt=f"{prefix}\n\n{example.positive.prompt}",
        ),
        negative=replace(
            example.negative,
            prompt=f"{prefix}\n\n{example.negative.prompt}",
        ),
    )


def train_steering_vector_for_persona_and_dataset(
    model: Model,
    tokenizer: Tokenizer,
    persona: Persona,
    dataset_name: str,
    formatter_name: Literal[
        "llama-chat-formatter", "qwen-chat-formatter", "llama3-chat-formatter"
    ],
    train_split: str,
    layer: int,
    use_sys_prompt: bool = True,
    show_progress: bool = True,
) -> SteeringVector:
    train_dataset = make_dataset(dataset_name, train_split)
    pipeline = persona_pipeline(
        model,
        tokenizer,
        persona,
        formatter_name=formatter_name,
        dataset_name=dataset_name,
        use_sys_prompt=use_sys_prompt,
    )
    steering_vector_training_data = build_steering_vector_training_data(
        pipeline, train_dataset
    )
    return train_steering_vector(
        model,
        tokenizer,
        steering_vector_training_data,
        layers=[layer],
        show_progress=show_progress,
    )


@dataclass
class PersonaCrossSteeringExperimentResult:
    dataset_name: str
    steering_vectors: dict[str, SteeringVector]
    cross_steering_result: CrossSteeringResult


def run_persona_cross_steering_experiment(
    model: Model,
    tokenizer: Tokenizer,
    dataset_name: str,
    formatter_name: Literal[
        "llama-chat-formatter", "qwen-chat-formatter", "llama3-chat-formatter"
    ],
    train_split: str,
    test_split: str,
    layer: int,
    multipliers: list[float],
    normalize_steering_magnitude_to_baseline: bool = True,
    show_progress: bool = True,
    patch_generation_tokens_only: bool = True,
    patch_operator: Literal["add", "ablate_then_add"] = "add",
    skip_first_n_generation_tokens: int = 0,
    completion_template: str | None = None,
) -> PersonaCrossSteeringExperimentResult:
    steering_vectors: dict[str, SteeringVector] = {}
    test_ds = make_dataset(dataset_name, test_split)
    steering_vectors["baseline"] = train_steering_vector_for_persona_and_dataset(
        model,
        tokenizer,
        "baseline",
        formatter_name=formatter_name,
        dataset_name=dataset_name,
        train_split=train_split,
        layer=layer,
        show_progress=show_progress,
    )
    non_baseline_personas: list[Persona] = [
        persona for persona in PERSONAS if persona != "baseline"
    ]
    test_dataset_mapping = {
        "baseline": test_ds,
    }
    # for persona in non_baseline_personas:
    #     steering_vectors["SYS_" + persona] = (
    #         train_steering_vector_for_persona_and_dataset(
    #             model,
    #             tokenizer,
    #             persona,
    #             dataset_name=dataset_name,
    #             train_split=train_split,
    #             formatter_name=formatter_name,
    #             layer=layer,
    #             use_sys_prompt=True,
    #             show_progress=show_progress,
    #         )
    #     )
    #     test_dataset_mapping["SYS_" + persona] = test_ds
    #     steering_vectors["PT_" + persona] = (
    #         train_steering_vector_for_persona_and_dataset(
    #             model,
    #             tokenizer,
    #             persona,
    #             formatter_name=formatter_name,
    #             dataset_name=dataset_name,
    #             train_split=train_split,
    #             layer=layer,
    #             use_sys_prompt=False,
    #             show_progress=show_progress,
    #         )
    #     )
    #     test_dataset_mapping["PT_" + persona] = test_ds
    mean_layer_activation = torch.stack(
        [sv.layer_activations[layer] for sv in steering_vectors.values()]
    ).mean(dim=0)
    steering_vectors["mean"] = SteeringVector(
        layer_activations={layer: mean_layer_activation},
        layer_type=steering_vectors["baseline"].layer_type,
    )
    if normalize_steering_magnitude_to_baseline:
        base_magnitude = steering_vectors["baseline"].layer_activations[layer].norm()
        for steering_vector in steering_vectors.values():
            vec = steering_vector.layer_activations[layer]
            steering_vector.layer_activations[layer] = vec * base_magnitude / vec.norm()

    def _eval_build_pipeline(
        model: Model,
        tokenizer: Tokenizer,
        persona_label: str,
    ) -> Pipeline:
        # just use base pipeline for baseline
        if persona_label == "baseline":
            return persona_pipeline(
                model,
                tokenizer,
                "baseline",
                formatter_name=formatter_name,
                dataset_name=dataset_name,
                use_sys_prompt=False,
            )
        # handle the SYS_ stuff
        persona = persona_label.split("_")[1]
        return persona_pipeline(
            model,
            tokenizer,
            cast(Persona, persona),
            dataset_name=dataset_name,
            formatter_name=formatter_name,
            use_sys_prompt="SYS_" in persona_label,
        )

    cross_steering_result = evaluate_cross_steering(
        model,
        tokenizer,
        layer,
        steering_vectors,
        build_pipeline=_eval_build_pipeline,
        datasets=test_dataset_mapping,
        show_progress=show_progress,
        patch_generation_tokens_only=patch_generation_tokens_only,
        patch_operator=patch_operator,
        skip_first_n_generation_tokens=skip_first_n_generation_tokens,
        completion_template=completion_template,
        multipliers=multipliers,
    )
    return PersonaCrossSteeringExperimentResult(
        dataset_name=dataset_name,
        steering_vectors=steering_vectors,
        cross_steering_result=cross_steering_result,
    )


def plot_steering_on_dataset(
    result: PersonaCrossSteeringExperimentResult,
    dataset_version: str,
    save_path: str | None = None,
    metric_name: str = "mean_pos_prob",
):
    cs = result.cross_steering_result
    ds_index = cs.dataset_labels.index(dataset_version)

    multipliers = [*list(cs.neg_steering.keys()), 0.0, *list(cs.pos_steering.keys())]

    results_line_mean = []
    for i, label in enumerate(cs.steering_labels):
        results_line_mean.append(
            [
                *[
                    res[ds_index][i].metrics[metric_name]
                    for res in cs.neg_steering.values()
                ],
                cs.dataset_baselines[ds_index].metrics[metric_name],
                *[
                    res[ds_index][i].metrics[metric_name]
                    for res in cs.pos_steering.values()
                ],
            ]
        )

    plt.figure(figsize=(8, 6))
    # Plot each line
    line_styles = ["--", "-.", ":"]
    for i, (label, data) in enumerate(
        zip(
            cs.steering_labels,
            results_line_mean,
        )
    ):
        sns.lineplot(
            x=multipliers,
            y=data,
            label=label,
            linestyle=line_styles[i % len(line_styles)],
        )

    # Adding title and labels
    plt.title(f"{result.dataset_name}: {dataset_version}")
    plt.xlabel("Multiplier")
    plt.ylabel("Avg answer probability")

    # # Set the y-axis limits
    # plt.ylim(0, 1)

    # Add grid lines
    plt.grid(True)

    # Show the legend
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()


def extract_layer(result: PersonaCrossSteeringExperimentResult) -> int:
    return list(list(result.steering_vectors.values())[0].layer_activations.keys())[0]


def plot_persona_steering_vector_cos_similarities(
    result: PersonaCrossSteeringExperimentResult,
    save_path: str | None = None,
):
    named_steering_vectors = {
        shorten(label): sv for label, sv in result.steering_vectors.items()
    }
    layer = extract_layer(result)
    return plot_steering_vector_cos_similarities(
        named_steering_vectors,
        layer=layer,
        title=result.dataset_name + " (Cosine Similarity)",
        save_path=save_path,
    )


def shorten(label: str) -> str:
    return (
        label.replace("SYS_", "S_")
        .replace("PT_", "P_")
        .replace("ative", "")
        .replace("itive", "")
        .replace("line", "")
    )


def base_dataset_position(
    result: PersonaCrossSteeringExperimentResult,
    dist_metric: Literal["raw", "js"] = "raw",
    metric_name: str = "mean_pos_prob",
) -> float:
    """
    Return a value between 0 and 1
    1 means the baseline is close to the positive scores
    0 means the baseline is close to the negative scores
    0.5 means it's equidistant
    """
    cs = result.cross_steering_result
    base_index = cs.dataset_labels.index("baseline")
    pos_indices = [i for i, label in enumerate(cs.dataset_labels) if "pos" in label]
    neg_indices = [i for i, label in enumerate(cs.dataset_labels) if "neg" in label]

    base_prob = cs.dataset_baselines[base_index].metrics[metric_name]
    pos_prob = mean([cs.dataset_baselines[i].metrics[metric_name] for i in pos_indices])
    neg_prob = mean([cs.dataset_baselines[i].metrics[metric_name] for i in neg_indices])

    if base_prob > pos_prob:
        return 1.0
    if base_prob < neg_prob:
        return 0.0
    if dist_metric == "raw":
        neg_dist = abs(neg_prob - base_prob)
        pos_dist = abs(pos_prob - base_prob)
    elif dist_metric == "js":
        neg_dist = abs(bernoulli_js_dist(base_prob, neg_prob))
        pos_dist = abs(bernoulli_js_dist(base_prob, pos_prob))
    else:
        raise ValueError(f"Invalid dist_metric: {dist_metric}")
    return neg_dist / (pos_dist + neg_dist)


@dataclass
class PersonaGeneralizationExperimentConfig:
    output_dir: str
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    formatter_name: Literal[
        "llama-chat-formatter", "qwen-chat-formatter", "llama3-chat-formatter"
    ] = "llama-chat-formatter"
    patch_generation_tokens_only: bool = True
    patch_operator: Literal["add", "ablate_then_add"] = "add"
    skip_first_n_generation_tokens: int = 0
    completion_template: str | None = None
    normalize_steering_magnitude_to_baseline: bool = True
    train_split: str = "0:50%"
    test_split: str = "50:100%"
    layer: int = 13
    multipliers: list[float] = field(
        default_factory=lambda: [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]
    )
    load_in_8bit: bool = False


def make_all_datasets():
    make_sycophancy_caa()
    make_truthfulqa_caa()
    make_mwe_xrisk_caa()
    make_mwe_personas_caa()


def run_persona_generalization_experiment(
    config: PersonaGeneralizationExperimentConfig,
    sge_task_id: int | None = None,
) -> None:
    print(f"Running persona generalization experiment with config: {config}")
    make_all_datasets()

    if config.load_in_8bit:
        model, tokenizer = get_model_and_tokenizer(config.model_name, load_in_8bit=True)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    model.eval()

    output_dir = Path(config.output_dir)
    if output_dir.exists():
        if not os.path.exists(output_dir / CONFIG_SAVE_PATH):
            raise ValueError(
                f"Output directory {output_dir} exists but does not contain a config file."
            )
        with open(output_dir / CONFIG_SAVE_PATH, "r") as f:
            old_config_dict = json.load(f)
        old_config = PersonaGeneralizationExperimentConfig(**old_config_dict)
        if old_config != config:
            raise ValueError(
                f"Output directory {output_dir} exists but contains a different config."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / CONFIG_SAVE_PATH, "w") as f:
            json.dump(config.__dict__, f, indent=4, ensure_ascii=False)

    all_persona_prompts = get_all_prompts()

    if sge_task_id is not None:
        # Select a single dataset
        persona_prompt = list(all_persona_prompts.keys())[sge_task_id]
        all_persona_prompts = {persona_prompt: all_persona_prompts[persona_prompt]}

    for i, dataset_name in enumerate(all_persona_prompts):
        results_save_file = output_dir / f"{dataset_name}.pt"
        if results_save_file.exists():
            print(f"already ran {dataset_name}, skipping")
            continue
        print(
            f"Running experiment for dataset {dataset_name} ({i+1} / {len(all_persona_prompts)})"
        )
        result = run_persona_cross_steering_experiment(
            model,
            tokenizer,
            train_split=config.train_split,
            test_split=config.test_split,
            dataset_name=dataset_name,
            formatter_name=config.formatter_name,
            layer=config.layer,
            normalize_steering_magnitude_to_baseline=config.normalize_steering_magnitude_to_baseline,
            patch_generation_tokens_only=config.patch_generation_tokens_only,
            patch_operator=config.patch_operator,
            skip_first_n_generation_tokens=config.skip_first_n_generation_tokens,
            completion_template=config.completion_template,
            multipliers=config.multipliers,
        )
        torch.save(result, results_save_file)
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(PersonaGeneralizationExperimentConfig, dest="config")
    parser.add_argument("--sge_task_id", type=int, default=None)
    args = parser.parse_args()
    config = args.config
    sge_task_id = args.sge_task_id
    run_persona_generalization_experiment(config, sge_task_id=sge_task_id)
