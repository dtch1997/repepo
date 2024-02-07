from collections import defaultdict
from dataclasses import dataclass

from steering_vectors import SteeringVector

from repepo.algorithms.repe import RepeReadingControl, SteeringHook
from repepo.core.evaluate import (
    EvalResult,
    MultipleChoiceAccuracyEvaluator,
    NormalizedCorrectProbabilityEvaluator,
    evaluate,
    select_repe_layer_at_eval,
    update_completion_template_at_eval,
)
from repepo.core.pipeline import Pipeline
from repepo.core.format import LlamaChatFormatter
from repepo.core.types import Dataset, Model, Tokenizer
from repepo.data.make_dataset import (
    DatasetSpec,
    list_datasets,
    make_dataset,
    parse_dataset_filename,
)
from repepo.translation.constants import LangOrStyleCode
from repepo.translation.load_translation import load_translation, TS


@dataclass
class TranslatedDatasetGroup:
    name: str
    translations: dict[LangOrStyleCode, str]

    @property
    def all_datasets(self) -> list[tuple[LangOrStyleCode | None, str]]:
        results: list[tuple[LangOrStyleCode | None, str]] = [(None, self.name)]
        for lang_or_style, dataset_name in self.translations.items():
            results.append((lang_or_style, dataset_name))
        return results


def get_translated_dataset_groups() -> list[TranslatedDatasetGroup]:
    grouped_datasets = defaultdict(list)
    for dataset_name in list_datasets():
        file_params = parse_dataset_filename(dataset_name)
        grouped_datasets[file_params.base_name].append(dataset_name)
    translation_groups = []
    for name, dataset_names in grouped_datasets.items():
        if len(dataset_names) == 1:
            continue
        translations: dict[LangOrStyleCode, str] = {}
        for dataset_name in dataset_names:
            info = parse_dataset_filename(dataset_name)
            if info.lang_or_style is None:
                continue
            translations[info.lang_or_style] = dataset_name
        translation_groups.append(TranslatedDatasetGroup(name, translations))
    return translation_groups


def train_group_steering_vectors(
    # NOTE: this is assumed to be a llama2 chat model
    model: Model,
    tokenizer: Tokenizer,
    group: TranslatedDatasetGroup,
    train_split: str,
) -> dict[LangOrStyleCode | None, SteeringVector]:
    results: dict[LangOrStyleCode | None, SteeringVector] = {}

    repe_algo = RepeReadingControl(
        patch_generation_tokens_only=True,
        # CAA reads from position -2, since the last token is ")"
        read_token_index=-2,
        # CAA skips the first generation token, so doing the same here to match
        skip_first_n_generation_tokens=1,
    )

    for lang_or_style, dataset_name in group.all_datasets:
        train_dataset = make_dataset(DatasetSpec(name=dataset_name, split=train_split))
        pipeline = Pipeline(
            model,
            tokenizer,
            formatter=LlamaChatFormatter(
                # use the translated version of the prompt
                system_prompt=load_translation(
                    TS.llama2_chat_caa_system_message, lang_or_style
                )
            ),
        )
        repe_algo.run(pipeline, train_dataset)
        hook = pipeline.hooks[0]
        assert isinstance(hook, SteeringHook)
        results[lang_or_style] = hook.steering_vector

    return results


def test_steering_performance(
    model: Model,
    tokenizer: Tokenizer,
    steering_vector: SteeringVector,
    test_dataset: Dataset,
    lang_or_style: LangOrStyleCode | None,
    direction_multiplier: float = 1.0,
    skip_first_n_generation_tokens: int = 0,
    patch_generation_tokens_only: bool = True,
    use_eval_prefix: bool = False,
    layer: int = 15,
) -> EvalResult:
    pipeline = Pipeline(
        model,
        tokenizer,
        formatter=LlamaChatFormatter(
            # use the translated version of the prompt
            system_prompt=load_translation(
                TS.llama2_chat_caa_system_message, lang_or_style
            )
        ),
    )
    hook = SteeringHook(
        steering_vector,
        direction_multiplier=direction_multiplier,
        patch_generation_tokens_only=patch_generation_tokens_only,
        skip_first_n_generation_tokens=skip_first_n_generation_tokens,
        layer_config=None,
    )
    pipeline.hooks.append(hook)
    eval_hooks = [select_repe_layer_at_eval(layer)]
    if use_eval_prefix:
        prefix_str = load_translation(TS.caa_eval_prefix, lang_or_style)
        eval_hooks.append(
            update_completion_template_at_eval(
                "{prompt} [PREFIX] {response}".replace("[PREFIX]", prefix_str)
            )
        )
    return evaluate(
        pipeline,
        test_dataset,
        evaluators=[
            NormalizedCorrectProbabilityEvaluator(),
            MultipleChoiceAccuracyEvaluator(),
        ],
        eval_hooks=eval_hooks,
    )


def sweep_steering_group_self_performance(
    model: Model,
    tokenizer: Tokenizer,
    dataset_group: TranslatedDatasetGroup,
    test_split: str,
    steering_vectors: dict[LangOrStyleCode | None, SteeringVector],
    direction_multipliers: list[float] = [-1.0, -0.5, 0.0, 0.5, 1.0],
    use_eval_prefix: bool = False,
) -> dict[LangOrStyleCode | None, list[dict[str, float]]]:
    results: dict[LangOrStyleCode | None, list[dict[str, float]]] = {}
    for lang_or_style, dataset_name in dataset_group.all_datasets:
        steering_vector = steering_vectors[lang_or_style]
        results[lang_or_style] = []
        dataset = make_dataset(DatasetSpec(name=dataset_name, split=test_split))
        for direction_multiplier in direction_multipliers:
            print(f"Testing {lang_or_style} at {direction_multiplier}")
            result = test_steering_performance(
                model,
                tokenizer,
                steering_vector,
                dataset,
                lang_or_style,
                direction_multiplier=direction_multiplier,
                use_eval_prefix=use_eval_prefix,
            )
            print(result.metrics)
            results[lang_or_style].append(result.metrics)
    return results


@dataclass
class CrossLangSteeringResult:
    labels: list[LangOrStyleCode | None]  # size N
    base_results: list[float]  # size N
    steering_results: list[list[float]]  # size N x N

    @property
    def steering_deltas(self) -> list[list[float]]:
        return [
            [steering_result - base_result for steering_result in steering_results]
            for base_result, steering_results in zip(
                self.base_results, self.steering_results
            )
        ]


def steer_across_languages(
    model: Model,
    tokenizer: Tokenizer,
    dataset_group: TranslatedDatasetGroup,
    test_split: str,
    steering_vectors: dict[LangOrStyleCode | None, SteeringVector],
    direction_multiplier: float = 1.0,
    use_eval_prefix: bool = False,
) -> CrossLangSteeringResult:
    labels: list[LangOrStyleCode | None] = [
        lang for lang, _ in dataset_group.all_datasets
    ]
    labels.sort(key=lambda x: x if x is not None else "en")
    base_results = []
    steering_results = []
    for data_lang in labels:
        cur_steering_results = []
        dataset_name = (
            dataset_group.name
            if data_lang is None
            else dataset_group.translations[data_lang]
        )
        dataset = make_dataset(DatasetSpec(name=dataset_name, split=test_split))
        base_result = test_steering_performance(
            model,
            tokenizer,
            steering_vectors[data_lang],
            dataset,
            data_lang,
            direction_multiplier=0.0,
            use_eval_prefix=use_eval_prefix,
        )
        base_results.append(base_result.metrics["average_key_prob"])
        for steering_lang in labels:
            steering_vector = steering_vectors[steering_lang]
            result = test_steering_performance(
                model,
                tokenizer,
                steering_vector,
                dataset,
                steering_lang,
                direction_multiplier=direction_multiplier,
                use_eval_prefix=use_eval_prefix,
            )
            cur_steering_results.append(result.metrics["average_key_prob"])
        steering_results.append(cur_steering_results)
    return CrossLangSteeringResult(
        labels=labels,
        base_results=base_results,
        steering_results=steering_results,
    )
