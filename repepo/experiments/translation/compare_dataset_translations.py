from collections import defaultdict
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Callable, NamedTuple

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


class DatasetVersion(NamedTuple):
    name: str
    base_dataset_name: str
    base_lang_or_style: LangOrStyleCode | None = None
    generator: Callable[[Dataset], Dataset] = lambda dataset: dataset


@dataclass
class TranslatedDatasetGroup:
    name: str
    translations: dict[LangOrStyleCode, str]
    extra_versions: dict[str, DatasetVersion] = field(default_factory=dict)

    def add_extra_version(
        self,
        name: str,
        base_lang_or_style: LangOrStyleCode | None,
        generator: Callable[[Dataset], Dataset],
    ):
        self.extra_versions[name] = DatasetVersion(
            name=name,
            base_dataset_name=self.name,
            base_lang_or_style=base_lang_or_style,
            generator=generator,
        )

    def get_dataset_name(self, lang_or_style: LangOrStyleCode | None) -> str:
        if lang_or_style is None:
            return self.name
        return self.translations[lang_or_style]

    def get_lang_or_style(self, version_name: str) -> LangOrStyleCode | None:
        for ds in self.all_dataset_versions:
            if ds.name == version_name:
                return ds.base_lang_or_style
        return None

    @property
    def all_dataset_versions(self) -> list[DatasetVersion]:
        results: list[DatasetVersion] = [
            DatasetVersion(name="en", base_dataset_name=self.name)
        ]
        for lang_or_style, dataset_name in self.translations.items():
            results.append(
                DatasetVersion(
                    name=lang_or_style,
                    base_lang_or_style=lang_or_style,
                    base_dataset_name=dataset_name,
                )
            )
        for generator in self.extra_versions.values():
            results.append(generator)
        return sorted(results, key=attrgetter("name"))


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
    existing_results: dict[str, SteeringVector] | None = None,
) -> dict[str, SteeringVector]:
    results: dict[str, SteeringVector] = existing_results or {}

    repe_algo = RepeReadingControl(
        patch_generation_tokens_only=True,
        # CAA reads from position -2, since the last token is ")"
        read_token_index=-2,
        # CAA skips the first generation token, so doing the same here to match
        skip_first_n_generation_tokens=1,
    )

    for ds in group.all_dataset_versions:
        if ds.name in results:
            print(f"Skipping {ds.name} since it already has a steering vector")
            continue
        base_train_dataset = make_dataset(
            DatasetSpec(name=ds.base_dataset_name, split=train_split)
        )
        train_dataset = ds.generator(base_train_dataset)
        pipeline = Pipeline(
            model,
            tokenizer,
            formatter=LlamaChatFormatter(
                # use the translated version of the prompt
                system_prompt=load_translation(
                    TS.llama2_chat_caa_system_message, ds.base_lang_or_style
                )
            ),
        )
        repe_algo.run(pipeline, train_dataset)
        hook = pipeline.hooks[0]
        assert isinstance(hook, SteeringHook)
        results[ds.name] = hook.steering_vector

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
    steering_vectors: dict[str, SteeringVector],
    direction_multipliers: list[float] = [-1.0, -0.5, 0.0, 0.5, 1.0],
    use_eval_prefix: bool = False,
    existing_results: dict[str, list[dict[str, float]]] | None = None,
) -> dict[str, list[dict[str, float]]]:
    results: dict[str, list[dict[str, float]]] = existing_results or {}
    for ds in dataset_group.all_dataset_versions:
        if ds.name in results:
            print(f"Skipping {ds.name} since it already has results")
            continue
        steering_vector = steering_vectors[ds.name]
        results[ds.name] = []
        base_dataset = make_dataset(
            DatasetSpec(name=ds.base_dataset_name, split=test_split)
        )
        dataset = ds.generator(base_dataset)
        for direction_multiplier in direction_multipliers:
            print(f"Testing {ds.name} at {direction_multiplier}")
            result = test_steering_performance(
                model,
                tokenizer,
                steering_vector,
                dataset,
                ds.base_lang_or_style,
                direction_multiplier=direction_multiplier,
                use_eval_prefix=use_eval_prefix,
            )
            print(result.metrics)
            results[ds.name].append(result.metrics)
    return results


@dataclass
class CrossLangSteeringResult:
    labels: list[str]  # size N
    base_results: list[float]  # size N
    steering_results: list[list[float]]  # size N x N

    def get_base_result(self, label: str) -> float:
        return self.base_results[self.labels.index(label)]

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
    steering_vectors: dict[str, SteeringVector],
    direction_multiplier: float = 1.0,
    use_eval_prefix: bool = False,
    existing_results: CrossLangSteeringResult | None = None,
) -> CrossLangSteeringResult:
    labels: list[str] = [ds.name for ds in dataset_group.all_dataset_versions]
    base_results = []
    steering_results = []
    for ds in dataset_group.all_dataset_versions:
        cur_steering_results = []
        base_dataset = make_dataset(
            DatasetSpec(name=ds.base_dataset_name, split=test_split)
        )
        dataset = ds.generator(base_dataset)
        if existing_results is not None and ds.name in existing_results.labels:
            base_result = existing_results.get_base_result(ds.name)
            base_results.append(base_result)
        else:
            base_result = test_steering_performance(
                model,
                tokenizer,
                steering_vectors[ds.name],
                dataset,
                ds.base_lang_or_style,
                direction_multiplier=0.0,
                use_eval_prefix=use_eval_prefix,
            )
            base_results.append(base_result.metrics["average_key_prob"])
        for steering_version in labels:
            if (
                existing_results is not None
                and steering_version in existing_results.labels
                and ds.name in existing_results.labels
            ):
                cur_steering_results.append(
                    existing_results.steering_results[labels.index(ds.name)][
                        labels.index(steering_version)
                    ]
                )
                continue
            steering_vector = steering_vectors[steering_version]
            result = test_steering_performance(
                model,
                tokenizer,
                steering_vector,
                dataset,
                dataset_group.get_lang_or_style(steering_version),
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
