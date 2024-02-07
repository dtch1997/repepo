from collections import defaultdict
from dataclasses import dataclass

from steering_vectors import SteeringVector

from repepo.algorithms.repe import RepeReadingControl, SteeringHook
from repepo.core.pipeline import Pipeline
from repepo.core.format import LlamaChatFormatter
from repepo.core.types import Model, Tokenizer
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
