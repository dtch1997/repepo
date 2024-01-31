""" Translate TQA and save as CSV """

from repepo.experiments.tqa_translate.translate import (
    translate_example,
    translate_to_leetspeak,
    translate_to_pig_latin,
    translate_to_pirate_speak,
    TranslationFn,
)

from repepo.core.types import Dataset
from repepo.data import make_dataset, DatasetSpec
from repepo.data.io import jdump
from repepo.data.make_dataset import get_dataset_dir
from tqdm import tqdm


def translate_dataset(
    dataset: Dataset,
    translation_fn: TranslationFn,
) -> Dataset:
    """Translate a dataset using a translation function"""
    examples = []
    for example in tqdm(dataset, desc="Translating dataset"):
        examples.append(translate_example(example, translation_fn))
    return examples


if __name__ == "__main__":
    datasets = {}
    datasets["tqa_english_train"] = make_dataset(
        DatasetSpec(name="truthfulqa", split=":10%", seed=0)
    )
    datasets["tqa_english_test"] = make_dataset(
        DatasetSpec(name="truthfulqa", split="10:20%", seed=0)
    )

    for language_name, translation_fn in (
        ("leetspeak", translate_to_leetspeak),
        ("pig_latin", translate_to_pig_latin),
        ("pirate_speak", translate_to_pirate_speak),
    ):
        datasets[f"tqa_{language_name}_train"] = translate_dataset(
            datasets["tqa_english_train"],
            translation_fn,
        )
        datasets[f"tqa_{language_name}_test"] = translate_dataset(
            datasets["tqa_english_test"],
            translation_fn,
        )

    for name, dataset in datasets.items():
        dataset_path = get_dataset_dir() / "experimental"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        jdump(dataset, dataset_path / f"{name}.json")