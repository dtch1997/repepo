from repepo.core.types import Dataset
from repepo.data.multiple_choice.make_bats import (
    convert_bats_dataset,
    get_raw_dataset_dir,
    build_dataset_filename,
    get_dataset_dir,
    jdump,
)

PROMPT = "I live in"


def make_country_capital_with_prompt():
    dataset_path = get_raw_dataset_dir() / (
        "bats/3_Encyclopedic_semantics/E01 [country - capital].txt"
    )
    assert dataset_path.exists()

    with open(dataset_path, "r") as file:
        list_dataset = []
        for line in file:
            element = line.strip().split()
            assert len(element) == 2
            list_dataset.append(tuple(element))

    dataset_name = "country-capital-with-prompt"
    filename = build_dataset_filename(dataset_name)
    mwe_dataset: Dataset = convert_bats_dataset(list_dataset, prompt=PROMPT)
    jdump(mwe_dataset, get_dataset_dir() / "bats" / filename)


if __name__ == "__main__":
    make_country_capital_with_prompt()
