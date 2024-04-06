import pathlib
from repepo.variables import Environ

ALL_ABSTRACT_CONCEPT_DATASETS = [
    # persona
    "anti-immigration",
    "believes-abortion-should-be-illegal",
    "conscientiousness",
    # "desire-for-acquiring-compute",
    # "risk-seeking",
    # "openness",
    # "self-replication",
    # "very-small-harm-justifies-very-large-benefit",
    # xrisk
    "corrigible-neutral-HHH",
    "myopic-reward",
    "power-seeking-inclination",
    # sycophancy
    "sycophancy_train",
]

SOME_TOKEN_CONCEPT_DATASETS = [
    # token
    # "D01 [noun+less_reg]",
    # "E01 [country - capital]",
    # "I01 [noun - plural_reg]",
]

ALL_TOKEN_CONCEPT_DATASETS = [
    fp.stem for fp in pathlib.Path(Environ.DatasetDir).glob("bats/*.json")
]

ALL_LANGUAGES = ["en", "fr", "ja", "pirate", "zh"]

ALL_LLAMA_7B_LAYERS = [0, 5, 10, 11, 12, 13, 14, 15, 20, 25, 31]

ALL_MULTIPLIERS = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
