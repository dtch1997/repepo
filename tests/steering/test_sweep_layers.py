from repepo.core.format import LlamaChatFormatter
from repepo.core.types import Tokenizer
from repepo.steering.sweep_layers import SWEEP_DATASETS, sweep_layers
from repepo.steering.utils.helpers import make_dataset
from transformers import (
    GPT2LMHeadModel,
)


def test_sweep_datasets_are_valid():
    for dataset in SWEEP_DATASETS:
        examples = make_dataset(dataset)
        assert len(examples) > 100


def test_sweep_layers(gpt2_model: GPT2LMHeadModel, gpt2_tokenizer: Tokenizer):
    results = sweep_layers(
        model=gpt2_model,
        tokenizer=gpt2_tokenizer,
        layers=(1, 2),
        train_split="0:3",
        test_split="5:8",
        formatter=LlamaChatFormatter(),
        datasets=SWEEP_DATASETS[:1],
        multipliers=(-0.5, 0.0, 0.5),
    )

    assert results.multipliers == [-0.5, 0.0, 0.5]
    assert results.layers == [1, 2]
    assert results.steering_vectors.keys() == {SWEEP_DATASETS[0]}
    assert results.steering_vectors[SWEEP_DATASETS[0]].keys() == {1, 2}
    assert results.steering_results.keys() == {SWEEP_DATASETS[0]}
    assert results.steering_results[SWEEP_DATASETS[0]].keys() == {1, 2}
    for multiplier_results in results.steering_results[SWEEP_DATASETS[0]].values():
        assert len(multiplier_results) == 3
