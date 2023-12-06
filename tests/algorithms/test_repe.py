# pyright: strict, reportMissingTypeStubs=false

from transformers import GPTNeoXForCausalLM
from repepo.algorithms.repe import (
    RepE,
    convert_old_to_new,
    convert_repepo_format_to_old_format,
)
from repepo.core.pipeline import Pipeline
from repepo.core.format import MinimalFormatter

from repepo.core.types import Example, Tokenizer, Dataset


def test_RepE_run(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer, repe_toy_dataset: Dataset
) -> None:
    pipeline = Pipeline(model=model, tokenizer=tokenizer, formatter=MinimalFormatter())
    algorithm = RepE(coeff=3)
    example = Example(
        instruction="",
        input="Math question: 6 - 3 = ",
        output="",
    )

    original_output = pipeline.generate(example=example)
    algorithm.run(pipeline=pipeline, dataset=repe_toy_dataset)
    repe_output = pipeline.generate(example=example)

    assert original_output == "\n\\begin{cases}\n\\frac{1}{"
    assert (
        repe_output[:53]
        == "\n\\begin{document}\n\n\\begin{document}\n\n\\begin{equation}"
    )


def test_repe_dataset_conversion(repe_bias_dataset) -> None:
    new_dataset = convert_old_to_new(repe_bias_dataset)
    new_new_dataset = convert_repepo_format_to_old_format(new_dataset)
    assert new_new_dataset == repe_bias_dataset["train"]
