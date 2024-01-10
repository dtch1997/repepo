from repepo.algorithms.repe import RepeReadingControl
from repepo.core.format import InputOutputFormatter
from repepo.core.types import Dataset, Example, Tokenizer
from repepo.core.pipeline import Pipeline
from syrupy import SnapshotAssertion
from transformers.models.gpt_neox import GPTNeoXForCausalLM


def test_RepeReadingControl_build_repe_training_data_and_labels(
    snapshot: SnapshotAssertion,
) -> None:
    dataset: Dataset = [
        Example(
            instruction="",
            input="Paris is in",
            output="France",
            incorrect_outputs=["Germany", "Italy"],
        ),
        Example(
            instruction="",
            input="1 + 1 =",
            output="2",
            incorrect_outputs=["11", "1234", "3.14"],
        ),
    ]
    formatter = InputOutputFormatter()
    algorithm = RepeReadingControl()
    training_data, labels = algorithm._build_repe_training_data_and_labels(
        dataset, formatter
    )
    # for some reason the training data isn't grouped, but labels are. This is how it is in the original code.
    assert len(training_data) == 7
    assert labels == [[1, 0, 0], [1, 0, 0, 0]]
    assert training_data == snapshot


def test_RepeReadingControl_get_directions(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    tokenizer.pad_token_id = model.config.eos_token_id
    pipeline = Pipeline(model, tokenizer)
    dataset: Dataset = [
        Example(
            instruction="",
            input="Paris is in",
            output="France",
            incorrect_outputs=["Germany", "Italy"],
        ),
    ]
    algorithm = RepeReadingControl()
    directions = algorithm._get_directions(pipeline, dataset)
    assert list(directions.activations.keys()) == [-1, -2, -3, -4, -5]
    assert list(directions.signs.keys()) == [-1, -2, -3, -4, -5]
    for act in directions.activations.values():
        assert act.shape == (1, 512)


def test_RepeReadingControl_run(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    tokenizer.pad_token_id = model.config.eos_token_id
    pipeline = Pipeline(model, tokenizer)

    example = Example(
        instruction="",
        input="Paris is in",
        output="France",
        incorrect_outputs=["Germany", "Italy"],
    )
    dataset: Dataset = [example]

    original_outputs = pipeline.generate(example)

    algorithm = RepeReadingControl()
    algorithm.run(pipeline, dataset)
    new_outputs = pipeline.generate(example)

    # TODO: find a better assertion that ensures this is actually doing what it should
    assert original_outputs != new_outputs
