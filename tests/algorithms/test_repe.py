from repepo.algorithms.repe import RepeReadingControl
from repepo.core.format import InputOutputFormatter
from repepo.core.types import Dataset, Example, Tokenizer
from repepo.core.pipeline import Pipeline
from syrupy import SnapshotAssertion
from transformers.models.gpt_neox import GPTNeoXForCausalLM


def test_RepeReadingControl_build_repe_training_data_and_labels_picks_one_neg_by_default(
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
            incorrect_outputs=["11", "34", "3.14"],
        ),
    ]
    formatter = InputOutputFormatter()
    algorithm = RepeReadingControl()
    training_data, labels = algorithm._build_repe_training_data_and_labels(
        dataset, formatter
    )
    # for some reason the training data isn't grouped, but labels are. This is how it is in the original code.
    assert len(training_data) == 4
    # should pick the first incorrect output only by default
    assert "Germany" in training_data[0]
    assert "France" in training_data[1]
    assert "2" in training_data[2]
    assert "11" in training_data[3]
    # should alternate between flipped and non-flipped labels
    assert labels == [[0, 1], [1, 0]]
    assert training_data == snapshot


def test_RepeReadingControl_build_repe_training_data_and_labels_with_random_incorrect() -> (
    None
):
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
            incorrect_outputs=["11", "34", "3.14"],
        ),
    ]
    formatter = InputOutputFormatter()
    algorithm = RepeReadingControl()
    training_data, labels = algorithm._build_repe_training_data_and_labels(
        dataset, formatter
    )
    # for some reason the training data isn't grouped, but labels are. This is how it is in the original code.
    assert len(training_data) == 4
    # should pick the a random incorrect output
    assert "France" in training_data[1]
    assert "2" in training_data[2]
    # should alternate between flipped and non-flipped labels
    assert labels == [[0, 1], [1, 0]]


def test_RepeReadingControl_build_repe_training_data_and_labels_with_repeat_correct() -> (
    None
):
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
            incorrect_outputs=["11", "34", "3.14"],
        ),
    ]
    formatter = InputOutputFormatter()
    algorithm = RepeReadingControl(multi_answer_method="repeat_correct")
    training_data, labels = algorithm._build_repe_training_data_and_labels(
        dataset, formatter
    )
    # for some reason the training data isn't grouped, but labels are. This is how it is in the original code.
    assert len(training_data) == 10
    # the positive example should be repeated once for each incorrect output
    assert "Germany" in training_data[0]
    assert "France" in training_data[1]
    assert "Italy" in training_data[2]
    assert "France" in training_data[3]
    assert "2" in training_data[4]
    assert "11" in training_data[5]
    assert "2" in training_data[6]
    assert "34" in training_data[7]
    assert "2" in training_data[8]
    assert "3.14" in training_data[9]
    # should alternate between flipped and non-flipped labels
    assert labels == [[0, 1, 0, 1], [1, 0, 1, 0, 1, 0]]


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
    algorithm = RepeReadingControl(multi_answer_method="repeat_correct")
    directions = algorithm._get_directions(pipeline, dataset)
    assert list(directions.activations.keys()) == [-1, -2, -3, -4, -5]
    assert list(directions.signs.keys()) == [-1, -2, -3, -4, -5]
    for act in directions.activations.values():
        assert act.shape == (1, 512)
    for sign in directions.signs.values():
        assert sign in [-1, 1]


def test_RepeReadingControl_run(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    tokenizer.pad_token_id = model.config.eos_token_id
    pipeline = Pipeline(model, tokenizer)

    test_example = Example(
        instruction="",
        input="Paris is in",
        output="France",
        incorrect_outputs=["Germany", "Italy"],
    )
    dataset: Dataset = [
        test_example,
        Example(
            instruction="",
            input="1 + 1 =",
            output="2",
            incorrect_outputs=["11", "34", "3.14"],
        ),
    ]

    original_outputs = pipeline.generate(test_example)

    algorithm = RepeReadingControl()
    algorithm.run(pipeline, dataset)
    new_outputs = pipeline.generate(test_example)

    # TODO: find a better assertion that ensures this is actually doing what it should
    assert original_outputs != new_outputs
