from repepo.algorithms.repe import RepeReadingControl
from repepo.core.format import InputOutputFormatter
from repepo.core.types import Dataset, Example, Tokenizer
from repepo.core.pipeline import Pipeline
from syrupy import SnapshotAssertion
from transformers import GPTNeoXForCausalLM


def test_RepeReadingControl_build_steering_vector_training_data_picks_one_neg_by_default(
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
    training_data = algorithm._build_steering_vector_training_data(dataset, formatter)
    assert len(training_data) == 2
    # should pick the first incorrect output only by default
    assert "France" in training_data[0].positive_prompt
    assert "Germany" in training_data[0].negative_prompt
    assert "2" in training_data[1].positive_prompt
    assert "11" in training_data[1].negative_prompt
    assert training_data == snapshot


def test_RepeReadingControl_build_steering_vector_training_data_with_random_incorrect() -> (
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
    training_data = algorithm._build_steering_vector_training_data(dataset, formatter)
    assert len(training_data) == 2
    assert "France" in training_data[0].positive_prompt
    assert "2" in training_data[1].positive_prompt


def test_RepeReadingControl_build_steering_vector_training_data_with_repeat_correct() -> (
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
    training_data = algorithm._build_steering_vector_training_data(dataset, formatter)
    assert len(training_data) == 5
    # the positive example should be repeated once for each incorrect output
    assert "Germany" in training_data[0].negative_prompt
    assert "France" in training_data[0].positive_prompt
    assert "Italy" in training_data[1].negative_prompt
    assert "France" in training_data[1].positive_prompt
    assert "2" in training_data[2].positive_prompt
    assert "11" in training_data[2].negative_prompt
    assert "2" in training_data[3].positive_prompt
    assert "34" in training_data[3].negative_prompt
    assert "2" in training_data[4].positive_prompt
    assert "3.14" in training_data[4].negative_prompt


def test_RepeReadingControl_get_steering_vector(
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
    steering_vector = algorithm._get_steering_vector(pipeline, dataset)
    assert list(steering_vector.layer_activations.keys()) == [0, 1, 2, 3, 4, 5]
    for act in steering_vector.layer_activations.values():
        assert act.shape == (512,)


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
