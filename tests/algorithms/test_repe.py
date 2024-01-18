from repepo.algorithms.repe import (
    RepeReadingControl,
    _find_generation_start_token_index,
)
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

    algorithm = RepeReadingControl(
        patch_generation_tokens_only=False, direction_multiplier=10
    )
    algorithm.run(pipeline, dataset)
    new_outputs = pipeline.generate(test_example)

    # TODO: find a better assertion that ensures this is actually doing what it should
    assert original_outputs != new_outputs


def test_RepeReadingControl_run_logprobs_with_patch_generation_tokens_only(
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

    original_outputs = pipeline.calculate_output_logprobs(test_example)

    algorithm = RepeReadingControl(patch_generation_tokens_only=True)
    algorithm.run(pipeline, dataset)
    new_outputs = pipeline.calculate_output_logprobs(test_example)

    assert original_outputs.sum_logprobs != new_outputs.sum_logprobs
    # only the final token should be patched and thus be different
    for old, new in zip(
        original_outputs.token_probs[:-1], new_outputs.token_probs[:-1]
    ):
        assert old.text == new.text
        assert old.logprob == new.logprob
    assert (
        original_outputs.token_probs[-1].logprob != new_outputs.token_probs[-1].logprob
    )


def test_find_generation_start_token_index_with_trailing_space(
    tokenizer: Tokenizer,
) -> None:
    base_prompt = "Paris is in: "
    full_prompt = "Paris is in: France"
    assert _find_generation_start_token_index(tokenizer, base_prompt, full_prompt) == 3


def test_find_generation_start_token_index_with_trailing_special_chars(
    tokenizer: Tokenizer,
) -> None:
    base_prompt = "<s> Paris is in: </s>"
    full_prompt = "<s> Paris is in: France </s>"
    assert _find_generation_start_token_index(tokenizer, base_prompt, full_prompt) == 6


def test_find_generation_start_token_base(
    tokenizer: Tokenizer,
) -> None:
    base_prompt = "<s> Paris is in:"
    full_prompt = "<s> Paris is in: France"
    assert _find_generation_start_token_index(tokenizer, base_prompt, full_prompt) == 6
