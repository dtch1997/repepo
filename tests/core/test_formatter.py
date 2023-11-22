from repepo.core.format import InputOutputFormatter
from repepo.core.format import InstructionFormatter
from repepo.core.types import Completion
from repepo.core.types import Example


def test_input_output_formatter():
    example = Example(instruction="add numbers", input="1, 2", output="3")
    formatter = InputOutputFormatter()
    completion = formatter.apply(example)
    expected = Completion(prompt="Input: add numbers 1, 2 \nOutput: ", response="3")
    assert completion == expected


def test_instruction_formatter():
    example = Example(instruction="add numbers", input="1, 2", output="3")
    formatter = InstructionFormatter()
    completion = formatter.apply(example)
    expected_prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nadd numbers\n\n### Input:\n1, 2\n\n### Response:"
    )
    assert completion.prompt == expected_prompt
    assert completion.response == "3"
