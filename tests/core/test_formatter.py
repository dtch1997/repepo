from repepo.core.format import (
    InputOutputFormatter,
    InstructionFormatter,
    LlamaChatFormatter,
)
from repepo.core.types import Completion, Example


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


def test_llama_chat_formatter():
    example = Example(instruction="add numbers", input="1, 2", output="3")
    formatter = LlamaChatFormatter()
    completion = formatter.apply(example)
    expected_prompt = "[INST] add numbers\n1, 2 [/INST]"
    assert completion.prompt == expected_prompt
    assert completion.response == "3"
