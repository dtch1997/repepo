from textwrap import dedent

from repepo.core.format import (
    InputOutputFormatter,
    InstructionFormatter,
    LlamaChatFormatter,
)
from repepo.core.types import Completion, Example


def test_input_output_formatter():
    example = Example(instruction="add numbers", input="1, 2", output="3")
    formatter = InputOutputFormatter()
    completion = formatter.format_conversation([example])[0]
    expected = Completion(prompt="Input: add numbers 1, 2 \nOutput: ", response="3")
    assert completion == expected


def test_instruction_formatter():
    example = Example(instruction="add numbers", input="1, 2", output="3")
    formatter = InstructionFormatter()
    completion = formatter.format_conversation([example])[0]
    expected_prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nadd numbers\n\n### Input:\n1, 2\n\n### Response:"
    )
    assert completion.prompt == expected_prompt
    assert completion.response == "3"


def test_LlamaChatFormatter_base():
    example = Example(instruction="add numbers", input="1, 2", output="3")
    formatter = LlamaChatFormatter()
    completion = formatter.format_conversation([example])[0]
    expected = dedent(
        """
        [INST] <<SYS>>
        You are a helpful, honest and concise assistant.
        <</SYS>>
        
        add numbers
        1, 2 [/INST] 
        """
    )
    assert completion.prompt.strip() == expected.strip()
    assert completion.response == "3"


def test_LlamaChatFormatter_only_adds_sys_prompt_to_first_message():
    convo = [
        Example(instruction="add numbers", input="1, 2", output="3"),
        Example(instruction="add numbers", input="3, 5", output="8"),
    ]
    formatter = LlamaChatFormatter()
    completions = formatter.format_conversation(convo)
    expected1 = dedent(
        """
        [INST] <<SYS>>
        You are a helpful, honest and concise assistant.
        <</SYS>>
        
        add numbers
        1, 2 [/INST]
        """
    )
    expected2 = dedent(
        """
        [INST] add numbers
        3, 5 [/INST]
        """
    )
    assert completions[0].prompt.strip() == expected1.strip()
    assert completions[0].response == "3"
    assert completions[1].prompt.strip() == expected2.strip()
    assert completions[1].response == "8"
