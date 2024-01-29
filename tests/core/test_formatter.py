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
    completion = formatter.format_conversation(example)
    expected = Completion(prompt="Input: add numbers 1, 2 \nOutput:", response="3")
    assert completion == expected


def test_instruction_formatter():
    example = Example(instruction="add numbers", input="1, 2", output="3")
    formatter = InstructionFormatter()
    completion = formatter.format_conversation(example)
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
    completion = formatter.format_conversation(example)
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
    ]
    msg = Example(instruction="add numbers", input="3, 5", output="8")

    formatter = LlamaChatFormatter()
    completion = formatter.format_conversation(msg, convo)
    expected = dedent(
        """
        [INST] <<SYS>>
        You are a helpful, honest and concise assistant.
        <</SYS>>
        
        add numbers
        1, 2 [/INST] 3
        [INST] add numbers
        3, 5 [/INST]
        """
    )
    assert completion.prompt.strip() == expected.strip()
    assert completion.response == "8"


def test_LlamaChatFormatter_format_conversation_empty_convo():
    example = Example(instruction="", input="Paris is in", output="France")
    formatter = LlamaChatFormatter()
    output = formatter.format_conversation(example, [])
    expected_prompt = dedent(
        """
        [INST] <<SYS>>
        You are a helpful, honest and concise assistant.
        <</SYS>>

        Paris is in [/INST]
        """
    )
    assert output.prompt.strip() == expected_prompt.strip()
    assert output.response == "France"


def test_LlamaChatFormatter_format_conversation_with_convo_history():
    history = [
        Example(instruction="", input="Berlin is in", output="Germany"),
    ]
    example = Example(instruction="", input="Paris is in", output="France")
    formatter = LlamaChatFormatter()
    output = formatter.format_conversation(example, history)
    expected_prompt = dedent(
        """
        [INST] <<SYS>>
        You are a helpful, honest and concise assistant.
        <</SYS>>

        Berlin is in [/INST] Germany
        [INST] Paris is in [/INST] 
        """
    )
    assert output.prompt.strip() == expected_prompt.strip()
    assert output.response == "France"


def test_LlamaChatFormatter_format_conversation_with_overwritten_separator():
    history = [
        Example(instruction="", input="Berlin is in", output="Germany"),
    ]
    example = Example(instruction="", input="Paris is in", output="France")
    formatter = LlamaChatFormatter(msg_separator="\n[SEP]\n")
    output = formatter.format_conversation(example, history)
    expected_prompt = dedent(
        """
        [INST] <<SYS>>
        You are a helpful, honest and concise assistant.
        <</SYS>>

        Berlin is in [/INST] Germany
        [SEP]
        [INST] Paris is in [/INST] 
        """
    )
    assert output.prompt.strip() == expected_prompt.strip()
    assert output.response == "France"


def test_LlamaChatFormatter_format_completion_with_caa_test_template():
    example = Example(instruction="", input="Paris is in", output="France")
    formatter = LlamaChatFormatter(
        completion_template="{prompt} My answer is: {response}"
    )
    completion = formatter.format_conversation(example)
    expected = dedent(
        """
        [INST] <<SYS>>
        You are a helpful, honest and concise assistant.
        <</SYS>>

        Paris is in [/INST] My answer is: France
        """
    )
    assert formatter.format_completion(completion).strip() == expected.strip()
