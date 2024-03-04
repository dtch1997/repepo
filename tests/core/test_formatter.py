from textwrap import dedent

from repepo.core.format import (
    IdentityFormatter,
    LlamaChatFormatter,
)
from repepo.core.types import Completion


def test_IdentityFormatter_format_conversation():
    input_completion = Completion(prompt="Paris is in", response="France")
    formatter = IdentityFormatter()
    output_completion = formatter.format_conversation(input_completion)
    assert output_completion.prompt == "Paris is in"
    assert output_completion.response == "France"


def test_IdentityFormatter_format_as_str_with_custom_completion_template():
    input_completion = Completion(prompt="Paris is in", response="France")
    formatter = IdentityFormatter(
        completion_template="Input: {prompt}\n Output: {response}"
    )
    output = formatter.format_as_str(input_completion)
    assert output == "Input: Paris is in\n Output: France"


def test_IdentityFormatter_format_prompt_as_str_with_custom_completion_template():
    input_completion = Completion(prompt="Paris is in", response="France")
    formatter = IdentityFormatter(
        completion_template="Input: {prompt}\n Output: {response}"
    )
    output = formatter.format_prompt_as_str(input_completion)
    assert output == "Input: Paris is in\n Output:"


def test_LlamaChatFormatter_base():
    input_completion = Completion(prompt="add numbers\n1, 2", response="3")
    formatter = LlamaChatFormatter()
    output_completion = formatter.format_conversation(input_completion)
    expected = dedent(
        """
        [INST] <<SYS>>
        You are a helpful, honest and concise assistant.
        <</SYS>>
        
        add numbers
        1, 2 [/INST] 
        """
    )
    assert output_completion.prompt.strip() == expected.strip()
    assert output_completion.response == "3"


def test_LlamaChatFormatter_only_adds_sys_prompt_to_first_message():
    convo = [
        Completion(prompt="add numbers\n1, 2", response="3"),
    ]
    msg = Completion(prompt="add numbers\n3, 5", response="8")

    formatter = LlamaChatFormatter()
    output_completion = formatter.format_conversation(msg, convo)
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
    assert output_completion.prompt.strip() == expected.strip()
    assert output_completion.response == "8"


def test_LlamaChatFormatter_format_conversation_empty_convo():
    input_completion = Completion(prompt="Paris is in", response="France")
    formatter = LlamaChatFormatter()
    output = formatter.format_conversation(input_completion)
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
        Completion(prompt="Berlin is in", response="Germany"),
    ]
    input_completion = Completion(prompt="Paris is in", response="France")
    formatter = LlamaChatFormatter()
    output = formatter.format_conversation(input_completion, history)
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
        Completion(prompt="Berlin is in", response="Germany"),
    ]
    example = Completion(prompt="Paris is in", response="France")
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
    input_completion = Completion(prompt="Paris is in", response="France")
    formatter = LlamaChatFormatter(
        completion_template="{prompt} My answer is: {response}"
    )
    output_completion = formatter.format_conversation(input_completion)
    expected = dedent(
        """
        [INST] <<SYS>>
        You are a helpful, honest and concise assistant.
        <</SYS>>

        Paris is in [/INST] My answer is: France
        """
    )
    assert formatter.format_as_str(output_completion).strip() == expected.strip()
