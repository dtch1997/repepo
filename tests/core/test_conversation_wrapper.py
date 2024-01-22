from repepo.core.conversation_wrapper import ConversationWrapper
from repepo.core.format import InputOutputFormatter
from repepo.core.types import Example


def test_ConversationWrapper_wrap_empty_convo():
    formatter = InputOutputFormatter()
    example = Example(instruction="", input="Paris is in", output="France")
    wrapper = ConversationWrapper()
    output = wrapper.wrap(formatter, example)
    assert output.prompt == "Input:  Paris is in \nOutput: "
    assert output.response == "France"


def test_ConversationWrapper_wrap_with_convo_history():
    formatter = InputOutputFormatter()
    history = [
        Example(instruction="", input="Berlin is in", output="Germany"),
    ]
    example = Example(instruction="", input="Paris is in", output="France")
    wrapper = ConversationWrapper(conversation_history=history)
    output = wrapper.wrap(formatter, example)
    assert (
        output.prompt
        == "Input:  Berlin is in \nOutput: Germany\nInput:  Paris is in \nOutput: "
    )
    assert output.response == "France"


def test_ConversationWrapper_wrap_with_custom_separator():
    formatter = InputOutputFormatter()
    history = [
        Example(instruction="", input="Berlin is in", output="Germany"),
    ]
    example = Example(instruction="", input="Paris is in", output="France")
    wrapper = ConversationWrapper(conversation_history=history, msg_separator=" [SEP] ")
    output = wrapper.wrap(formatter, example)
    assert (
        output.prompt
        == "Input:  Berlin is in \nOutput: Germany [SEP] Input:  Paris is in \nOutput: "
    )
    assert output.response == "France"


def test_ConversationWrapper_wrap_with_custom_template():
    formatter = InputOutputFormatter()
    example = Example(instruction="", input="Paris is in", output="France")
    wrapper = ConversationWrapper(template="[START] {conversation} [END]")
    output = wrapper.wrap(formatter, example)
    assert output.prompt == "[START] Input:  Paris is in \nOutput:  [END]"
    assert output.response == "France"
