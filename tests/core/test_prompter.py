from repepo.core.prompt import FewShotPrompter, IdentityPrompter, LlamaChatPrompter
from repepo.core.types import Completion


def test_identity_prompter():
    completion = Completion(prompt="Hello", response="Hi!")
    prompter = IdentityPrompter()
    output = prompter.apply(completion)
    assert completion == output


def test_few_shot_prompter():
    completion = Completion(prompt="Hello", response="Hi!")
    examples = [Completion(prompt="Example 1", response="Response 1")]
    prompter = FewShotPrompter(examples)
    output = prompter.apply(completion)
    assert (
        output.prompt == examples[0].prompt + " " + examples[0].response + "\n\nHello"
    )


def test_llama_chat_prompter():
    completion = Completion(prompt="Hello", response="Hi!")
    prompter = LlamaChatPrompter()
    output = prompter.apply(completion)
    assert output.prompt == (
        "<<SYS>>\nYou are a helpful, honest and concise assistant.\n<</SYS>>\n\nHello"
    )
