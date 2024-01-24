from dataclasses import dataclass, field

from .types import Completion, Example
from .format import Formatter


def completion_to_str(completion: Completion) -> str:
    return completion.prompt.rstrip() + " " + completion.response.lstrip()


@dataclass
class ConversationWrapper:
    """Wraps a conversation, summarizing a list of messages into a single completion."""

    template: str = "{conversation}"
    conversation_history: list[Example] = field(default_factory=list)
    msg_separator: str = "\n"

    def wrap(self, formatter: Formatter, message: Example) -> Completion:
        completions = formatter.format_conversation(
            [*self.conversation_history, message]
        )
        prefix_completions = completions[:-1]
        final_completion = completions[-1]
        convo_prefix = self.msg_separator.join(
            [completion_to_str(c) for c in prefix_completions]
        )
        convo_str = final_completion.prompt
        if len(convo_prefix) > 0:
            convo_str = convo_prefix + self.msg_separator + final_completion.prompt
        prompt = self.template.format(conversation=convo_str)
        return Completion(prompt=prompt, response=final_completion.response)
