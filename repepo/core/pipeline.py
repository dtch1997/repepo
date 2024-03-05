from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch

from .types import Completion, Model, Tokenizer
from .format import Formatter, IdentityFormatter


@dataclass
class TokenProb:
    token_id: int
    logprob: float
    logit: float
    text: str


@dataclass
class TextProbs:
    text: str
    token_probs: list[TokenProb]

    @property
    def sum_logits(self) -> float:
        return sum([tp.logit for tp in self.token_probs])

    @property
    def sum_logprobs(self) -> float:
        return sum([tp.logprob for tp in self.token_probs])

    def __repr__(self) -> str:
        return f"TextProbs({self.text}:{self.sum_logprobs:.2f})"


@dataclass
class PipelineContext:
    method: Literal["generate", "logprobs"]
    base_prompt: str
    full_prompt: str
    inputs: Any
    pipeline: "Pipeline"


class PipelineHook(Protocol):
    def __call__(self, context: PipelineContext) -> AbstractContextManager[None]:
        ...


@dataclass
class Pipeline:
    """Generation pipeline"""

    model: Model
    tokenizer: Tokenizer
    formatter: Formatter = field(default_factory=IdentityFormatter)
    conversation_history: list[Completion] = field(default_factory=list)
    hooks: list[PipelineHook] = field(default_factory=list)

    print_first_example: bool = True

    def build_generation_prompt(self, completion: Completion) -> str:
        """Build the generation prompt from the completion"""
        return self.formatter.format_prompt_as_str(
            self.formatter.format_conversation(completion, self.conversation_history)
        )

    def build_full_prompt(self, completion: Completion) -> str:
        """Build the full prompt from the completion"""
        return self.formatter.format_as_str(
            self.formatter.format_conversation(completion, self.conversation_history)
        )

    def calculate_output_logprobs(self, completion: Completion) -> TextProbs:
        """Calculate the logprobs for each token in the prompt + output"""
        base_prompt = self.build_generation_prompt(completion)
        full_prompt = self.build_full_prompt(completion)
        inputs: Any = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        context = PipelineContext(
            method="logprobs",
            base_prompt=base_prompt,
            full_prompt=full_prompt,
            inputs=inputs,
            pipeline=self,
        )
        with ExitStack() as stack:
            for hook in self.hooks:
                stack.enter_context(hook(context))
            outputs = self.model(**inputs, output_hidden_states=False, return_dict=True)
            logits = outputs.logits.detach().cpu()
            logprobs = torch.log_softmax(logits, dim=-1).detach().cpu()

            # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
            logits = logits[:, :-1, :]
            logprobs = logprobs[:, :-1, :]

            # get the logprobs for the target tokens
            target_ids = inputs.input_ids[:, 1:].cpu()
            gen_logprobs = torch.gather(logprobs, 2, target_ids[:, :, None]).squeeze(
                -1
            )[0]
            gen_logits = torch.gather(logits, 2, target_ids[:, :, None]).squeeze(-1)[0]

            text_probs: list[TokenProb] = []

            for token, logprob, logit in zip(target_ids[0], gen_logprobs, gen_logits):
                if token not in self.tokenizer.all_special_ids:
                    text_probs.append(
                        TokenProb(
                            token_id=token.item(),
                            text=self.tokenizer.decode(token),
                            logprob=logprob.item(),
                            logit=logit.item(),
                        )
                    )
            return TextProbs(text=full_prompt, token_probs=text_probs)
        raise RuntimeError("Should never get here")
