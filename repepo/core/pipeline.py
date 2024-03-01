from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from transformers.generation import GenerationConfig
import torch

from .types import Completion, Model, Tokenizer
from .format import Formatter

@dataclass
class TokenProb:
    token_id: int
    logprob: float
    text: str


@dataclass
class TextProbs:
    text: str
    token_probs: list[TokenProb]

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


PipelineHook = Callable[[PipelineContext], AbstractContextManager[None]]


@dataclass
class Pipeline:
    """Generation pipeline"""

    model: Model
    tokenizer: Tokenizer
    conversation_history: list[Completion] = field(default_factory=list)
    hooks: list[PipelineHook] = field(default_factory=list)
    
    _print_first_example: bool = True
    
    def calculate_output_logprobs(self, completion: Completion) -> TextProbs:
        """Calculate the logprobs for each token in the prompt + output"""
        base_prompt = completion.prompt
        full_prompt = completion.full_prompt
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
            probs = torch.log_softmax(outputs.logits, dim=-1).detach().cpu()
            # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
            probs = probs[:, :-1, :]
            target_ids = inputs.input_ids[:, 1:].cpu()
            gen_probs = torch.gather(probs, 2, target_ids[:, :, None]).squeeze(-1)[0]
            text_probs: list[TokenProb] = []
            for token, p in zip(target_ids[0], gen_probs):
                if token not in self.tokenizer.all_special_ids:
                    text_probs.append(
                        TokenProb(
                            token_id=token.item(),
                            text=self.tokenizer.decode(token),
                            logprob=p.item(),
                        )
                    )
            return TextProbs(text=full_prompt, token_probs=text_probs)
        raise RuntimeError("Should never get here")
