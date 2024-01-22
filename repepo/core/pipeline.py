from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from transformers.generation import GenerationConfig
import torch

from .types import Example, Model, Tokenizer
from .prompt import Prompter, IdentityPrompter
from .format import Formatter, InputOutputFormatter


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


PipelineHook = Callable[[PipelineContext], AbstractContextManager]


@dataclass
class Pipeline:
    """Generation pipeline"""

    model: Model
    tokenizer: Tokenizer
    prompter: Prompter = field(default_factory=IdentityPrompter)
    formatter: Formatter = field(default_factory=InputOutputFormatter)
    hooks: list[PipelineHook] = field(default_factory=list)

    def build_generation_prompt(self, example: Example) -> str:
        """Build a prompt for generation"""
        completion = self.formatter.apply(example)
        completion = self.prompter.apply(completion)
        return completion.prompt

    def generate(
        self,
        example: Example,
        generation_config: Optional[GenerationConfig] = None,
        remove_base_prompt: bool = True,
    ) -> str:
        """Generate a completion for a given example"""
        base_prompt = self.build_generation_prompt(example)
        inputs: Any = self.tokenizer(base_prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        context = PipelineContext(
            method="generate",
            base_prompt=base_prompt,
            full_prompt=base_prompt,
            inputs=inputs,
            pipeline=self,
        )
        with ExitStack() as stack:
            for hook in self.hooks:
                stack.enter_context(hook(context))
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )[0]
            outputs_str = self.tokenizer.decode(outputs, skip_special_tokens=True)
            if remove_base_prompt:
                return outputs_str.replace(base_prompt, "")
            return outputs_str
        raise RuntimeError("Should never get here")

    def calculate_output_logprobs(self, example: Example) -> TextProbs:
        """Calculate the logprobs for each token in the prompt + output"""
        base_prompt = self.build_generation_prompt(example)
        full_prompt = base_prompt + example.output
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
            probs = torch.log_softmax(outputs.logits, dim=-1)
            # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
            probs = probs[:, :-1, :]
            target_ids = inputs.input_ids[:, 1:]
            gen_probs = (
                torch.gather(probs, 2, target_ids[:, :, None])
                .squeeze(-1)[0]
                .detach()
                .cpu()
            )
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
