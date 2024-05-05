from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from transformers.generation import GenerationConfig

import torch

from .types import Completion, Model, Tokenizer
from .format import Formatter, IdentityFormatter


@dataclass
class TokenProb:
    token_id: int
    # Note: the logit, logprob are for this token, not the next token
    logprob: float
    logit: float
    text: str | None = None
    # Metrics for logits of other tokens that were in this token position
    logit_mean: float | None = None
    logit_std: float | None = None
    logit_skew: float | None = None
    logit_kurtosis: float | None = None
    logit_100_quantile: float | None = None
    logit_75_quantile: float | None = None
    logit_50_quantile: float | None = None
    logit_25_quantile: float | None = None
    logit_0_quantile: float | None = None

    @property
    def logit_max(self) -> float:
        if self.logit_100_quantile is None:
            raise ValueError("logit_100_quantile is not set")
        return self.logit_100_quantile

    @property
    def logit_min(self) -> float:
        if self.logit_0_quantile is None:
            raise ValueError("logit_0_quantile is not set")
        return self.logit_0_quantile


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
    def __call__(self, context: PipelineContext) -> AbstractContextManager[None]: ...


def compute_moments(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute mean, std, skew, kurtosis along the specified dimension
    Input: tensor of shape (batch_size, num_classes)
    Returns a tensor of shape (batch_size, 4)
    """
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    skew = ((tensor - mean) ** 3).mean(dim=dim, keepdim=True) / (std**3)
    kurtosis = ((tensor - mean) ** 4).mean(dim=dim, keepdim=True) / (std**4)
    return torch.cat([mean, std, skew, kurtosis], dim=dim)


def compute_quantiles(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute quantiles along the specified dimension
    Input: tensor of shape (batch_size, num_classes)
    Returns a tensor of shape (batch_size, num_quantiles)
    """
    quantile_thresholds = torch.tensor([0, 0.25, 0.5, 0.75, 1], device=tensor.device)
    quantiles = torch.quantile(tensor, quantile_thresholds, dim=dim)
    # transpose to get the shape (batch_size, num_quantiles)
    quantiles = quantiles.transpose(0, 1)
    return quantiles


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

    def generate(
        self,
        completion: Completion,
        generation_config: GenerationConfig | None = None,
        remove_base_prompt: bool = True,
    ) -> str:
        """Generate a completion for a given example"""
        base_prompt = self.build_generation_prompt(completion)
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

    @torch.no_grad()
    def calculate_output_logprobs(
        self, completion: Completion, slim_results: bool = False
    ) -> TextProbs:
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
            logits = outputs.logits
            logprobs = torch.log_softmax(logits, dim=-1)

            # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
            logits = logits[:, :-1, :]
            logprobs = logprobs[:, :-1, :]

            # get the logprobs for the target tokens
            # first, get the tokens which correspond to completions
            target_ids = inputs.input_ids[:, 1:]
            # next, select the indices corresponding to the target token ids
            gen_logprobs = (
                torch.gather(logprobs, 2, target_ids[:, :, None]).squeeze(-1)[0].cpu()
            )
            gen_logits = (
                torch.gather(logits, 2, target_ids[:, :, None]).squeeze(-1)[0].cpu()
            )

            # For each logit, calculate the moments and quantiles
            # logits is of shape (1, seq_len, vocab_size)
            assert logits.shape[0] == 1
            logits = logits[0]
            text_probs: list[TokenProb] = []

            logit_moments = None
            logit_quantiles = None
            if not slim_results:
                logit_moments = compute_moments(logits, dim=-1).cpu()
                logit_quantiles = compute_quantiles(logits, dim=-1).cpu()

            for i, (token, logprob, logit) in enumerate(
                zip(
                    target_ids[0].cpu(),
                    gen_logprobs,
                    gen_logits,
                )
            ):
                if token not in self.tokenizer.all_special_ids:
                    token_prob = TokenProb(
                        token_id=token.item(),
                        logprob=logprob.item(),
                        logit=logit.item(),
                    )
                    if not slim_results:
                        assert logit_moments is not None
                        assert logit_quantiles is not None
                        token_prob.text = self.tokenizer.decode(token)
                        token_prob.logit_mean = logit_moments[i, 0].item()
                        token_prob.logit_std = logit_moments[i, 1].item()
                        token_prob.logit_skew = logit_moments[i, 2].item()
                        token_prob.logit_kurtosis = logit_moments[i, 3].item()
                        token_prob.logit_0_quantile = logit_quantiles[i, 0].item()
                        token_prob.logit_25_quantile = logit_quantiles[i, 1].item()
                        token_prob.logit_50_quantile = logit_quantiles[i, 2].item()
                        token_prob.logit_75_quantile = logit_quantiles[i, 3].item()
                        token_prob.logit_100_quantile = logit_quantiles[i, 4].item()
                    text_probs.append(token_prob)
            return TextProbs(text=full_prompt, token_probs=text_probs)
        raise RuntimeError("Should never get here")
