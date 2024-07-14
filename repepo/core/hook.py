from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal
from repepo.core.pipeline import PipelineContext
from steering_vectors import (
    SteeringVector,
    ModelLayerConfig,
    ablation_then_addition_operator,
)

from repepo.core.types import Tokenizer
from repepo.core.pipeline import PipelineHook


@dataclass
class SteeringHook(PipelineHook):
    """
    Pipeline hook that applies a steering vector to the model.
    All relevant state for the hook is stored in this class.
    If the params included in this class are changed, it will
    affect future generation and logprob calls using this hook.
    """

    steering_vector: SteeringVector
    direction_multiplier: float
    patch_generation_tokens_only: bool
    skip_first_n_generation_tokens: int
    layer_config: ModelLayerConfig | None
    patch_operator: Literal["add", "ablate_then_add"] = "add"

    # PipelineContext is created in both `pipeline.generate` or `pipeline.calculate_output_logprobs`,
    # It also contains info about the current prompt which is used to determine which tokens to patch.
    @contextmanager
    def __call__(self, context: PipelineContext):
        handle = None
        try:
            min_token_index = 0
            if self.patch_generation_tokens_only:
                gen_start_index = _find_generation_start_token_index(
                    context.pipeline.tokenizer,
                    context.base_prompt,
                    context.full_prompt,
                )
                min_token_index = gen_start_index + self.skip_first_n_generation_tokens
            # multiplier 0 is equivalent to no steering, so just skip patching in that case
            if self.direction_multiplier != 0:
                handle = self.steering_vector.patch_activations(
                    model=context.pipeline.model,
                    layer_config=self.layer_config,
                    multiplier=self.direction_multiplier,
                    min_token_index=min_token_index,
                    operator=(
                        ablation_then_addition_operator()
                        if self.patch_operator == "ablate_then_add"
                        else None
                    ),
                )
            yield
        finally:
            if handle is not None:
                handle.remove()


def _find_generation_start_token_index(
    tokenizer: Tokenizer, base_prompt: str, full_prompt: str
) -> int:
    """Finds the index of the first generation token in the prompt"""
    base_toks = tokenizer.encode(base_prompt)
    full_toks = tokenizer.encode(full_prompt)
    prompt_len = len(base_toks)
    # try to account for cases where the final prompt token is different
    # from the first generation token, ususally weirdness with spaces / special chars
    for i, (base_tok, full_tok) in enumerate(zip(base_toks, full_toks)):
        if base_tok != full_tok:
            prompt_len = i
            break
    # The output of the last prompt token is the first generation token
    # so need to subtract 1 here
    return prompt_len - 1
