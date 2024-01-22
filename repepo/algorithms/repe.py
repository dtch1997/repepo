from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Literal, Optional
from typing_extensions import override
import random

import torch
from repepo.core.pipeline import PipelineContext

from steering_vectors import (
    LayerType,
    SteeringVector,
    train_steering_vector,
    SteeringVectorTrainingSample,
    LayerMatcher,
    ModelLayerConfig,
    guess_and_enhance_layer_config,
)

from repepo.core import Pipeline
from repepo.core.format import Formatter
from repepo.core.types import Dataset, Example, Model, Tokenizer
from repepo.algorithms.base import Algorithm

MultiAnswerMethod = Literal["first_incorrect", "random_incorrect", "repeat_correct"]

DEFAULT_READING_TEMPLATE = "Consider the amount of truthfulness in the following answer.\n\n{question} {answer}\n\nThe amount of truthfulness in the answer is "


def _validate_reading_template(reading_template: str):
    if "{answer}" not in reading_template:
        raise ValueError(
            "reading_template must contain {answer} to be used with RepEngReadingControl"
        )
    if "{question}" not in reading_template:
        raise ValueError(
            "reading_template must contain {question} to be used with RepEngReadingControl"
        )


@dataclass
class SteeringHook:
    """
    Pipeline hook that applies a steering vector to the model.
    All relevant state for the hook is stored in this class.
    If the params included in this class are changed, it will
    affect future generation and logprob calls using this hook.
    """

    steering_vector: SteeringVector
    direction_multiplier: float
    patch_generation_tokens_only: bool
    layer_config: ModelLayerConfig | None

    # PipelineContext is created in both `pipeline.generate` or `pipeline.calculate_output_logprobs`,
    # It also contains info about the current prompt which is used to determine which tokens to patch.
    @contextmanager
    def __call__(self, context: PipelineContext):
        handle = None
        try:
            min_token_index = 0
            if self.patch_generation_tokens_only:
                min_token_index = _find_generation_start_token_index(
                    context.pipeline.tokenizer,
                    context.base_prompt,
                    context.full_prompt,
                )
            handle = self.steering_vector.patch_activations(
                model=context.pipeline.model,
                layer_config=self.layer_config,
                multiplier=self.direction_multiplier,
                min_token_index=min_token_index,
            )
            yield
        finally:
            if handle is not None:
                handle.remove()


class RepeReadingControl(Algorithm):
    layer_type: LayerType
    multi_answer_method: MultiAnswerMethod
    reading_template: str
    layers: list[int] | None
    direction_multiplier: float
    layer_config: ModelLayerConfig | None
    patch_generation_tokens_only: bool
    read_token_index: int
    seed: int

    def __init__(
        self,
        reading_template: str = DEFAULT_READING_TEMPLATE,
        layer_type: LayerType = "decoder_block",
        multi_answer_method: MultiAnswerMethod = "first_incorrect",
        seed: int = 0,
        layers: Optional[list[int]] = None,
        layer_config: Optional[ModelLayerConfig] = None,
        direction_multiplier: float = 1.0,
        patch_generation_tokens_only: bool = True,
        skip_reading: bool = False,
        override_vector: Optional[SteeringVector] = None,
        skip_control: bool = False,
        # For (A/B) datasets, the second last token corresponds to 'A' or 'B'
        # which is what the CAA paper extracts.
        # Reference: https://github.com/nrimsky/SycophancySteering/blob/25f93a1f1aad51f94288f52d01f6a10d10f42bf1/generate_vectors.py#L102C13-L102C67
        read_token_index: int = -2,
    ):
        self.multi_answer_method = multi_answer_method
        self.layer_type = layer_type
        self.seed = seed
        self.patch_generation_tokens_only = patch_generation_tokens_only
        _validate_reading_template(reading_template)
        self.reading_template = reading_template
        self.layers = layers
        self.read_token_index = read_token_index
        self.layer_config = layer_config
        self.direction_multiplier = direction_multiplier

        self.skip_reading = skip_reading
        self.override_vector = override_vector
        self.skip_control = skip_control
        if self.skip_reading and self.override_vector is None:
            raise RuntimeError(
                "Either reading or override vector must be provided for control"
            )

    def _build_steering_vector_training_data(
        self, dataset: Dataset, formatter: Formatter
    ) -> list[SteeringVectorTrainingSample]:
        paired_prompts: list[SteeringVectorTrainingSample] = []
        for example in dataset:
            example_prompts = self._convert_example_to_training_samples(
                example, formatter
            )
            paired_prompts.extend(example_prompts)
        return paired_prompts

    def _get_layer_matcher_for_model(self, model: Model) -> LayerMatcher:
        layer_config = guess_and_enhance_layer_config(model, self.layer_config)
        if self.layer_type not in layer_config:
            raise ValueError(
                f"layer_type {self.layer_type} not found in model layer_config, please provide an expicit layer_config"
            )
        return layer_config[self.layer_type]

    @torch.no_grad()
    def _get_steering_vector(
        self, pipeline: Pipeline, dataset: Dataset
    ) -> SteeringVector:
        repe_training_data = self._build_steering_vector_training_data(
            dataset, pipeline.formatter
        )
        return train_steering_vector(
            pipeline.model,
            pipeline.tokenizer,
            repe_training_data,
            layers=self.layers,
            layer_type=self.layer_type,
            layer_config=self.layer_config,
            move_to_cpu=True,
            read_token_index=self.read_token_index,
        )

    @override
    def run(self, pipeline: Pipeline, dataset: Dataset) -> Pipeline:
        # Steering vector reading
        # NOTE: The hooks read from this steering vector.

        if self.override_vector is not None:
            steering_vector: SteeringVector = self.override_vector
        elif not self.skip_reading:
            steering_vector: SteeringVector = self._get_steering_vector(
                pipeline, dataset
            )
        else:
            raise RuntimeError(
                "Either reading or override vector must be provided for control"
            )

        # Creating the hooks that will do steering vector control
        # NOTE: How this works is that we create a context manager that creates a hook
        # whenever we are in a `PipelineContext`'s scope.
        # After exiting the context, the hook is deleted.

        # need to use a hook so we can inspect the current thing being generated to know
        # which tokens to patch
        steering_hook = SteeringHook(
            steering_vector=steering_vector,
            direction_multiplier=self.direction_multiplier,
            patch_generation_tokens_only=self.patch_generation_tokens_only,
            layer_config=self.layer_config,
        )

        if not self.skip_control:
            pipeline.hooks.append(steering_hook)

        return pipeline

    def _convert_example_to_training_samples(
        self, example: Example, formatter: Formatter
    ) -> list[SteeringVectorTrainingSample]:
        """Converts an example to the format expected by steering-vectors"""
        if example.incorrect_outputs is None:
            raise ValueError("RepeReadingControl requires incorrect_outputs to be set")
        incorrect_examples = [
            replace(example, output=output) for output in example.incorrect_outputs
        ]
        correct_examples = [example]
        if self.multi_answer_method == "repeat_correct":
            correct_examples = [example] * len(example.incorrect_outputs)
        elif self.multi_answer_method == "first_incorrect":
            incorrect_examples = [incorrect_examples[0]]
        elif self.multi_answer_method == "random_incorrect":
            rand_gen = random.Random(f"{self.seed}-{example.input}")
            incorrect_examples = [rand_gen.choice(incorrect_examples)]
        else:
            raise ValueError(f"Unknown multi_answer_method {self.multi_answer_method}")
        assert len(incorrect_examples) == len(correct_examples)
        paired_completions = [
            (formatter.apply(pos), formatter.apply(neg))
            for pos, neg in zip(correct_examples, incorrect_examples)
        ]
        return [
            SteeringVectorTrainingSample(
                positive_prompt=self.reading_template.format(
                    question=pos.prompt, answer=pos.response
                ),
                negative_prompt=self.reading_template.format(
                    question=neg.prompt, answer=neg.response
                ),
            )
            for pos, neg in paired_completions
        ]


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
