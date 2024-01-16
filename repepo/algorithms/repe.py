from dataclasses import replace
from typing import Literal, Optional
from typing_extensions import override
import random

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
from repepo.core.types import Dataset, Example, Model
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


class RepeReadingControl(Algorithm):
    layer_type: LayerType
    multi_answer_method: MultiAnswerMethod
    reading_template: str
    layers: list[int] | None
    direction_multiplier: float
    layer_config: ModelLayerConfig | None
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
    ):
        self.multi_answer_method = multi_answer_method
        self.layer_type = layer_type
        self.seed = seed
        _validate_reading_template(reading_template)
        self.reading_template = reading_template
        self.layers = layers
        self.layer_config = layer_config
        self.direction_multiplier = direction_multiplier

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
        )

    @override
    def run(self, pipeline: Pipeline, dataset: Dataset) -> Pipeline:
        steering_vector = self._get_steering_vector(pipeline, dataset)
        # this will modify the model in place
        steering_vector.patch_activations(
            model=pipeline.model,
            layer_config=self.layer_config,
            multiplier=self.direction_multiplier,
        )
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
