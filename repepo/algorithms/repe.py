from dataclasses import replace
from typing import Any, Literal, NamedTuple, Optional, cast
from typing_extensions import override
import numpy as np

import torch
from repepo.core import Pipeline
from repepo.core.format import Formatter
from repepo.core.types import Dataset, Model
from repepo.repe.rep_reading_pipeline import RepReadingPipeline
from repepo.algorithms.base import Algorithm
from repepo.utils.layer_matching import (
    LayerMatcher,
    ModelLayerConfig,
    get_num_matching_layers,
    guess_and_enhance_layer_config,
)
from repepo.utils.model_patcher import LayerType, ModelPatcher

DirectionMethod = Literal["pca", "cluster_mean", "random"]

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


class RepeTrainingData(NamedTuple):
    # bizarrely, repe data labels are a list of lists of ints, but the prompts are just a list of strings
    # why are the prompts not grouped the same way as the labels?
    prompts: list[str]
    labels: list[list[int]]


class RepeDirections(NamedTuple):
    activations: dict[int, torch.Tensor]
    # why is this a dict of np arrays instead of a dict of torch tensors? mystery
    signs: dict[int, np.ndarray]


class RepeReadingControl(Algorithm):
    direction_method: DirectionMethod
    layer_type: LayerType
    reading_template: str
    layers: list[int] | None
    n_difference: int
    batch_size: int
    max_length: int
    layer_config: ModelLayerConfig | None
    direction_finder_kwargs: dict[str, Any]

    def __init__(
        self,
        reading_template: str = DEFAULT_READING_TEMPLATE,
        direction_method: DirectionMethod = "pca",
        layer_type: LayerType = "decoder_block",
        n_difference: int = 1,  # TODO: what does this do?
        batch_size: int = 8,
        max_length: int = 2048,
        layers: Optional[list[int]] = None,
        layer_config: Optional[ModelLayerConfig] = None,
        # TODO: remove this when refactoring repe reading pipeline
        direction_finder_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.direction_method = direction_method
        self.layer_type = layer_type
        _validate_reading_template(reading_template)
        self.reading_template = reading_template
        self.layers = layers
        self.layer_config = layer_config
        self.n_difference = n_difference
        self.batch_size = batch_size
        self.max_length = max_length
        self.direction_finder_kwargs = direction_finder_kwargs or {}

    def _build_repe_training_data_and_labels(
        self, dataset: Dataset, formatter: Formatter
    ) -> RepeTrainingData:
        prompts: list[str] = []
        grouped_labels: list[list[int]] = []
        for example in dataset:
            if example.incorrect_outputs is None:
                raise ValueError(
                    "RepEngReadingControl requires incorrect_outputs to be set"
                )
            incorrect_examples = [
                replace(example, output=output) for output in example.incorrect_outputs
            ]
            label_group = [1, *([0] * len(incorrect_examples))]
            grouped_labels.append(label_group)
            group_examples = [example, *incorrect_examples]
            for group_example in group_examples:
                completion = formatter.apply(group_example)
                prompts.append(
                    self.reading_template.format(
                        question=completion.prompt, answer=completion.response
                    )
                )
        return RepeTrainingData(prompts=prompts, labels=grouped_labels)

    def _get_layer_matcher_for_model(self, model: Model) -> LayerMatcher:
        layer_config = guess_and_enhance_layer_config(model, self.layer_config)
        if self.layer_type not in layer_config:
            raise ValueError(
                f"layer_type {self.layer_type} not found in model layer_config, please provide an expicit layer_config"
            )
        return layer_config[self.layer_type]

    def _get_layers_with_default(self, model: Model) -> list[int]:
        """Helper to fill in the default layers for the model if none are provided"""
        if self.layers is not None:
            return self.layers
        layer_matcher = self._get_layer_matcher_for_model(model)
        num_layers = get_num_matching_layers(model, layer_matcher)
        return list(range(-1, -num_layers, -1))

    def _get_directions(self, pipeline: Pipeline, dataset: Dataset) -> RepeDirections:
        layers = self._get_layers_with_default(pipeline.model)
        repe_training_data = self._build_repe_training_data_and_labels(
            dataset, pipeline.formatter
        )
        repe_reading_pipeline = RepReadingPipeline(
            model=pipeline.model, tokenizer=pipeline.tokenizer
        )

        rep_reader = cast(
            Any,
            repe_reading_pipeline.get_directions(
                train_inputs=repe_training_data.prompts,
                train_labels=repe_training_data.labels,
                hidden_layers=layers,
                n_difference=self.n_difference,
                batch_size=self.batch_size,
                direction_method=self.direction_method,
                direction_finder_kwargs=self.direction_finder_kwargs,
                # this must be for the tokenizer
                max_length=self.max_length,
                padding="longest",
            ),
        )
        return RepeDirections(
            activations={
                # rep reader retuns np.ndarray values, not tensors, so need to convert
                key: torch.FloatTensor(val)
                for key, val in rep_reader.directions.items()
            },
            signs=rep_reader.direction_signs,
        )

    @override
    def run(self, pipeline: Pipeline, dataset: Dataset) -> Pipeline:
        directions = self._get_directions(pipeline, dataset)
        model_patcher = ModelPatcher(pipeline.model, self.layer_config)
        # this will modify the model in place
        model_patcher.patch_activations(
            directions.activations,
            layer_type=self.layer_type,
        )
        return pipeline
