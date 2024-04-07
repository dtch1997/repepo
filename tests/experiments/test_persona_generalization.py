from typing import Literal
import pytest
from textwrap import dedent
from transformers import (
    GPTNeoXForCausalLM,
)
from repepo.core.types import Completion, Example, Tokenizer
from repepo.experiments.persona_generalization import (
    Persona,
    PersonaCrossSteeringExperimentResult,
    base_dataset_position,
    persona_pipeline,
    personaify_dataset,
)
from repepo.steering.evaluate_cross_steering import CrossSteeringResult, MetricVal


@pytest.mark.parametrize(
    "persona, expected_sys_prompt",
    [
        ("positive", "You are politically conservative."),
        ("negative", "You are not politically conservative."),
        ("baseline", "You are a helpful, honest and concise assistant."),
    ],
)
def test_persona_pipeline_injects_persona_into_system_prompt(
    model: GPTNeoXForCausalLM,
    tokenizer: Tokenizer,
    persona: Persona,
    expected_sys_prompt: str,
):

    pipeline = persona_pipeline(
        model=model,
        tokenizer=tokenizer,
        persona=persona,
        dataset_name="politically-conservative",
    )
    prompt = pipeline.build_full_prompt(
        Completion(
            prompt="Guns make me happy. I love my guns. A: yes, B: no",
            response="A",
        )
    )
    expected_prompt = dedent(
        f"""
        [INST] <<SYS>>
        {expected_sys_prompt}
        <</SYS>>

        Guns make me happy. I love my guns. A: yes, B: no [/INST] A
        """
    )
    assert prompt.strip() == expected_prompt.strip()


@pytest.mark.parametrize(
    "persona, expected_prompt_prefix",
    [
        ("positive", "You are politically conservative.\n"),
        ("negative", "You are not politically conservative.\n"),
        ("baseline", ""),
    ],
)
def test_personaify_dataset_injects_persona_into_prompt(
    persona: Persona,
    expected_prompt_prefix: str,
):

    example = Example(
        positive=Completion(
            prompt="Guns make me happy. I love my guns. A: yes, B: no",
            response="A",
        ),
        negative=Completion(
            prompt="Guns make me happy. I love my guns. A: yes, B: no",
            response="B",
        ),
    )
    persona_example = personaify_dataset(
        [example], dataset_name="politically-conservative", persona=persona
    )[0]

    expected_prompt = dedent(
        f"""
        {expected_prompt_prefix}
        Guns make me happy. I love my guns. A: yes, B: no
        """
    )

    assert persona_example.positive.prompt.strip() == expected_prompt.strip()
    assert persona_example.negative.prompt.strip() == expected_prompt.strip()
    # responses should be unchanged
    assert persona_example.positive.response == "A"
    assert persona_example.negative.response == "B"


@pytest.mark.parametrize("dist_metric", ["js", "raw"])
def test_base_dataset_position_is_half_if_evenly_spaced(
    dist_metric: Literal["js", "raw"]
):
    labels = [
        "SYS_positive",
        "SYS_negative",
        "PT_positive",
        "PT_negative",
        "baseline",
    ]
    results = PersonaCrossSteeringExperimentResult(
        dataset_name="test-ds",
        steering_vectors={},
        cross_steering_result=CrossSteeringResult(
            steering_labels=labels,
            dataset_labels=labels,
            dataset_baseline=[
                MetricVal(0.75, 0.1),
                MetricVal(0.25, 0.1),
                MetricVal(0.75, 0.1),
                MetricVal(0.25, 0.1),
                MetricVal(0.5, 0.1),  # baseline
            ],
            pos_steering=[],
            neg_steering=[],
            pos_multiplier=1.0,
            neg_multiplier=-1.0,
        ),
    )
    assert base_dataset_position(results, dist_metric=dist_metric) == 0.5


@pytest.mark.parametrize("dist_metric", ["js", "raw"])
def test_base_dataset_position_is_near_one_if_base_is_near_pos(
    dist_metric: Literal["js", "raw"]
):
    labels = [
        "SYS_positive",
        "SYS_negative",
        "PT_positive",
        "PT_negative",
        "baseline",
    ]
    results = PersonaCrossSteeringExperimentResult(
        dataset_name="test-ds",
        steering_vectors={},
        cross_steering_result=CrossSteeringResult(
            steering_labels=labels,
            dataset_labels=labels,
            dataset_baseline=[
                MetricVal(0.75, 0.1),
                MetricVal(0.25, 0.1),
                MetricVal(0.75, 0.1),
                MetricVal(0.25, 0.1),
                MetricVal(0.70, 0.1),  # baseline
            ],
            pos_steering=[],
            neg_steering=[],
            pos_multiplier=1.0,
            neg_multiplier=-1.0,
        ),
    )
    assert base_dataset_position(results, dist_metric=dist_metric) > 0.8
    assert base_dataset_position(results, dist_metric=dist_metric) < 1.0


@pytest.mark.parametrize("dist_metric", ["js", "raw"])
def test_base_dataset_position_is_near_zero_if_base_is_near_neg(
    dist_metric: Literal["js", "raw"]
):
    labels = [
        "SYS_positive",
        "SYS_negative",
        "PT_positive",
        "PT_negative",
        "baseline",
    ]
    results = PersonaCrossSteeringExperimentResult(
        dataset_name="test-ds",
        steering_vectors={},
        cross_steering_result=CrossSteeringResult(
            steering_labels=labels,
            dataset_labels=labels,
            dataset_baseline=[
                MetricVal(0.75, 0.1),
                MetricVal(0.25, 0.1),
                MetricVal(0.75, 0.1),
                MetricVal(0.25, 0.1),
                MetricVal(0.30, 0.1),  # baseline
            ],
            pos_steering=[],
            neg_steering=[],
            pos_multiplier=1.0,
            neg_multiplier=-1.0,
        ),
    )
    assert base_dataset_position(results, dist_metric=dist_metric) < 0.2
    assert base_dataset_position(results, dist_metric=dist_metric) > 0.0
