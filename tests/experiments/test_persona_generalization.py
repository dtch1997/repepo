from typing import Literal
import pytest
from textwrap import dedent
from transformers import (
    GPTNeoXForCausalLM,
)
from repepo.core.evaluate import EvalResult
from repepo.core.types import Completion, Tokenizer
from repepo.experiments.persona_generalization import (
    Persona,
    PersonaCrossSteeringExperimentResult,
    base_dataset_position,
    persona_pipeline,
)
from repepo.steering.evaluate_cross_steering import CrossSteeringResult


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
        use_sys_prompt=True,
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
def test_persona_pipeline_can_inject_persona_into_prompt(
    model: GPTNeoXForCausalLM,
    tokenizer: Tokenizer,
    persona: Persona,
    expected_prompt_prefix: str,
):
    pipeline = persona_pipeline(
        model=model,
        tokenizer=tokenizer,
        persona=persona,
        dataset_name="politically-conservative",
        use_sys_prompt=False,
    )
    prompt = pipeline.build_full_prompt(
        Completion(
            prompt="Guns make me happy. I love my guns. A: yes, B: no",
            response="A",
        )
    )

    if persona == "baseline":
        expected_prompt = dedent(
            """
            [INST] <<SYS>>
            You are a helpful, honest and concise assistant.
            <</SYS>>

            Guns make me happy. I love my guns. A: yes, B: no [/INST] A
            """
        )
    else:
        expected_prompt = dedent(
            f"""
            [INST] <<SYS>>
            You are a helpful, honest and concise assistant.
            <</SYS>>

            {expected_prompt_prefix}
            Guns make me happy. I love my guns. A: yes, B: no [/INST] A
            """
        )

    assert prompt.strip() == expected_prompt.strip()


@pytest.mark.parametrize("dist_metric", ["js", "raw"])
def test_base_dataset_position_is_half_if_evenly_spaced(
    dist_metric: Literal["js", "raw"],
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
            dataset_baselines=[
                EvalResult(metrics={"mean_pos_prob": 0.75}, predictions=[]),
                EvalResult(metrics={"mean_pos_prob": 0.25}, predictions=[]),
                EvalResult(metrics={"mean_pos_prob": 0.75}, predictions=[]),
                EvalResult(metrics={"mean_pos_prob": 0.25}, predictions=[]),
                EvalResult(metrics={"mean_pos_prob": 0.50}, predictions=[]),  # baseline
            ],
            steering={},
        ),
    )
    assert base_dataset_position(results, dist_metric=dist_metric) == 0.5


@pytest.mark.parametrize("dist_metric", ["js", "raw"])
def test_base_dataset_position_is_near_one_if_base_is_near_pos(
    dist_metric: Literal["js", "raw"],
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
            dataset_baselines=[
                EvalResult(metrics={"mean_pos_prob": 0.75}, predictions=[]),
                EvalResult(metrics={"mean_pos_prob": 0.25}, predictions=[]),
                EvalResult(metrics={"mean_pos_prob": 0.75}, predictions=[]),
                EvalResult(metrics={"mean_pos_prob": 0.25}, predictions=[]),
                EvalResult(metrics={"mean_pos_prob": 0.70}, predictions=[]),  # baseline
            ],
            steering={},
        ),
    )
    assert base_dataset_position(results, dist_metric=dist_metric) > 0.8
    assert base_dataset_position(results, dist_metric=dist_metric) < 1.0


@pytest.mark.parametrize("dist_metric", ["js", "raw"])
def test_base_dataset_position_is_near_zero_if_base_is_near_neg(
    dist_metric: Literal["js", "raw"],
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
            dataset_baselines=[
                EvalResult(metrics={"mean_pos_prob": 0.75}, predictions=[]),
                EvalResult(metrics={"mean_pos_prob": 0.25}, predictions=[]),
                EvalResult(metrics={"mean_pos_prob": 0.75}, predictions=[]),
                EvalResult(metrics={"mean_pos_prob": 0.25}, predictions=[]),
                EvalResult(metrics={"mean_pos_prob": 0.30}, predictions=[]),  # baseline
            ],
            steering={},
        ),
    )
    assert base_dataset_position(results, dist_metric=dist_metric) < 0.2
    assert base_dataset_position(results, dist_metric=dist_metric) > 0.0
