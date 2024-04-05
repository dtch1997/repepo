import pytest
from textwrap import dedent
from transformers import (
    GPTNeoXForCausalLM,
)
from repepo.core.types import Completion, Tokenizer
from repepo.experiments.persona_generalization import Persona, persona_pipeline


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
        ("positive", "You are politically conservative."),
        ("negative", "You are not politically conservative."),
        ("baseline", ""),
    ],
)
def test_persona_pipeline_injects_persona_into_prompt(
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
        You are a helpful, honest and concise assistant.
        <</SYS>>
        {expected_prompt_prefix}
        Guns make me happy. I love my guns. A: yes, B: no [/INST] A
        """
    )
    assert prompt.strip() == expected_prompt.strip()
