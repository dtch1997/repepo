from repepo.experiments.persona_prompts import get_all_persona_prompts


def test_get_all_persona_prompts():
    prompts = get_all_persona_prompts()
    assert len(prompts) > 60
