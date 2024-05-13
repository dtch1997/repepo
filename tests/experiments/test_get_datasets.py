from repepo.experiments.get_datasets import get_all_prompts


def test_get_all_prompts():
    assert len(get_all_prompts()) == 40
