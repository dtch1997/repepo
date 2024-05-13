from pathlib import Path
from repepo.experiments.persona_generalization import make_all_datasets
from repepo.experiments.persona_prompts import get_all_persona_prompts

# relative to current path
PERSONA_DATASETS_DIR = (
    Path(__file__).parent.parent.parent / "raw_datasets" / "mwe" / "persona"
)


def test_get_all_persona_prompts():
    make_all_datasets()
    prompts = get_all_persona_prompts()
    all_persona_dataset_paths = list(PERSONA_DATASETS_DIR.glob("*.jsonl"))
    all_persona_datasets = [path.stem for path in all_persona_dataset_paths]
    missing_datasets = set(all_persona_datasets) - prompts.keys()
    assert (
        missing_datasets == set()
    ), f"Missing {len(missing_datasets)} datasets: {missing_datasets}"
    assert len(prompts) > len(all_persona_datasets)
