import json
import matplotlib.pyplot as plt

from copy import deepcopy
from pathlib import Path
from repepo.experiments.plot_results import (
    plot_in_distribution_data_for_layer,
)
from repepo.experiments.evaluate_tqa_caa import EvaluateCaaResult

if __name__ == "__main__":
    source_languages = ["english", "leetspeak"]
    target_language = "english"

    experiments_dir = Path("repepo/experiments")
    save_suffix = "max_new_tokens=100_type=in_distribution_few_shot=none_do_projection=False_use_base_model=False_model_size=13b_add_every_token_position=False"
    results_fp = f"results_{save_suffix}.json"

    results_paths = [
        experiments_dir
        / f"tqa_{source_language}_train_tqa_{target_language}_val"
        / "results"
        / results_fp
        for source_language in source_languages
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    linestyles = ["solid", "dashed"]
    label_suffixes = deepcopy(source_languages)
    for results_path in results_paths:
        with open(results_path, "r") as f:
            _results = json.load(f)
            results = [EvaluateCaaResult(**res) for res in _results]

        ax = plot_in_distribution_data_for_layer(
            ax,
            results,
            f"Steering vector applied to {target_language}",
            [15, 16, 17, 18],
            linestyle=linestyles.pop(0),
            label_suffix=label_suffixes.pop(0),
        )

    fig.tight_layout()
    fig.savefig(experiments_dir / f"results_target={target_language}.png")
