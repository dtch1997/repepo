import matplotlib.pyplot as plt
from repepo.steering.run_experiment import run_experiment
from repepo.steering.utils.helpers import SteeringConfig, load_results
from repepo.steering.plot_results_by_layer import plot_results_by_layer


def run_sweep(configs: list[SteeringConfig], suffix=""):
    for config in configs:
        run_experiment(config)

    # load results
    all_results = []
    for config in configs:
        results = load_results(config)
        all_results.append((config, results))

    # plot results
    fig, axs = plt.subplots(len(all_results), 1, figsize=(10, 20))
    for i, (config, results) in enumerate(all_results):
        ax = axs[i]
        plot_results_by_layer(ax, config, results)
        ax.set_title(f"{config.train_dataset_name}")

    fig.tight_layout()
    if suffix:
        suffix = f"_{suffix}"
    fig.savefig(f"results{suffix}.png")
