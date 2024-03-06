import matplotlib.pyplot as plt
from repepo.experiments.run_experiment import run_experiment
from repepo.steering.utils.helpers import SteeringConfig, load_results
from repepo.steering.plot_results_by_layer import plot_results_by_layer


def list_configs():
    datasets = [
        "machiavellianism",
        "desire-for-recursive-self-improvement",
        "sycophancy_train",
        "willingness-to-be-non-HHH-to-be-deployed-in-the-real-world",
        # "truthful_qa",
        "believes-abortion-should-be-illegal",
    ]

    return [
        SteeringConfig(
            use_base_model=False,
            model_size="7b",
            train_dataset_name=dataset_name,
            train_split_name="train-dev",
            formatter="llama-chat-formatter",
            aggregator="mean",
            verbose=True,
            layers=[0, 15, 31],
            multipliers=[-1, -0.5, 0, 0.5, 1],
            test_dataset_name=dataset_name,
            test_split_name="val-dev",
            test_completion_template="{prompt} My answer is: {response}",
        )
        for dataset_name in datasets
    ]


if __name__ == "__main__":
    for config in list_configs():
        run_experiment(config)

    # load results
    all_results = []
    for config in list_configs():
        results = load_results(config)
        all_results.append((config, results))

    # plot results
    fig, axs = plt.subplots(len(all_results), 1, figsize=(10, 20))
    for i, (config, results) in enumerate(all_results):
        ax = axs[i]
        plot_results_by_layer(ax, config, results)
        ax.set_title(f"{config.train_dataset_name}")

    fig.tight_layout()
    fig.savefig("results_persona.png")
