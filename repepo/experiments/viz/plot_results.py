import matplotlib.pyplot as plt
from repepo.steering.utils.helpers import load_results, SteeringConfig, SteeringResult


def plot_results_for_layer(ax, config: SteeringConfig, results: list[SteeringResult]):
    for i, layer in enumerate(config.layers):
        layer_results = [x for x in results if x.layer_id == layer]
        layer_results.sort(key=lambda x: x.multiplier)

        ax.plot(
            [x.multiplier for x in layer_results],
            [x.logit_diff for x in layer_results],
            marker="o",
            linestyle="dashed",
            markersize=5,
            linewidth=2.5,
            label=f"Layer {layer}",
        )

    ax.set_title(f"{config.train_dataset_name}")
    ax.set_xlabel("Multiplier")
    ax.set_ylabel("Mean logit difference")
    ax.legend()
    return ax


if __name__ == "__main__":
    import simple_parsing

    config = simple_parsing.parse(config_class=SteeringConfig, add_config_path_arg=True)
    print(config)

    fig, ax = plt.subplots()
    results = load_results(config)
    plot_results_for_layer(ax, config, results)
    fig.tight_layout()

    save_path = f"results_{config.make_save_suffix()}.png"
    print("Saving results to: ", save_path)
    fig.savefig(save_path)
