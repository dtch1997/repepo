import matplotlib.pyplot as plt
from collections import namedtuple
from repepo.steering.utils.helpers import SteeringConfig, load_result


def plot_results_by_layer_and_multiplier(ax: plt.Axes, configs: list[SteeringConfig]):
    results = [load_result(config.eval_hash) for config in configs]

    Datum = namedtuple("Datum", ["layer", "multiplier", "logit_diff"])
    data: list[Datum] = []
    for config, result in zip(configs, results):
        assert config.eval_hash == result.config_hash
        data.append(
            Datum(
                layer=config.layer,
                multiplier=config.multiplier,
                logit_diff=result.logit_diff,
            )
        )

    layers = set([x.layer for x in data])
    for i, layer in enumerate(layers):
        layer_data = [x for x in data if x.layer == layer]
        layer_data.sort(key=lambda x: x.multiplier)

        ax.plot(
            [x.multiplier for x in layer_data],
            [x.logit_diff for x in layer_data],
            marker="o",
            linestyle="dashed",
            markersize=5,
            linewidth=2.5,
            label=f"Layer {layer}",
        )

        # Calculate stdev of logit diff
        logit_diffs = [x.logit_diff for x in layer_data]
        mean = sum(logit_diffs) / len(logit_diffs)
        variance = sum([((x - mean) ** 2) for x in logit_diffs]) / len(logit_diffs)
        stdev = variance**0.5

        # Plot error bar
        ax.errorbar(
            [x.multiplier for x in layer_data],
            [x.logit_diff for x in layer_data],
            yerr=stdev,
            fmt="o",
            linestyle="dashed",
            markersize=5,
            linewidth=2.5,
            label=f"Layer {layer}",
        )

    ax.set_xlabel("Multiplier")
    ax.set_ylabel("Mean logit difference")
    ax.legend()
    return ax
