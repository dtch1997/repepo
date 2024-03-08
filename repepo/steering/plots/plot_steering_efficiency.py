import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from repepo.steering.utils.helpers import SteeringConfig, load_result
from repepo.steering.run_sweep import get_sweep_variables

Datum = namedtuple("Datum", ["layer", "multiplier", "logit_diff"])


def compute_steering_efficiency(
    layer_data: list[Datum],
    multiplier_lower_bound=-1.0,
    multiplier_upper_bound=1.0,
) -> float:
    """Calculate the slope of the best-fit line within range (-1, 1) for a given layer."""
    filtered_data = [
        x
        for x in layer_data
        if multiplier_lower_bound <= x.multiplier <= multiplier_upper_bound
    ]
    x = [x.multiplier for x in filtered_data]
    y = [x.logit_diff for x in filtered_data]
    slope, _ = np.polyfit(x, y, 1)
    return slope


def plot_steering_efficiency(
    ax: plt.Axes,
    configs: list[SteeringConfig],
    label: str | None = None,
    multiplier_lower_bound=-1.0,
    multiplier_upper_bound=1.0,
) -> plt.Axes:
    """Plot steering efficiency per layer."""

    results = [load_result(config.eval_hash) for config in configs]
    sweep_variables = get_sweep_variables(configs)
    assert "layer" in sweep_variables, "layer must be a sweep variable"
    assert "multiplier" in sweep_variables, "multiplier must be a sweep variable"

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
    steering_efficiency = {layer: 0.0 for layer in layers}
    for i, layer in enumerate(layers):
        layer_data = [x for x in data if x[0] == layer]
        layer_data.sort(key=lambda x: x.multiplier)
        steering_efficiency[layer] = compute_steering_efficiency(
            layer_data, multiplier_lower_bound, multiplier_upper_bound
        )

    ax.plot(
        [x for x in steering_efficiency.keys()],
        [x for x in steering_efficiency.values()],
        marker="o",
        linestyle="dashed",
        markersize=5,
        linewidth=2.5,
        label=label,
    )

    return ax
