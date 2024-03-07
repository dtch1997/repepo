import numpy as np
from repepo.steering.utils.helpers import SteeringResult


def steering_efficiency(
    results: list[SteeringResult],
    multiplier_lower_bound=-1.0,
    multiplier_upper_bound=1.0,
):
    slopes = {}

    # Get set of all layers in results
    layers = set([x.layer_id for x in results])

    for layer in layers:
        # Calculate slope of best-fit line within range (-1, 1)
        layer_results = [x for x in results if x.layer_id == layer]
        layer_results.sort(key=lambda x: x.multiplier)
        layer_results = [
            x
            for x in layer_results
            if multiplier_lower_bound <= x.multiplier <= multiplier_upper_bound
        ]
        x = [x.multiplier for x in layer_results]
        y = [x.logit_diff for x in layer_results]
        m, _ = np.polyfit(x, y, 1)
        slopes[layer] = m

    return slopes
