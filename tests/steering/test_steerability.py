import numpy as np

from repepo.steering.steerability import (
    get_steerability_slope,
    get_steerability_residuals,
    get_steerabilty_spearman,
)


def test_steerability_positive():
    multipliers = np.arange(100)
    propensities = (np.arange(100) / 100)[np.newaxis, :].repeat(10, axis=0)
    expected_slope = 0.01
    expected_spearman = 1.0
    actual_slope = get_steerability_slope(multipliers, propensities)
    actual_residual = get_steerability_residuals(multipliers, propensities)
    actual_spearman = get_steerabilty_spearman(multipliers, propensities)

    assert actual_residual.shape == (10,)
    assert np.isclose(actual_slope, expected_slope).all()
    assert np.isclose(actual_spearman, expected_spearman).all()


def test_steerability_negative():
    multipliers = np.arange(100)
    propensities = (np.arange(100) / 100 * -1)[np.newaxis, :].repeat(10, axis=0)
    expected_slope = -0.01
    expected_spearman = -1.0
    actual_slope = get_steerability_slope(multipliers, propensities)
    actual_spearman = get_steerabilty_spearman(multipliers, propensities)

    assert np.isclose(actual_slope, expected_slope).all()
    assert np.isclose(actual_spearman, expected_spearman).all()
