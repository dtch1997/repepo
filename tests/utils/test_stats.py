import pytest
from repepo.utils.stats import bernoulli_js_dist, bernoulli_kl_dist


def test_bernoulli_kl_dist_is_negative_if_steered_prob_is_less_than_base_prob() -> None:
    assert bernoulli_kl_dist(0.5, 0.45) < 0
    assert bernoulli_kl_dist(0.5, 0.55) > 0


def test_bernoulli_kl_dist_is_larger_closer_to_0_and_1() -> None:
    assert bernoulli_kl_dist(0.0, 0.05) > bernoulli_kl_dist(0.5, 0.55)
    assert bernoulli_kl_dist(0.9, 0.95) > bernoulli_kl_dist(0.5, 0.55)


def test_bernoulli_js_dist_is_symmetric() -> None:
    assert bernoulli_js_dist(0.5, 0.55) == -1 * bernoulli_js_dist(0.55, 0.5)


def test_bernoulli_js_dist_is_finite() -> None:
    assert bernoulli_js_dist(0, 0.1) < 1.0
    assert bernoulli_js_dist(0.9, 1) < 1.0


def test_bernoulli_js_dist_specific_value() -> None:
    assert bernoulli_js_dist(0.69, 0.71) == pytest.approx(0.000238, abs=1e-6)
    assert bernoulli_js_dist(0.89, 0.91) == pytest.approx(0.000556, abs=1e-6)


def test_bernoulli_js_dist_is_larger_closer_to_0_and_1() -> None:
    assert bernoulli_js_dist(0.0, 0.05) > bernoulli_js_dist(0.5, 0.55)
    assert bernoulli_js_dist(0.9, 0.95) > bernoulli_js_dist(0.5, 0.55)
