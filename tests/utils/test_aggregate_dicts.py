# pyright: strict

from repepo.utils.aggregate_dicts import aggregate_dicts


def test_aggregate_dicts_basic_usage() -> None:
    dicts = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        {"a": 5, "b": 6},
    ]
    assert aggregate_dicts(dicts) == {"a": [1, 3, 5], "b": [2, 4, 6]}
