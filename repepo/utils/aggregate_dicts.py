# pyright: strict

from collections import defaultdict
from typing import Iterable, TypeVar

T = TypeVar("T")


def aggregate_dicts(dicts: Iterable[dict[str, T]]) -> dict[str, list[T]]:
    """
    Aggregate a list of dicts into a single dict of lists.

    Parameters:
    - dicts: The list of dicts to aggregate.

    Returns:
    - A dict of lists, where each key is a key from the original dicts and each
      value is a list of the values from the original dicts.
    """
    result: dict[str, list[T]] = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            result[k].append(v)
    return result
