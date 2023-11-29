from abc import ABC
import abc
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar
from typing_extensions import override

# pyright: strict

from repepo.core.types import Example
from repepo.utils.aggregate_dicts import aggregate_dicts
from repepo.utils.metrics import AggregateMetric


class EvalResult(ABC):
    example: Example
    output: str

    @abc.abstractmethod
    def stats(self) -> dict[str, float]:
        pass


EvalResT = TypeVar("EvalResT", bound=EvalResult)


Evaluator = Callable[[Example, str], EvalResT]


@dataclass
class ClassificationEvalResult(EvalResult):
    example: Example
    output: str
    expected: str

    @property
    def is_correct(self) -> bool:
        return self.output.strip().startswith(self.expected.strip())

    @property
    def score(self) -> float:
        return 1.0 if self.is_correct else 0.0

    @override
    def stats(self) -> dict[str, float]:
        return {"score": self.score}


class ClassificationEvaluator:
    def __call__(self, example: Example, output: str) -> ClassificationEvalResult:
        return ClassificationEvalResult(
            example=example, output=output, expected=example.output
        )


@dataclass
class OverallEvalResults(Generic[EvalResT]):
    eval_results: list[EvalResT]

    @property
    def stats(self) -> dict[str, AggregateMetric]:
        result_stats = aggregate_dicts(res.stats() for res in self.eval_results)
        return {stat: AggregateMetric(vals) for stat, vals in result_stats.items()}
