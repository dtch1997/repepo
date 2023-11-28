from torch import tensor, allclose
from repepo.core.types import Tokenizer

from repepo.data.utils import tokenize


def test_tokenize(tokenizer: Tokenizer) -> None:
    # TODO: pythia default tokenizer max length breaks tokenize()
    tokenizer.model_max_length = 256
    res = tokenize(["Hello world"], tokenizer)
    assert len(res["input_ids"]) == 1
    assert len(res["labels"]) == 1
    assert allclose(res["input_ids"][0], tensor([12092, 1533]))
    assert allclose(res["labels"][0], tensor([12092, 1533]))
