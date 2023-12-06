from transformers import GPTNeoXForCausalLM
from repepo.algorithms.sft import SupervisedFineTuning, SupervisedFineTuningConfig
from repepo.core.pipeline import Pipeline

from repepo.core.types import Example, Tokenizer


def test_SupervisedFineTuning_run(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
    config = SupervisedFineTuningConfig
    config.batch_size = 4
    config.num_train_epochs = 1
    algorithm = SupervisedFineTuning(config)
    dataset = [
        Example(instruction="", input="Paris is in", output="France"),
        Example(instruction="", input="London is in", output="England"),
        Example(instruction="", input="Berlin is in", output="Germany"),
    ]

    new_pipeline = algorithm.run(pipeline, dataset=dataset)

    # Skip testing outputs as they will be gibberish
    # TODO: assert final train loss lower than starting train loss?
    # How to expose this cleanly?
