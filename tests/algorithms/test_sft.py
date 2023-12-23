import torch
from transformers import GPTNeoXForCausalLM
from repepo.algorithms.sft import SupervisedFineTuning, SupervisedFineTuningConfig
from repepo.core.pipeline import Pipeline

from repepo.core.types import Example, Tokenizer


def test_SupervisedFineTuning_run(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    
    NUM_TRAIN_EPOCHS = 100

    pipeline = Pipeline(model, tokenizer)
    config = SupervisedFineTuningConfig(
        num_train_epochs=NUM_TRAIN_EPOCHS,        
        report_to=[],
    )
    algorithm = SupervisedFineTuning(config)
    dataset = [
        Example(instruction="", input="Paris is in", output="France"),
        Example(instruction="", input="London is in", output="England"),
        Example(instruction="", input="Berlin is in", output="Germany"),
    ]

    metrics = algorithm.run(pipeline, dataset=dataset)
    train_state_log_history = metrics['train_state_log_history']
    final_train_state = train_state_log_history[-1]

    assert int(final_train_state['epoch'] == NUM_TRAIN_EPOCHS)
    assert final_train_state['train_loss'] < 0.1
