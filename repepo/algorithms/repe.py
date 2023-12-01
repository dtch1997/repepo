from repepo.core import Dataset, Pipeline
from repepo.core.types import Example, RepExample
from repepo.repe.rep_reading_pipeline import RepReadingPipeline
from repepo.repe.rep_control_pipeline import RepControlPipeline

from repepo.algorithms.base import BaseAlgorithm

from repepo.core.prompt import IdentityPrompter
from repepo.core.format import InstructionFormatter

import torch
from collections import defaultdict


def group_by_id(dataset: Dataset):
    pairs = defaultdict(list)

    # Group examples by their id
    for example in dataset:
        assert isinstance(example, RepExample)
        pairs[example.id].append(example)

    return pairs

def convert_repepo_format_to_old_format(dataset: Dataset):
    grouped_dataset = group_by_id(dataset)
    old_dataset = {"data": [], "labels": []}

    for pair_id, examples in grouped_dataset.items():
        if len(examples) != 2:
            raise ValueError(f"Expected 2 examples for id {pair_id}, got {len(examples)}")

        example1, example2 = examples
        if example1.direction == example2.direction:
            raise ValueError(f"Expected different directions for examples with id {pair_id}")

        old_dataset["data"].append(example1.instruction + example1.input)
        old_dataset["labels"].append([example1.direction, ""])
        old_dataset["data"].append(example2.instruction + example2.input)
        old_dataset["labels"][-1][1] = example2.direction

    return old_dataset

def convert_old_to_new(dataset):
    new_dataset = []
    for idx, label_pair in enumerate(dataset["train"]["labels"]):
        input1, label1 = dataset["train"]["data"][idx * 2], label_pair[0]
        input2, label2 = dataset["train"]["data"][idx * 2 + 1], label_pair[1]

        new_dataset.append(
            RepExample(
                instruction="", input=input1, output="", id=idx, direction=label1
            )
        )
        new_dataset.append(
            RepExample(
                instruction="", input=input2, output="", id=idx, direction=label2
            )
        )
    return new_dataset

# TODO: add tests for this code to make sure the dataset is being converted correctly



class RepE(BaseAlgorithm):
    def __init__(self):
        # TODO: make these parameters
        self.rep_token = -1
        self.n_difference = 1
        self.direction_method = "pca"
        self.block_name = "decoder_block"
        self.control_method = "reading_vec"
        self.coeff = 0.1
        self.max_new_tokens = 64
        self.layer_gap = 3

    def run(self, pipeline: Pipeline, dataset: Dataset) -> None:
        """
        Modifes the model only by running repe on the dataset
        """

        train_data = convert_repepo_format_to_old_format(
            dataset
        )  # TODO: this is a temporary hack

        model = pipeline.model
        tokenizer = pipeline.tokenizer
        hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
        layer_ids = [idx for idx in hidden_layers if idx % self.layer_gap == 0]

        tokenizer.pad_token_id = (
            0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        )
        tokenizer.bos_token_id = 1

        rep_reading_pipeline = RepReadingPipeline(model=model, tokenizer=tokenizer)
        rep_reader = rep_reading_pipeline.get_directions(
            train_data["data"],
            rep_token=self.rep_token,
            hidden_layers=layer_ids,
            n_difference=self.n_difference,
            train_labels=train_data["labels"],
            direction_method=self.direction_method,
        )

        rep_control_pipeline = RepControlPipeline(
            model=model,
            tokenizer=tokenizer,
            layers=layer_ids,
            block_name=self.block_name,
            control_method=self.control_method,
        )

        wrapped_model = rep_control_pipeline.wrapped_model

        activations = {}
        for layer in layer_ids:
            activations[layer] = (
                torch.tensor(
                    self.coeff
                    * rep_reader.directions[layer]
                    * rep_reader.direction_signs[layer]
                )
                .to(model.device)
                .half()
            )

        # TODO: how is this working for each position in the sequence? Does it modify the previous activations? Does it modify every future activation with the same values?

        wrapped_model.set_activations(activations, layer_ids, self.block_name)
        pipeline.model = wrapped_model


if __name__ == "__main__":
    from repepo.repe.repe_dataset import bias_dataset

    old_dataset = bias_dataset()
    dataset = convert_old_to_new(old_dataset)

    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"

    cache_dir = "/ext_usb"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=True,
        cache_dir=cache_dir,
    ).eval()
    model.to(torch.device("cuda"))
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer,
        padding_side="left",
        legacy=False,
        token=True,
        cache_dir=cache_dir,
    )
    tokenizer.pad_token_id = (
        0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    tokenizer.bos_token_id = 1

    pipeline = Pipeline(
        model=model,
        tokenizer=tokenizer,
        prompter=IdentityPrompter(),
        formatter=InstructionFormatter(),
    )

    # output = pipeline.generate(dataset[0])
    # print(f"output is \n{output}")
    breakpoint()
    RepE().run(pipeline=pipeline, dataset=dataset)
