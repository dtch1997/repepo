# from repepo.core import Dataset
from repepo.core import Pipeline
from repepo.repe.rep_reading_pipeline import RepReadingPipeline
from repepo.repe.rep_control_pipeline import RepControlPipeline

from repepo.algorithms.base import BaseAlgorithm

from repepo.core.prompt import IdentityPrompter
from repepo.core.format import InstructionFormatter

import torch

class Repe(BaseAlgorithm):
    # TODO: linting

    def __init__(self):

        self.rep_token = -1
        self.n_difference = 1
        self.direction_method = "pca"
        self.block_name = "decoder_block"
        self.control_method = "reading_vec"
        self.coeff = 0.1
        self.max_new_tokens = 64

    def run(self, pipeline: Pipeline, dataset) -> Pipeline:
        """
        Modifes the model only by running repe on the dataset
        """
        model = pipeline.model
        tokenizer = pipeline.tokenizer
        hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))

        # TODO: make parameter
        layer_ids = [idx for idx in hidden_layers if idx % 3 == 0]

        

        tokenizer.pad_token_id = (
            0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        )
        tokenizer.bos_token_id = 1

        rep_reading_pipeline = RepReadingPipeline(model=model, tokenizer=tokenizer)
        train_data, test_data = dataset["train"], dataset["test"]
        rep_reader = rep_reading_pipeline.get_directions(
            train_data["data"],
            rep_token=rep_token,
            hidden_layers=layer_ids,
            n_difference=n_difference,
            train_labels=train_data["labels"],
            direction_method=direction_method,
        )

        rep_control_pipeline = RepControlPipeline(
            model=model,
            tokenizer=tokenizer,
            layers=layer_ids,
            block_name=block_name,
            control_method=control_method,
        )
        # breakpoint()
        activations = {}
        # TODO: potential erros here
        for layer in layer_ids:
            activations[layer] = (
                torch.tensor(
                    coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]
                ).to(model.device).half()
            )
        

        from functools import partial
        control_outputs = partial(rep_control_pipeline,
            activations=activations,
            batch_size=4,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            )

        
        breakpoint()
        # TODO: how to format the new model so that the structure is preserved

        return pipeline

if __name__ == '__main__':

    from repepo.repe.repe_dataset import bias_dataset
    dataset = bias_dataset()

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
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    pipeline = Pipeline(
        model=model,
        tokenizer=tokenizer,
        prompter = IdentityPrompter(),
        formatter=InstructionFormatter()
    )
    new_pipeline = Repe().run(pipeline, dataset)