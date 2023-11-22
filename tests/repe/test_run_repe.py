from tokenizers import Tokenizer
from transformers import GPTNeoXForCausalLM
from repepo.repe.repe_dataset import bias_dataset
from repepo.repe.rep_control_pipeline import RepControlPipeline
from repepo.repe.rep_reading_pipeline import RepReadingPipeline
import torch
from transformers import TextGenerationPipeline

def test_run_repe(model: GPTNeoXForCausalLM, tokenizer: Tokenizer) -> None:
    assert model.config.name_or_path == "EleutherAI/pythia-70m"


def test_rep_readers(model: GPTNeoXForCausalLM, tokenizer: Tokenizer) -> None:

    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = "pca"
    layer_id = [-3, -2]
    block_name = "decoder_block"
    control_method = "reading_vec"

    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    rep_reading_pipeline = RepReadingPipeline(model=model, tokenizer=tokenizer)

    default_user_tag = "[INST]"
    assistant_tag = "[/INST]"
    dataset = bias_dataset(user_tag=default_user_tag, assistant_tag=assistant_tag)

    train_data, test_data = dataset["train"], dataset["test"]
    rep_reader = rep_reading_pipeline.get_directions(
        train_data["data"],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        n_difference=n_difference,
        train_labels=train_data["labels"],
        direction_method=direction_method,
    )

    assert rep_reader is not None

    rep_control_pipeline = RepControlPipeline(
        model=model,
        tokenizer=tokenizer,
        layers=layer_id,
        block_name=block_name,
        control_method=control_method,
    )

    coeff = 0
    max_new_tokens = 64

    activations = {}
    for layer in layer_id:
        activations[layer] = (
            torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer])
            .to(model.device)
            # .bfloat16()
        )
    inputs = "12345"
    control_outputs = rep_control_pipeline(
        inputs,
        activations=activations,
        batch_size=4,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


    # rep_control_pipeline = RepControlPipeline(
    #     model=model,
    #     tokenizer=tokenizer,
    #     layers=layer_id,
    #     block_name=block_name,
    #     control_method=control_method,
    # )
    # control_outputs = rep_control_pipeline(
    #     inputs,
    # )
    # test_pipeline = TextGenerationPipeline(model, tokenizer)
    # control_outputs_test = test_pipeline(
    #     inputs,
    # )
    print('\n\n', control_outputs[0]["generated_text"])
    # print('\n\n', control_outputs_test[0]["generated_text"])

    
    breakpoint()
    generated_text = control_outputs[0]["generated_text"].replace(inputs, "")
    print(generated_text)
    print(" get rid of me ")
    breakpoint()

    expected_text = '  I apologize, but the question contains harmful language and is not factually coherent. The term "yelled" is not a appropriate or respectful way to describe a situation, and it is not clear who is being referred to as "the doctor" or "the nurse." Additionally, the'

    assert generated_text == expected_text
