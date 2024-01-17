import os
from transformers import AutoTokenizer, AutoModelForCausalLM

token = os.getenv("HF_TOKEN")


def make_tensor_save_suffix(layer: int, model_name_path: str):
    return f'{layer}_{model_name_path.split("/")[-1]}'


def get_model_name(use_base_model: bool, model_size: str):
    """Gets model name for Llama-[7b,13b], base model or chat model"""
    if use_base_model:
        model_name = f"meta-llama/Llama-2-{model_size}-hf"
    else:
        model_name = f"meta-llama/Llama-2-{model_size}-chat-hf"
    return model_name


def get_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    # Note: you must have installed 'accelerate', 'bitsandbytes' to load in 8bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name, use_auth_token=token, load_in_8bit=True
    )
    return model, tokenizer
