import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    device: str = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=0
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
