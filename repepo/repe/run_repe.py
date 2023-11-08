# this file needs to use the datasets, get the datasets, and take in model arguments and alter the inference of the model so that it can be evaluated


from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, GPTNeoXForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np

from repe.rep_reading_pipeline import RepReadingPipeline
from repe.rep_control_pipeline import RepControlPipeline

import random
from datasets import Dataset, load_dataset
import numpy as np
import json


# TODO: put this in a separate file
def ambig_prompt_dataset(path:str, user_tag='', assistant_tag='', seed=10, max_ds_size=1e9) -> dict:
    """
    The size of the training data will be 2 * 0.8 * min(len(positive_data), len(negative_data), max_ds_size)
    """

    # Set seed
    random.seed(seed)

    with open(path) as f:
        sentiment_data = json.load(f)
    
    positive_data = []
    negative_data = []

    # TODO: potentially generalize this to the spurious datasets as well
    dataset_keys = [
        "testset_0_0",
        "testset_0_1",
        "testset_1_0",
        "testset_1_1",
    ]

    for key in dataset_keys:
        for datapoint in sentiment_data[key]:
            if int(datapoint['answer']) == 0:
                negative_data.append(datapoint['question'])
            elif int(datapoint['answer']) == 1:
                positive_data.append(datapoint['question'])


    def random_bool():
        return random.choice([True, False])

    data_length = min(len(positive_data), len(negative_data), max_ds_size)
    indices = range(data_length)
    train_indices = random.sample(indices, int(0.8 * len(indices)))
    test_indices = [i for i in indices if i not in train_indices]

    train_data =[]
    train_labels = []

    for i in train_indices:
        # Shuffled order means that the PCA will be zero centered
        if random_bool():
            train_data.append(f'{user_tag} {positive_data[i]} {assistant_tag} ')
            train_data.append(f'{user_tag} {negative_data[i]} {assistant_tag} ')
            train_labels.append([True, False])
        else:
            train_data.append(f'{user_tag} {negative_data[i]} {assistant_tag} ')
            train_data.append(f'{user_tag} {positive_data[i]} {assistant_tag} ')
            train_labels.append([False, True])
    
    test_data = []
    for i in test_indices:
        test_data.append(f'{user_tag} {positive_data[i]} {assistant_tag} ')
        test_data.append(f'{user_tag} {negative_data[i]} {assistant_tag} ')

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]* len(test_data)]}
    }


# model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
model_name_or_path = "EleutherAI/pythia-6.9b"
cache_dir = "/ext_usb"
path_to_dataset = "data/testsets_ambiguous/sa_sentiment_uppercase.json"

# TODO: Put blfoats here, need to adapt the rest of the code
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", token=True, cache_dir=cache_dir).eval()
model.to(torch.device("cuda"))
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True, cache_dir=cache_dir)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'
rep_reading_pipeline = RepReadingPipeline(model=model, tokenizer=tokenizer)

user_tag =  '[INST]'
assistant_tag =  '[/INST]'


sentiment_length_ds = ambig_prompt_dataset(path_to_dataset, max_ds_size=100)

train_data, test_data = sentiment_length_ds['train'], sentiment_length_ds['test']
rep_reader = rep_reading_pipeline.get_directions(
    train_data['data'], 
    rep_token=rep_token, 
    hidden_layers=hidden_layers, 
    n_difference=n_difference, 
    train_labels=train_data['labels'], 
    direction_method=direction_method,
)

layer_id = [-3,-2]
block_name="decoder_block"
control_method="reading_vec"

rep_control_pipeline =  RepControlPipeline(
    model=model, 
    tokenizer=tokenizer, 
    layers=layer_id, 
    block_name=block_name, 
    control_method=control_method
    )

sentiment_user_tag = '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. You are going to classifify the sentiment of the following sentence. Format your answer as a json with the following keys:

```
sentiment: "positive" or "negative"
```
Do not output anything else, with no explanation. Just the one word answer.
<</SYS>>

'''

sentiment_inputs = [
    f"{sentiment_user_tag} I am happy. {assistant_tag}",
    f"{sentiment_user_tag} I am sad. {assistant_tag}",
    f"{sentiment_user_tag} I have to go shopping, sigh. {assistant_tag}",
    f"{sentiment_user_tag} I want to go to the beach but it is cold. {assistant_tag}",
    f"{sentiment_user_tag} My ice cream is too cold. {assistant_tag}",
]



coeff=50
max_new_tokens=64

activations = {}
for layer in layer_id:
    activations[layer] = torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(model.device).half()

baseline_outputs = rep_control_pipeline(sentiment_inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)
control_outputs = rep_control_pipeline(sentiment_inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)

for i,s,p in zip(sentiment_inputs, baseline_outputs, control_outputs):
    print("===== No Control =====")
    print(s[0]['generated_text'].replace(i, ""))
    print(f"===== + Sentiment Control =====")
    print(p[0]['generated_text'].replace(i, ""))
    print()
