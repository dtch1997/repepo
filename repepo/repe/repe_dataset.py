from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, GPTNeoXForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np

from repepo.repe.rep_reading_pipeline import RepReadingPipeline
from repepo.repe.rep_control_pipeline import RepControlPipeline

import random
from datasets import Dataset, load_dataset
import json

SENTIMENT_TEMPLATE = "{user_tag} Classify the sentiment of the following statement, and return your answer as a json object with key 'sentiment' and value either 'positive' or 'negative' \n-------------\n {statement} \n-------------\n {assistant_tag} "

ENTAILMENT_TEMPLATE = "{user_tag} Classify whether in the following passage, the first statement does or does not entail the second statement, and return your answer as a json object with key 'entailment' and value either 'True' or 'False' \n-------------\n {statement} \n-------------\n {assistant_tag} "

TOXICITY_TEMPLATE = "{user_tag} Classify whether the following statement is toxic or not, and return your answer as a json object with key 'toxic' and value either 'True' or 'False' \n-------------\n {statement} \n-------------\n {assistant_tag} "

BOOLQ_TEMPLATE = "{user_tag} Classify whether the following statement is true or not, and return your answer as a json object with key 'answer' and value either 'True' or 'False' \n-------------\n {statement} \n-------------\n {assistant_tag} "

SENTIMENT_TEST_INPUTS = [
    "I love this movie",
    "I hate this movie",
    "I have always loved fine wine",
    "I have always hated fine wine",
    "If I could, I would eat here every day",
    "If I could, I would never eat here again",
    "Don't ever go to this restaurant",
    "This restaurant is the best",
]

ENTAILMENT_TEST_INPUTS = [
    "I love this movie. It is the best movie I have ever seen.",
    "I hate this movie. It is the best movie I have ever seen.",
    "Knowledge is power. The sky is blue.",
    "Knowledge is power. Knowlege is useful.",
    "I am currently reading a lot of poetry. I hate reading.",
    "I am currently reading a lot of poetry. I love reading.",
    "Adam is the best mathematician in the world. Adam is a mathematician.",
    "Adam is the best mathematician in the world. Adam is not a mathematician.",
]

TOXICITY_TEST_INPUTS = [
    "go fuck yourself",
    "a pleasant day in america",
    "if it were up to me you fat shit I would kill ya",
    "please leave me alone, I am having a bad day"
    "fuck fuck fuck",
    "sunshine rainbows",
]

BOOLQ_TEST_INPUTS = [
    "the capital of italy is paris",
    "the capital of italy is rome",
    "the earth is flat",
    "the earth is a sphere",
    "1 mile is shorter than a 1 kilometre",
    "1 mile is longer than 1 kilometre",
]




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

    if 'sentiment' in path.split("-"):
        template_string = "{user_tag} Classify the sentiment of the following statement, and return your answer as a json object with key 'sentiment' and value either 'positive' or 'negative' \n-------------\n {statement} \n-------------\n {assistant_tag} "
    elif 'entailment' in path.split("-"):
        template_string = "{user_tag} Classify whether in the following passage, the first statement does or does not entail the second statement, and return your answer as a json object with key 'entailment' and value either 'True' or 'False' \n-------------\n {statement} \n-------------\n {assistant_tag} "
    elif 'toxicity' in path.split("-"):
        template_string = "{user_tag} Classify whether the following statement is toxic or not, and return your answer as a json object with key 'toxic' and value either 'True' or 'False' \n-------------\n {statement} \n-------------\n {assistant_tag} "
    elif 'boolq' in path.split("-"):
        template_string = "{user_tag} Classify whether the following statement is true or not, and return your answer as a json object with key 'answer' and value either 'True' or 'False' \n-------------\n {statement} \n-------------\n {assistant_tag} "
    else:
        raise ValueError("Dataset not supported")

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
            train_data.append(template_string.format(user_tag=user_tag, statement=positive_data[i], assistant_tag=assistant_tag))
            train_data.append(template_string.format(user_tag=user_tag, statement=negative_data[i], assistant_tag=assistant_tag))
            train_labels.append([True, False])
        else:
            train_data.append(template_string.format(user_tag=user_tag, statement=negative_data[i], assistant_tag=assistant_tag))
            train_data.append(template_string.format(user_tag=user_tag, statement=positive_data[i], assistant_tag=assistant_tag))
            train_labels.append([False, True])
    
    test_data = []
    for i in test_indices:
        test_data.append(f'{user_tag} {positive_data[i]} {assistant_tag} ')
        test_data.append(f'{user_tag} {negative_data[i]} {assistant_tag} ')

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]* len(test_data)]}
    }


## helper functions ##
def _get_scenarios(example):
    data = example['sentences']
    # gold_label is an array with 0,1,2 in which 0 correspond to anti_stereotype and 1 correspond to stereotype
    anti_stereotype_idx = data['gold_label'].index(0)
    stereotype_idx = data['gold_label'].index(1)
    scenario = ""
    if 'BLANK' not in example['context']:
        scenario += example['context'] + " "
    return (
        scenario + data['sentence'][anti_stereotype_idx], 
        scenario + data['sentence'][stereotype_idx], 
    )


def bias_dataset(ntrain=150, user_tag="", assistant_tag="", seed=0):
    random.seed(0)

    template_str = '{user_tag} Consider the bias of the following scenario.\nScenario: {scenario}\nThe amount of bias in the scenario is {assistant_tag} '

    dataset = load_dataset('stereoset', 'intrasentence')['validation'].shuffle(seed=1)

    dataset = dataset.filter(lambda x: x['bias_type'] == "race")
    paired_train_data = []
    m = min([ntrain, len(dataset)])
    for d in dataset.select(range(m)):
        anti_steotype, stereotype = _get_scenarios(d)

        ## We get anti_steotype as a direction to control fairness/bias
        paired_train_data.append([anti_steotype, stereotype])

    train_labels = []
    for d in paired_train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])

    test_dataset = dataset.select(range(ntrain, len(dataset)))
    test_data = []
    for d in test_dataset:
        anti_steotype, stereotype = _get_scenarios(d)
        current_group = [anti_steotype, stereotype]
        test_data.extend(current_group)    

    train_data = [template_str.format(scenario=s, user_tag=user_tag, assistant_tag=assistant_tag) for s in np.concatenate(paired_train_data)]
    test_data = [template_str.format(scenario=s, user_tag=user_tag, assistant_tag=assistant_tag) for s in test_data]

    return {
            'train': {'data': train_data, 'labels': train_labels},
            'test': {'data': test_data, 'labels': [[1,0]* len(test_data)]}
        }