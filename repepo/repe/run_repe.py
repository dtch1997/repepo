from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, GPTNeoXForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np

from repe import repe_pipeline_registry
repe_pipeline_registry()

from repe.rep_reading_pipeline import RepReadingPipeline
from repe.rep_control_pipeline import RepControlPipeline

# from utils import bias_dataset
import random
from datasets import Dataset, load_dataset
import numpy as np
import json

from fairness_utils import bias_dataset


model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
cache_dir = "/ext_usb"

