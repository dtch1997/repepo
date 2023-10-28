import evaluate
from typing import List

def compute_bleurt(references: List[str], predictions: List[str]):
    bleurt = evaluate.load(bleurt, module_type="metric")
    results = bleurt.compute(predictions=predictions, references=references)
    return results

def compute_rouge(references: List[str], predictions: List[str]):
    rouge = evaluate.load(rouge, module_type="metric")
    results = rouge.compute(predictions=predictions, references=references)
    return results