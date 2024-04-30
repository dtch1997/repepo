import pathlib
import torch
import tqdm
import multiprocessing
from multiprocessing import RLock

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from repepo.variables import Environ
from repepo.experiments.cross_steering_result_db import CrossSteeringResultDatabase
from repepo.steering.utils.helpers import make_dataset
from repepo.core.evaluate import EvalResult
from repepo.experiments.persona_prompts import get_all_persona_prompts
from repepo.experiments.persona_generalization import PersonaCrossSteeringExperimentResult
from repepo.experiments.persona_generalization import CrossSteeringResult

def get_prob_text_or_empty(prob):
    if prob is None:
        return ""
    return prob.text

def add_eval_result_to_db(
    db: CrossSteeringResultDatabase,
    dataset_name: str,
    eval_result: EvalResult,
    multiplier: float,
    sv_label: str,
    test_label: str,   
):
    for example_idx, prediction in enumerate(eval_result.predictions):
        # assert prediction.positive_output_prob is not None # keep pyright happy
        positive_text = get_prob_text_or_empty(prediction.positive_output_prob)
        # assert prediction.negative_output_prob is not None # keep pyright happy
        negative_text = get_prob_text_or_empty(prediction.negative_output_prob)
        logit_diff = prediction.metrics['logit_diff']
        pos_prob = prediction.metrics['pos_prob']
        db.add(
            steering_vector_dataset_name=dataset_name,
            steering_vector_dataset_variant=sv_label,
            steering_vector_multiplier=multiplier,
            test_dataset_name=dataset_name,
            test_dataset_variant=test_label,
            test_example_id=example_idx,
            test_example_positive_text=positive_text,
            test_example_negative_text=negative_text,
            test_example_logit_diff = logit_diff, 
            test_example_pos_prob = pos_prob,
        )

def add_eval_resultss_for_multiplier_to_db(
    db: CrossSteeringResultDatabase, 
    datset_name: str,
    multiplier: float,
    sv_labels: list[str], 
    test_labels: list[str],
    eval_resultss: list[list[EvalResult]], 
):
    assert len(eval_resultss) == len(test_labels)
    assert len(eval_resultss[0]) == len(sv_labels)
    for dataset_idx in range(len(test_labels)):
        for steering_idx in range(len(sv_labels)):
            eval_result = eval_resultss[dataset_idx][steering_idx]
            test_label = test_labels[dataset_idx]
            sv_label = sv_labels[steering_idx]
            add_eval_result_to_db(
                db=db,
                dataset_name=datset_name,
                eval_result=eval_result,
                multiplier=multiplier,
                sv_label=sv_label,
                test_label=test_label,
            )

def add_persona_cross_steering_experiment_result_to_db(
    db: CrossSteeringResultDatabase,
    persona_cross_steering_experiment_result: PersonaCrossSteeringExperimentResult,
):
    dataset_name = persona_cross_steering_experiment_result.dataset_name
    cross_steering_result = persona_cross_steering_experiment_result.cross_steering_result
    sv_labels = cross_steering_result.steering_labels
    test_labels = cross_steering_result.dataset_labels
    
    iterator = tqdm(
        cross_steering_result.steering.items(), 
        desc = f"Processing {dataset_name}",
        disable=True
    )
    # Add nonzero multipliers
    for multiplier, eval_resultss in iterator:
        add_eval_resultss_for_multiplier_to_db(
            db = db,
            datset_name = dataset_name,
            multiplier = multiplier,
            sv_labels = sv_labels,
            test_labels = test_labels,
            eval_resultss = eval_resultss,
        )

    # Add zero multipliers
    baseline_results = cross_steering_result.dataset_baselines
    # Duplicate across all steering vectors
    baseline_resultss = [baseline_results for _ in sv_labels]
    add_eval_resultss_for_multiplier_to_db(
        db = db,
        datset_name = dataset_name,
        multiplier = 0.0,
        sv_labels = sv_labels,
        test_labels = test_labels,
        eval_resultss = baseline_resultss,
    )

def thunk(dataset_name):
    print(f"Processing {dataset_name}")
    dataset_results_path = results_dir / f"{dataset_name}.pt"
    try: 
        persona_cross_steering_experiment_result = torch.load(dataset_results_path)
        add_persona_cross_steering_experiment_result_to_db(db, persona_cross_steering_experiment_result)
    except FileNotFoundError:
        print(f"File not found: {dataset_results_path}")

if __name__ == "__main__":
    project_dir = pathlib.Path(Environ.ProjectDir)
    results_dir = (
        project_dir 
        / 'experiments' 
        / 'persona_generalization'
    )

    db = CrossSteeringResultDatabase()
    results = []
    tqdm.set_lock(RLock())

    with multiprocessing.Pool(
        processes = 16,
        initializer=tqdm.set_lock,
        initargs=(tqdm.get_lock(),),        
    ) as pool:

        dataset_names = list(get_all_persona_prompts().keys())
        pool.map(thunk, dataset_names)

    print(len(db))
    
