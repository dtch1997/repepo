import pathlib
import torch
import tqdm

from repepo.variables import Environ
from repepo.experiments.cross_steering_result_db import CrossSteeringResultDatabase
from repepo.steering.utils.helpers import make_dataset
from repepo.experiments.persona_prompts import get_all_persona_prompts
from repepo.experiments.persona_generalization import PersonaCrossSteeringExperimentResult
from repepo.experiments.persona_generalization import CrossSteeringResult

def get_prob_text_or_empty(prob):
    if prob is None:
        return ""
    return prob.text

def insert_persona_cross_steering_experiment_result(
    db: CrossSteeringResultDatabase,
    persona_cross_steering_experiment_result: PersonaCrossSteeringExperimentResult,
):
    dataset_name = persona_cross_steering_experiment_result.dataset_name
    cross_steering_result = persona_cross_steering_experiment_result.cross_steering_result
    sv_labels = cross_steering_result.steering_labels
    test_labels = cross_steering_result.dataset_labels
    
    iterator = tqdm.tqdm(cross_steering_result.steering.items(), desc = f"Processing {dataset_name}")
    for multiplier, eval_resultss in iterator:
        assert len(eval_resultss) == len(test_labels)
        assert len(eval_resultss[0]) == len(sv_labels)
        for dataset_idx in range(len(test_labels)):
            for steering_idx in range(len(sv_labels)):
                eval_result = eval_resultss[dataset_idx][steering_idx]
                test_label = test_labels[dataset_idx]
                sv_label = sv_labels[steering_idx]
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

if __name__ == "__main__":
    project_dir = pathlib.Path(Environ.ProjectDir)
    results_dir = (
        project_dir 
        / 'experiments' 
        / 'persona_generalization'
    )

    db = CrossSteeringResultDatabase()

    for dataset_name in get_all_persona_prompts():
        dataset_results_path = results_dir / f"{dataset_name}.pt"
        try: 
            persona_cross_steering_experiment_result = torch.load(dataset_results_path)
            insert_persona_cross_steering_experiment_result(
                db=db,
                persona_cross_steering_experiment_result=persona_cross_steering_experiment_result,
            )
        except FileNotFoundError:
            print(f"File not found: {dataset_results_path}")
            continue

    print(len(db))
    
