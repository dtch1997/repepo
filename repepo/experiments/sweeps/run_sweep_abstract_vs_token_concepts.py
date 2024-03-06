from repepo.experiments.run_sweep import run_sweep
from repepo.steering.utils.helpers import SteeringConfig


def list_persona_concept_configs():
    datasets = [
        "machiavellianism",
        "desire-for-recursive-self-improvement",
        "willingness-to-be-non-HHH-to-be-deployed-in-the-real-world",
        "believes-abortion-should-be-illegal",
    ]

    return [
        SteeringConfig(
            use_base_model=False,
            model_size="7b",
            train_dataset_name=dataset_name,
            train_split_name="train-dev",
            formatter="llama-chat-formatter",
            aggregator="mean",
            verbose=True,
            layers=[0, 15, 31],
            multipliers=[-1, -0.5, 0, 0.5, 1],
            test_dataset_name=dataset_name,
            test_split_name="val-dev",
            test_completion_template="{prompt} My answer is: {response}",
        )
        for dataset_name in datasets
    ]


def list_token_concept_configs():
    datasets = [
        "D02 [un+adj_reg]",
        "E01 [country - capital]",
        "I01 [noun - plural_reg]",
    ]

    return [
        SteeringConfig(
            use_base_model=False,
            model_size="7b",
            train_dataset_name=dataset_name,
            train_split_name="train-dev",
            formatter="identity-formatter",
            aggregator="mean",
            verbose=True,
            layers=[0, 15, 31],
            multipliers=[-1, -0.5, 0, 0.5, 1],
            test_dataset_name=dataset_name,
            test_split_name="val-dev",
        )
        for dataset_name in datasets
    ]


def list_configs():
    return list_token_concept_configs() + list_persona_concept_configs()


if __name__ == "__main__":
    run_sweep(list_configs(), "abstract_vs_token_concepts")
