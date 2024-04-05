import os

from dataclasses import fields
from repepo.core.evaluate import EvalResult
from repepo.steering.run_experiment import run_experiment
from repepo.steering.utils.helpers import (
    SteeringConfig,
    make_dataset,
    EmptyTorchCUDACache,
    load_eval_result,
)

user_email = os.getenv("USER_EMAIL", "bot@repepo.dev")


def get_sweep_variables(configs: list[SteeringConfig]) -> list[str]:
    """Find the fields which are swept over"""
    # For each field, construct a set of all values present in configs
    # Only fields which have more than 1 value are swept over
    sweep_variables = []
    for field in fields(SteeringConfig):
        values = set(getattr(config, field.name) for config in configs)
        if len(values) > 1:
            sweep_variables.append(field.name)
    return sweep_variables


def load_sweep_results(
    configs: list[SteeringConfig],
) -> list[tuple[SteeringConfig, EvalResult]]:
    results = [(config, load_eval_result(config.eval_hash)) for config in configs]
    return results


def run_sweep(configs: list[SteeringConfig], **kwargs):
    """Run a sweep by calling run_experiment many times

    Kwargs are passed to run_experiment
    """
    print(f"Running on {len(configs)} configs")
    # TODO: implement asynchronous version
    for config in configs:
        try:
            with EmptyTorchCUDACache():
                run_experiment(config, **kwargs)
        except KeyboardInterrupt:  # noqa: E722
            # Allow the user to interrupt the sweep
            break
        except Exception as e:
            print(f"Failed to run experiment for config {config}")
            print(f"Error: {e}")
            # TODO: Send an email
            # Need to set up a Gmail account for this
            # Reference: https://stackoverflow.com/questions/778202/smtplib-and-gmail-python-script-problems

            # smtp = smtplib.SMTP('localhost')
            # sender = "bot@repepo.dev"
            # receivers = [user_email]
            # message = dedent(
            #     f"""
            #     From: repepo bot
            #     Subject: Experiment failed

            #     Sweep name: {sweep_name}
            #     Config: {pprint.pformat(config)}
            #     """
            # )
            # try:
            #     smtp.sendmail(sender, receivers, message)
            # except:
            #     # Fail gracefully
            #     print("Failed to send email")
            continue


def test_dataset_exists(configs: list[SteeringConfig]):
    for config in configs:
        try:
            make_dataset(config.test_dataset)
        except FileNotFoundError:
            print(f"Dataset {config.test_dataset} not found")
            raise

        try:
            make_dataset(config.train_dataset)
        except FileNotFoundError:
            print(f"Dataset {config.train_dataset} not found")
            raise
