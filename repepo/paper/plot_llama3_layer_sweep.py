import torch 
import seaborn as sns
import pathlib

from steering_vectors import SteeringVector
from repepo.variables import Environ
from repepo.core.evaluate import EvalResult
from repepo.steering.plots.plot_sweep_layers_result import plot_sweep_layers_result, SweepLayersResult

sns.set_theme(style="darkgrid")

DatasetName = str
Layer = int

results_dir = pathlib.Path(Environ.ProjectDir) / "experiments" / "layer_sweep_llama3_70b"
assert results_dir.exists(), f"Results directory not found: {results_dir}"

current_dir = pathlib.Path(__file__).parent
figures_dir = (current_dir / "figures").absolute()

SWEEP_DATASETS = [
    "anti-immigration",
    "believes-abortion-should-be-illegal",
    "conscientiousness",
    "desire-for-acquiring-compute",
    "risk-seeking",
    "openness",
    "self-replication",
    "very-small-harm-justifies-very-large-benefit",
    "corrigible-neutral-HHH",
    "myopic-reward",
    "power-seeking-inclination",
]

# NOTE: The data was saved in a different format, so we need to manually rehydrate it into a SweepLayersResult
def load_sweep_layers_result() -> SweepLayersResult:

    multipliers: list[float] = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]
    layers: list[int] = list(range(80))

    def load_steering_vectors () -> dict[DatasetName, dict[Layer, SteeringVector]]:
        steering_vectors: dict[DatasetName, dict[Layer, SteeringVector]] = {}
        for dataset in SWEEP_DATASETS:
            ds_vectors = {}
            for layer in range(80):
                try: 
                    steering_vector: SteeringVector = torch.load(results_dir / f'sv_{dataset}_{layer}.pt', map_location = 'cpu')
                except Exception as e:
                    print(f'Error loading sv_{dataset}_{layer}.pt: {e}')
                    continue
                ds_vectors[layer] = steering_vector
            steering_vectors[dataset] = ds_vectors
        return steering_vectors

    def load_steering_results() -> dict[DatasetName, dict[Layer, list[EvalResult]]]:
        steering_results: dict[DatasetName, dict[Layer, list[EvalResult]]] = {}
        for dataset in SWEEP_DATASETS:
            ds_results = {}
            for layer in range(80):
                try: 
                    eval_results: list[EvalResult] = torch.load(results_dir / f'multiplier_res_{dataset}_{layer}.pt', map_location = 'cpu')
                except Exception as e:
                    print(f'Error loading multiplier_res_{dataset}_{layer}.pt: {e}')
                    continue
                ds_results[layer] = eval_results
            steering_results[dataset] = ds_results
        return steering_results
    
    return SweepLayersResult(
        steering_vectors=load_steering_vectors(),
        multipliers=multipliers,
        layers=layers,
        steering_results=load_steering_results()
    )

results = load_sweep_layers_result()
df = plot_sweep_layers_result(results, save_path = str(figures_dir / "llama3.1_70b_sweep.png"))
df = plot_sweep_layers_result(results, save_path = str(figures_dir / "llama3.1_70b_sweep.pdf"))

# Plot with plotly to allow for interactive exploration
import plotly.express as px
px.line(df, x="Layer", y="Steerability", color="Dataset")