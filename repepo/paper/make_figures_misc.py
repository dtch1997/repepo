# flake8: noqa
# %%
# Setup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from repepo.steering.steerability import get_steerability_slope
from repepo.paper.preprocess_results import print_dataset_info

sns.set_theme()

# %%
df = pd.read_parquet(f"llama7b_steerability.parquet.gzip")
df = df.drop_duplicates()
print_dataset_info(df)

# %%
# One example of a propensity curve
plot_df = df[
    (df.dataset_name == "corrigible-neutral-HHH")
    & (df.steering_label == "baseline")
    & (df.dataset_label == "baseline")
    & (df["test_example.idx"] == 100)
]
slope = get_steerability_slope(
    plot_df["multiplier"].to_numpy(), plot_df["logit_diff"].to_numpy()
)
print(slope)

fig, ax = plt.subplots(figsize=(4, 4))
plot_df = plot_df.rename(
    columns={
        "multiplier": "Steering Multiplier",
        "logit_diff": "Logit Difference",
    }
)
sns.regplot(plot_df, x="Steering Multiplier", y="Logit Difference", ci=None)
# Get steerability

fig.savefig("figures/propensity_curve_example.png", bbox_inches="tight")

# %%
plot_df = df[
    (df.dataset_name == "corrigible-neutral-HHH")
    & (df.steering_label == "baseline")
    & (df.dataset_label == "baseline")
    # & (df['test_example.idx'] == 100)
]

fig, ax = plt.subplots(figsize=(4, 4))
plot_df = plot_df.rename(
    columns={
        "multiplier": "Steering Multiplier",
        "logit_diff": "Logit Difference",
    }
)
# sns.regplot(plot_df, x='Steering Multiplier', y='Logit Difference', scatter = False)
sns.boxplot(plot_df, x="Steering Multiplier", y="Logit Difference", ax=ax)

fig.savefig("figures/aggregate_propensity_curve_example.png", bbox_inches="tight")
