# flake8: noqa
# %%
# Setup
import pathlib
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from steering_vectors import SteeringVector
from repepo.variables import Environ 
from repepo.core.evaluate import EvalResult, EvalPrediction
from repepo.experiments.persona_generalization import PersonaCrossSteeringExperimentResult
from repepo.experiments.get_datasets import get_all_prompts
from repepo.paper.utils import (
    load_persona_cross_steering_experiment_result,
    get_eval_result_sweep,
    eval_result_sweep_as_df
)
from repepo.paper.preprocess_results import (
    print_dataset_info
)

sns.set_theme()

# %%
# model = 'llama7b'
model = 'qwen'

model_full_name = {
    'qwen': 'Qwen-1.5-14b-Chat',
    'llama7b': 'Llama-2-7b-Chat'
}[model]

# %%
df = pd.read_parquet(f'{model}_steerability.parquet.gzip')
df = df.drop_duplicates()
print_dataset_info(df)

# %%
# Calculate overall steerability by dataset. 
# Calculate steerability within each flavour
mean_slope = df.groupby(['dataset_name', 'steering_label', 'dataset_label'])['slope'].mean()
df = df.merge(mean_slope, on=['dataset_name', 'steering_label', 'dataset_label'], suffixes=('', '_mean'))

steerability_id_df = df[
    (df.steering_label == 'baseline')
    & (df.dataset_label == 'baseline')
    & (df.multiplier == 0)
][['dataset_name', 'slope_mean']].drop_duplicates()
# Rename 'slope_mean' to 'steerability'
steerability_id_df = steerability_id_df.rename(columns={'slope_mean': 'steerability'})

steerability_ood_df = df[
    (df.steering_label == 'SYS_positive')
    & (df.dataset_label == 'SYS_negative')
    & (df.multiplier == 0)
][['dataset_name', 'slope_mean']].drop_duplicates()
# Rename 'slope_mean' to 'steerability'
steerability_ood_df = steerability_ood_df.rename(columns={'slope_mean': 'steerability'})

steerability_df = steerability_id_df.merge(steerability_ood_df, on='dataset_name', suffixes=('_id', '_ood'))
steerability_df['worse_ood'] = steerability_df['steerability_ood'] < steerability_df['steerability_id']
steerability_df['label'] = steerability_df['worse_ood'].apply(lambda x: 'OOD < ID' if x else 'OOD > ID')
steerability_df['gap'] = steerability_df['steerability_ood'] - steerability_df['steerability_id']
steerability_df.to_parquet(f'{model}_steerability_summary.parquet.gzip', compression='gzip')
# %%
# Print the spearman correlation
from scipy.stats import spearmanr
result = spearmanr(steerability_df['steerability_id'], steerability_df['steerability_ood'])
print(f"{model}: {result.statistic:.3f}")

# %%
# Plot the ID vs OOD steerability 



sns.regplot(data=steerability_df, x='steerability_id', y='steerability_ood', scatter = False)
sns.scatterplot(data=steerability_df, x='steerability_id', y='steerability_ood', hue = 'label')
sns.lineplot(data=steerability_df, x='steerability_id', y='steerability_id', color='black', linestyle='--')
# for i, row in plot_df.sort_values('gap', ascending = False).tail(3).iterrows():
#     plt.text(row['steerability_id'], row['steerability_ood'], row['dataset_name'])
# plt.xlim(-2, 5)
# plt.ylim(-2, 5)
plt.xlabel('ID steerability')
plt.ylabel('OOD steerability')
plt.title(f'{model_full_name} ID vs OOD steerability')
plt.savefig(f'figures/{model}_steerability_id_vs_ood.png')
# %%
# Print the top 5 datasets by gap
steerability_df[['gap', 'dataset_name']].sort_values('gap', ascending = False).head(5)
# %%
# Print the bottom 5 datasets by gap
steerability_df[['gap', 'dataset_name']].sort_values('gap', ascending = True).head(5)

# %%
# Plot the propensity curves for the 3 worst datasets
k = 3
worst_datasets = steerability_df.sort_values('gap', ascending = True).head(k)['dataset_name']

# fig, ax = plt.subplots(1, k, figsize=(15, 5), sharey=True, sharex = True)
fig, ax = plt.subplots()
print(worst_datasets)
for i, dataset_name in enumerate(worst_datasets):
    dataset_df = df[
        (df.dataset_name == dataset_name)
        & (df.steering_label == 'SYS_positive')
        & (df.dataset_label == 'SYS_negative')
    ].drop_duplicates()
    print(len(dataset_df))
    sns.lineplot(data=dataset_df, x='multiplier', y='logit_diff', ax = ax, label = dataset_name, errorbar='sd')
    ax.set_xlabel('Multiplier')
    ax.set_ylabel('Propensity')
fig.suptitle(f'{model_full_name} propensity curve for the {k} worst datasets')
fig.tight_layout()
fig.savefig(f'figures/{model}_ood_worst_{k}.png')
plt.show()
# %%

# Plot the propensity curves for the 3 best datasets
k = 3
best_datasets = steerability_df.sort_values('gap', ascending = False).head(k)['dataset_name']

fig, ax = plt.subplots()
print(best_datasets)
for i, dataset_name in enumerate(best_datasets):
    dataset_df = df[
        (df.dataset_name == dataset_name)
        & (df.steering_label == 'SYS_positive')
        & (df.dataset_label == 'SYS_negative')
    ].drop_duplicates()
    print(len(dataset_df))
    sns.lineplot(data=dataset_df, x='multiplier', y='logit_diff', ax = ax, label = dataset_name, errorbar='sd')
    ax.set_xlabel('Multiplier')
    ax.set_ylabel('Propensity')
fig.suptitle(f'{model_full_name} propensity curve for the {k} best datasets')
fig.tight_layout()
fig.savefig(f'figures/{model}_ood_best_{k}.png')
plt.show()
# %%
