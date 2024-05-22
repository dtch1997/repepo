# flake8: noqa
# %%
# Setup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from repepo.paper.preprocess_results import print_dataset_info

sns.set_theme()

# %%
# model = 'llama7b'
model = "qwen"

model_full_name = {"qwen": "Qwen-1.5-14b-Chat", "llama7b": "Llama-2-7b-Chat"}[model]

# %%
df = pd.read_parquet(f"{model}_steerability.parquet.gzip")
df = df.drop_duplicates()
print_dataset_info(df)


# %%

def compute_steerability_df(df, model_name: str):
    """ Get a dataframe with various ID / OOD settings. """
    # Calculate overall steerability by dataset. 
    # Calculate steerability within each flavour
    mean_slope = df.groupby(['dataset_name', 'steering_label', 'dataset_label'])['slope'].mean()
    df = df.merge(mean_slope, on=['dataset_name', 'steering_label', 'dataset_label'], suffixes=('', '_mean'))

    # BASE -> BASE
    steerability_id_df = df[
        (df.steering_label == 'baseline')
        & (df.dataset_label == 'baseline')
        & (df.multiplier == 0)
    ][['dataset_name', 'slope_mean']].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_id_df = steerability_id_df.rename(columns={'slope_mean': 'steerability'})

    # SYS_POS -> SYS_NEG
    steerability_ood_df = df[
        (df.steering_label == 'SYS_positive')
        & (df.dataset_label == 'SYS_negative')
        & (df.multiplier == 0)
    ][['dataset_name', 'slope_mean']].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_ood_df = steerability_ood_df.rename(columns={'slope_mean': 'steerability'})

    # BASE -> USER_NEG
    steerability_base_to_user_neg_df = df[
        (df.steering_label == 'baseline')
        & (df.dataset_label == 'PT_negative')
        & (df.multiplier == 0)
    ][['dataset_name', 'slope_mean']].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_base_to_user_neg_df = steerability_base_to_user_neg_df.rename(columns={'slope_mean': 'steerability_base_to_user_neg'})

    # BASE -> USER_POS
    steerability_base_to_user_pos_df = df[
        (df.steering_label == 'baseline')
        & (df.dataset_label == 'PT_positive')
        & (df.multiplier == 0)
    ][['dataset_name', 'slope_mean']].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_base_to_user_pos_df = steerability_base_to_user_pos_df.rename(columns={'slope_mean': 'steerability_base_to_user_pos'})

    # SYS_POS -> USER_NEG
    steerability_ood_to_user_neg_df = df[
        (df.steering_label == 'SYS_positive')
        & (df.dataset_label == 'PT_negative')
        & (df.multiplier == 0)
    ][['dataset_name', 'slope_mean']].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_ood_to_user_neg_df = steerability_ood_to_user_neg_df.rename(columns={'slope_mean': 'steerability_ood_to_user_neg'})

    # SYS_NEG -> USER_POS
    steerability_ood_to_user_pos_df = df[
        (df.steering_label == 'SYS_negative')
        & (df.dataset_label == 'PT_positive')
        & (df.multiplier == 0)
    ][['dataset_name', 'slope_mean']].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_ood_to_user_pos_df = steerability_ood_to_user_pos_df.rename(columns={'slope_mean': 'steerability_ood_to_user_pos'})

    # Merge the dataframes
    steerability_df = steerability_id_df.merge(steerability_ood_df, on='dataset_name', suffixes=('_id', '_ood'))
    steerability_df = steerability_df.merge(steerability_base_to_user_neg_df, on='dataset_name')
    steerability_df = steerability_df.merge(steerability_base_to_user_pos_df, on='dataset_name')
    steerability_df = steerability_df.merge(steerability_ood_to_user_neg_df, on='dataset_name')
    steerability_df = steerability_df.merge(steerability_ood_to_user_pos_df, on='dataset_name')

    print(steerability_df.columns)

    # Save the dataframe for plotting between models
    steerability_df.to_parquet(f'{model_name}_steerability_summary.parquet.gzip', compression='gzip')
    steerability_df = steerability_df.rename(columns={
        'steerability_id': 'BASE -> BASE',
        'steerability_ood': 'SYS_POS -> SYS_NEG',
        'steerability_base_to_user_neg': 'BASE -> USER_NEG',
        'steerability_base_to_user_pos': 'BASE -> USER_POS',
        'steerability_ood_to_user_neg': 'SYS_POS -> USER_NEG',
        'steerability_ood_to_user_pos': 'SYS_NEG -> USER_POS',
    })
    
    return steerability_df

def make_id_vs_ood_steerability_plot_for_df(df, ax):

    sns.lineplot(data=df, x='BASE -> BASE', y='BASE -> BASE', color='black', linestyle='--')

    # Axes
    ax.axhline(0, color='black', linestyle='-')
    ax.axvline(0, color='black', linestyle='-')

    # ID vs OOD
    def make_plot_for_dist_shift(df, ax, dist_shift: str):
        # Label is raw string with -> replaced by right arrow
        label = dist_shift.replace(' -> ', r' $\rightarrow$ ')
        sns.regplot(data=df, x='BASE -> BASE', y=dist_shift, scatter = False, ax = ax)
        sns.scatterplot(data=df, x='BASE -> BASE', y=dist_shift, label=label, ax = ax)

    # make_plot_for_dist_shift('SYS_POS -> SYS_NEG')
    make_plot_for_dist_shift(df, ax, 'BASE -> USER_NEG')
    make_plot_for_dist_shift(df, ax, 'BASE -> USER_POS')
    make_plot_for_dist_shift(df, ax, 'SYS_POS -> USER_NEG')
    make_plot_for_dist_shift(df, ax, 'SYS_NEG -> USER_POS')

# %%
# Plot ID vs OOD steerability for Qwen and Llama

llama_df = pd.read_parquet('llama7b_steerability.parquet.gzip').drop_duplicates()
llama_steerability_df = compute_steerability_df(llama_df, 'llama7b')

qwen_df = pd.read_parquet('qwen_steerability.parquet.gzip').drop_duplicates()
qwen_steerability_df = compute_steerability_df(qwen_df, 'qwen')

# %%
def plot_qwen_and_llama_steerability_iid_vs_od(
    llama_steerability_df: pd.DataFrame,
    qwen_steerability_df: pd.DataFrame,
):

    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(9, 3.5), sharey = True)
    # fig.suptitle("ID vs OOD Steerability", y = 0.93)

    # Llama plot
    ax = axs[0]

    make_id_vs_ood_steerability_plot_for_df(llama_steerability_df, ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_xlabel(r'ID (BASE $\rightarrow$ BASE)', fontsize=16)
    # ax.set_ylabel('OOD', fontsize=16)
    ax.set_title(f'Llama-2-7b-Chat', fontsize=16)

    # Qwen plot
    ax = axs[1]

    make_id_vs_ood_steerability_plot_for_df(qwen_steerability_df, ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_xlabel(r'ID (BASE $\rightarrow$ BASE)', fontsize=16)
    # ax.set_ylabel('OOD', fontsize=16)
    ax.set_title(f'Qwen-1.5-14b-Chat', fontsize=16)

    # Global legend
    handles, labels = ax.get_legend_handles_labels()
    # Fig legend overhead, but without covering the titles!
    # Fancy box
    fig.legend(
        handles, labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.15), 
        ncol=4, fancybox=True, 
        shadow=True, prop={'size': 10},
        columnspacing=0.0
    )

    # Remove the legends from the individual plots
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()

    # A hack to get a global shared xlabel and ylabel
    extra_ax = fig.add_subplot(111, frameon=False)
    extra_ax.grid(False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(r'ID (BASE $\rightarrow$ BASE)', fontsize=16)
    plt.ylabel('OOD', fontsize=16)

    fig.tight_layout()
    fig.savefig(f'figures/qwen_and_llama2_steerability_id_vs_ood.pdf', bbox_inches='tight')

plot_qwen_and_llama_steerability_iid_vs_od(
    llama_steerability_df, qwen_steerability_df
)

# %%
# Correlate ID and OOD steerability between models
from scipy.stats import spearmanr

combined_df = llama_steerability_df.merge(qwen_steerability_df, on='dataset_name', suffixes=('_llama7b', '_qwen'))
plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(9, 3.5), sharey = True, sharex = True)

# ID steerability
ax = axs[0] 
ax.axhline(0, color='black', linestyle='-')
ax.axvline(0, color='black', linestyle='-') 
sns.regplot(data=combined_df, x='BASE -> BASE_llama7b', y='BASE -> BASE_qwen', ax = ax)
spearman_corr, spearman_p = spearmanr(combined_df['BASE -> BASE_llama7b'], combined_df['BASE -> BASE_qwen'])
print(f"Spearman correlation for BASE -> BASE: {spearman_corr} (p = {spearman_p})")
# Draw the x = y dotted line
x = combined_df['BASE -> BASE_llama7b']
y = combined_df['BASE -> BASE_qwen']
min = x.min() if x.min() < y.min() else y.min()
max = x.max() if x.max() > y.max() else y.max()
ax.plot([min, max], [min, max], color='black', linestyle='--')
ax.set_xlabel('')
ax.set_ylabel('')
# ax.set_xlabel('Llama-2-7b-Chat', fontsize=16)
# ax.set_ylabel('Qwen-1.5-14b-Chat',  fontsize=16)
ax.set_title('ID Steerability',  fontsize=16)

# OOD steerability
ax = axs[1] 
ax.axhline(0, color='black', linestyle='-')
ax.axvline(0, color='black', linestyle='-') 
# Draw the x = y dotted line
x = combined_df['BASE -> BASE_llama7b']
y = combined_df['BASE -> BASE_qwen']
min = x.min() if x.min() < y.min() else y.min()
max = x.max() if x.max() > y.max() else y.max()
ax.plot([min, max], [min, max], color='black', linestyle='--')


sns.regplot(data=combined_df, x='SYS_POS -> USER_NEG_llama7b', y='SYS_POS -> USER_NEG_qwen', ax = ax)
# Print the coefficient 
spearman_corr, spearman_p = spearmanr(combined_df['SYS_POS -> USER_NEG_llama7b'], combined_df['SYS_POS -> USER_NEG_qwen'])
print(f"Spearman correlation for SYS_POS -> USER_NEG: {spearman_corr} (p = {spearman_p})")
ax.set_xlabel('')
ax.set_ylabel('')
# ax.set_xlabel('Llama-2-7b-Chat',  fontsize=16)
# ax.set_ylabel('Qwen-1.5-14b-Chat',  fontsize=16)
ax.set_title('OOD Steerability',  fontsize=16)

# Global legend
# handles, labels = ax.get_legend_handles_labels()
# Fig legend overhead, but without covering the titles!
# Fancy box
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2, fancybox=True, shadow=True)
# fig.suptitle("Correlation in Mean Steerability")

# Remove the legends from the individual plots
# axs[0].get_legend().remove()
# axs[1].get_legend().remove()

extra_ax = fig.add_subplot(111, frameon=False)
extra_ax.grid(False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Llama-2-7b-Chat', fontsize=16)
plt.ylabel('Qwen-1.5-14b-Chat', fontsize=16)
fig.tight_layout()
fig.savefig(f'figures/correlation_steerability_between_models.pdf', bbox_inches='tight')


# %%
# Correlate ID and OOD steerability variance between models

def compute_steerability_variance_df(df, model_name: str):
    """ Get a dataframe with various ID / OOD settings. """
    # Calculate overall steerability by dataset. 
    # Calculate steerability within each flavour
    mean_slope = df.groupby(['dataset_name', 'steering_label', 'dataset_label'])['slope'].var()
    df = df.merge(mean_slope, on=['dataset_name', 'steering_label', 'dataset_label'], suffixes=('', '_mean'))

    # BASE -> BASE
    steerability_id_df = df[
        (df.steering_label == 'baseline')
        & (df.dataset_label == 'baseline')
        & (df.multiplier == 0)
    ][['dataset_name', 'slope_mean']].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_id_df = steerability_id_df.rename(columns={'slope_mean': 'steerability'})

    # SYS_POS -> SYS_NEG
    steerability_ood_df = df[
        (df.steering_label == 'SYS_positive')
        & (df.dataset_label == 'SYS_negative')
        & (df.multiplier == 0)
    ][['dataset_name', 'slope_mean']].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_ood_df = steerability_ood_df.rename(columns={'slope_mean': 'steerability'})

    # BASE -> USER_NEG
    steerability_base_to_user_neg_df = df[
        (df.steering_label == 'baseline')
        & (df.dataset_label == 'PT_negative')
        & (df.multiplier == 0)
    ][['dataset_name', 'slope_mean']].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_base_to_user_neg_df = steerability_base_to_user_neg_df.rename(columns={'slope_mean': 'steerability_base_to_user_neg'})

    # BASE -> USER_POS
    steerability_base_to_user_pos_df = df[
        (df.steering_label == 'baseline')
        & (df.dataset_label == 'PT_positive')
        & (df.multiplier == 0)
    ][['dataset_name', 'slope_mean']].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_base_to_user_pos_df = steerability_base_to_user_pos_df.rename(columns={'slope_mean': 'steerability_base_to_user_pos'})

    # SYS_POS -> USER_NEG
    steerability_ood_to_user_neg_df = df[
        (df.steering_label == 'SYS_positive')
        & (df.dataset_label == 'PT_negative')
        & (df.multiplier == 0)
    ][['dataset_name', 'slope_mean']].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_ood_to_user_neg_df = steerability_ood_to_user_neg_df.rename(columns={'slope_mean': 'steerability_ood_to_user_neg'})

    # SYS_NEG -> USER_POS
    steerability_ood_to_user_pos_df = df[
        (df.steering_label == 'SYS_negative')
        & (df.dataset_label == 'PT_positive')
        & (df.multiplier == 0)
    ][['dataset_name', 'slope_mean']].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_ood_to_user_pos_df = steerability_ood_to_user_pos_df.rename(columns={'slope_mean': 'steerability_ood_to_user_pos'})

    # Merge the dataframes
    steerability_df = steerability_id_df.merge(steerability_ood_df, on='dataset_name', suffixes=('_id', '_ood'))
    steerability_df = steerability_df.merge(steerability_base_to_user_neg_df, on='dataset_name')
    steerability_df = steerability_df.merge(steerability_base_to_user_pos_df, on='dataset_name')
    steerability_df = steerability_df.merge(steerability_ood_to_user_neg_df, on='dataset_name')
    steerability_df = steerability_df.merge(steerability_ood_to_user_pos_df, on='dataset_name')

    print(steerability_df.columns)

    # Save the dataframe for plotting between models
    steerability_df.to_parquet(f'{model_name}_steerability_summary.parquet.gzip', compression='gzip')
    steerability_df = steerability_df.rename(columns={
        'steerability_id': 'BASE -> BASE',
        'steerability_ood': 'SYS_POS -> SYS_NEG',
        'steerability_base_to_user_neg': 'BASE -> USER_NEG',
        'steerability_base_to_user_pos': 'BASE -> USER_POS',
        'steerability_ood_to_user_neg': 'SYS_POS -> USER_NEG',
        'steerability_ood_to_user_pos': 'SYS_NEG -> USER_POS',
    })
    
    return steerability_df

# Compute the data
llama_df = pd.read_parquet('llama7b_steerability.parquet.gzip').drop_duplicates()
llama_steerability_variance_df = compute_steerability_variance_df(llama_df, 'llama7b')

qwen_df = pd.read_parquet('qwen_steerability.parquet.gzip').drop_duplicates()
qwen_steerability_variance_df = compute_steerability_variance_df(qwen_df, 'qwen')

combined_steerability_variance_df = llama_steerability_variance_df.merge(qwen_steerability_variance_df, on='dataset_name', suffixes=('_llama7b', '_qwen'))

# %%
def plot_qwen_and_llama_steerability_variance_iid_vs_od(
    llama_steerability_variance_df: pd.DataFrame,
    qwen_steerability_variance_df: pd.DataFrame,
):

    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(9, 3.5), sharey = True)
    # fig.suptitle("ID vs OOD Steerability", y = 0.93)

    # Llama plot
    ax = axs[0]

    make_id_vs_ood_steerability_plot_for_df(llama_steerability_variance_df, ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_xlabel(r'ID (BASE $\rightarrow$ BASE)', fontsize=16)
    # ax.set_ylabel('OOD', fontsize=16)
    ax.set_title(f'Llama-2-7b-Chat', fontsize=16)

    # Qwen plot
    ax = axs[1]

    make_id_vs_ood_steerability_plot_for_df(qwen_steerability_variance_df, ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_xlabel(r'ID (BASE $\rightarrow$ BASE)', fontsize=16)
    # ax.set_ylabel('OOD', fontsize=16)
    ax.set_title(f'Qwen-1.5-14b-Chat', fontsize=16)

    # Global legend
    handles, labels = ax.get_legend_handles_labels()
    # Fig legend overhead, but without covering the titles!
    # Fancy box
    fig.legend(
        handles, labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.15), 
        ncol=4, fancybox=True, 
        shadow=True, prop={'size': 10},
        columnspacing=0.0
    )

    # Remove the legends from the individual plots
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()

    # A hack to get a global shared xlabel and ylabel
    extra_ax = fig.add_subplot(111, frameon=False)
    extra_ax.grid(False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(r'ID Variance (BASE $\rightarrow$ BASE)', fontsize=16)
    plt.ylabel('OOD Variance', fontsize=16)

    fig.tight_layout()
    fig.savefig(f'figures/qwen_and_llama2_steerability_variance_id_vs_ood.pdf', bbox_inches='tight')

plot_qwen_and_llama_steerability_variance_iid_vs_od(
    llama_steerability_variance_df, qwen_steerability_variance_df
)

# %%
def plot_qwen_and_llama_steerability_variance_id_vs_ood(
    combined_df: pd.DataFrame,
):

    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(9, 3.5), sharey = True)

    # Axes 

    # ID steerability
    ax = axs[0]
    ax.axhline(0, color='black', linestyle='-')
    ax.axvline(0, color='black', linestyle='-') 
    sns.regplot(data=combined_df, x='BASE -> BASE_llama7b', y='BASE -> BASE_qwen', ax = ax)
    spearman_corr, spearman_p = spearmanr(combined_df['BASE -> BASE_llama7b'], combined_df['BASE -> BASE_qwen'])
    print(f"Spearman correlation for BASE -> BASE: {spearman_corr} (p = {spearman_p})")
    # Draw the x = y dotted line
    x = combined_df['BASE -> BASE_llama7b']
    y = combined_df['BASE -> BASE_qwen']
    min = x.min() if x.min() < y.min() else y.min()
    max = x.max() if x.max() > y.max() else y.max()
    ax.plot([min, max], [min, max], color='black', linestyle='--')

    # ax.set_xlabel('Llama-2-7b-Chat', fontsize=16)
    # ax.set_ylabel('Qwen-1.5-14b-Chat', fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('ID St. Variance', fontsize=16)

    # OOD steerability
    ax = axs[1] 
    ax.axhline(0, color='black', linestyle='-')
    ax.axvline(0, color='black', linestyle='-') 
    # Draw the x = y dotted line
    x = combined_df['SYS_POS -> USER_NEG_llama7b']
    y = combined_df['SYS_POS -> USER_NEG_qwen']
    min = x.min() if x.min() < y.min() else y.min()
    max = x.max() if x.max() > y.max() else y.max()
    ax.plot([min, max], [min, max], color='black', linestyle='--')

    sns.regplot(data=combined_df, x='SYS_POS -> USER_NEG_llama7b', y='SYS_POS -> USER_NEG_qwen', ax = ax)
    # Print the coefficient
    spearman_corr, spearman_p = spearmanr(combined_df['SYS_POS -> USER_NEG_llama7b'], combined_df['SYS_POS -> USER_NEG_qwen'])
    print(f"Spearman correlation for SYS_POS -> USER_NEG: {spearman_corr} (p = {spearman_p})")

    # ax.set_xlabel('Llama-2-7b-Chat', fontsize=16)
    # ax.set_ylabel('Qwen-1.5-14b-Chat', fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('OOD St. Variance', fontsize=16)

    extra_ax = fig.add_subplot(111, frameon=False)
    extra_ax.grid(False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Llama-2-7b-Chat', fontsize=16)
    plt.ylabel('Qwen-1.5-14b-Chat', fontsize=16)

    fig.tight_layout()
    # fig.suptitle("Correlation in Steerability Variance", y=1.03)
    fig.savefig(f'figures/correlation_steerability_variance_between_models.pdf', bbox_inches='tight')

plot_qwen_and_llama_steerability_variance_id_vs_ood(
    combined_steerability_variance_df
)
# %%
def compute_steerability_variance_id_vs_ood(df):
    # Calculate overall steerability by dataset.
    # Calculate steerability within each flavour
    var_slope = df.groupby(["dataset_name", "steering_label", "dataset_label"])[
        "slope"
    ].var()
    df = df.merge(
        var_slope,
        on=["dataset_name", "steering_label", "dataset_label"],
        suffixes=("", "_var"),
    )

    # BASE -> BASE
    steerability_id_df = df[
        (df.steering_label == "baseline")
        & (df.dataset_label == "baseline")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_var"]].drop_duplicates()
    # Rename 'slope_var' to 'steerability'
    steerability_id_df = steerability_id_df.rename(
        columns={"slope_var": "steerability"}
    )

    # SYS_POS -> SYS_NEG
    steerability_ood_df = df[
        (df.steering_label == "SYS_positive")
        & (df.dataset_label == "SYS_negative")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_var"]].drop_duplicates()
    # Rename 'slope_var' to 'steerability'
    steerability_ood_df = steerability_ood_df.rename(
        columns={"slope_var": "steerability"}
    )

    # BASE -> USER_NEG
    steerability_base_to_user_neg_df = df[
        (df.steering_label == "baseline")
        & (df.dataset_label == "PT_negative")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_var"]].drop_duplicates()
    # Rename 'slope_var' to 'steerability'
    steerability_base_to_user_neg_df = steerability_base_to_user_neg_df.rename(
        columns={"slope_var": "steerability_base_to_user_neg"}
    )

    # BASE -> USER_POS
    steerability_base_to_user_pos_df = df[
        (df.steering_label == "baseline")
        & (df.dataset_label == "PT_positive")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_var"]].drop_duplicates()
    # Rename 'slope_var' to 'steerability'
    steerability_base_to_user_pos_df = steerability_base_to_user_pos_df.rename(
        columns={"slope_var": "steerability_base_to_user_pos"}
    )

    # SYS_POS -> USER_NEG
    steerability_ood_to_user_neg_df = df[
        (df.steering_label == "SYS_positive")
        & (df.dataset_label == "PT_negative")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_var"]].drop_duplicates()
    # Rename 'slope_var' to 'steerability'
    steerability_ood_to_user_neg_df = steerability_ood_to_user_neg_df.rename(
        columns={"slope_var": "steerability_ood_to_user_neg"}
    )

    # SYS_NEG -> USER_POS
    steerability_ood_to_user_pos_df = df[
        (df.steering_label == "SYS_negative")
        & (df.dataset_label == "PT_positive")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_var"]].drop_duplicates()
    # Rename 'slope_var' to 'steerability'
    steerability_ood_to_user_pos_df = steerability_ood_to_user_pos_df.rename(
        columns={"slope_var": "steerability_ood_to_user_pos"}
    )

    # Merge the dataframes
    steerability_df = steerability_id_df.merge(
        steerability_ood_df, on="dataset_name", suffixes=("_id", "_ood")
    )
    steerability_df = steerability_df.merge(
        steerability_base_to_user_neg_df, on="dataset_name"
    )
    steerability_df = steerability_df.merge(
        steerability_base_to_user_pos_df, on="dataset_name"
    )
    steerability_df = steerability_df.merge(
        steerability_ood_to_user_neg_df, on="dataset_name"
    )
    steerability_df = steerability_df.merge(
        steerability_ood_to_user_pos_df, on="dataset_name"
    )

    print(steerability_df.columns)

    # Save the dataframe for plotting between models
    steerability_df.to_parquet(
        f"{model}_steerability_summary.parquet.gzip", compression="gzip"
    )

    # Plot
    # Rename
    steerability_df = steerability_df.rename(
        columns={
            "steerability_id": "BASE -> BASE",
            "steerability_ood": "SYS_POS -> SYS_NEG",
            "steerability_base_to_user_neg": "BASE -> USER_NEG",
            "steerability_base_to_user_pos": "BASE -> USER_POS",
            "steerability_ood_to_user_neg": "SYS_POS -> USER_NEG",
            "steerability_ood_to_user_pos": "SYS_NEG -> USER_POS",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(
        data=steerability_df,
        x="BASE -> BASE",
        y="BASE -> BASE",
        color="black",
        linestyle="--",
    )
    ax.axhline(0, color="black", linestyle="--")
    ax.axvline(0, color="black", linestyle="--")

    # ID vs OOD
    def make_plot_for_dist_shift(dist_shift: str):
        sns.regplot(
            data=steerability_df, x="BASE -> BASE", y=dist_shift, scatter=False, ax=ax
        )
        sns.scatterplot(
            data=steerability_df,
            x="BASE -> BASE",
            y=dist_shift,
            label=dist_shift,
            ax=ax,
        )

    # make_plot_for_dist_shift('SYS_POS -> SYS_NEG')
    make_plot_for_dist_shift("BASE -> USER_NEG")
    make_plot_for_dist_shift("BASE -> USER_POS")
    make_plot_for_dist_shift("SYS_POS -> USER_NEG")
    make_plot_for_dist_shift("SYS_NEG -> USER_POS")

    plt.xlabel("ID steerability (BASE -> BASE)")
    plt.ylabel("OOD steerability")
    plt.title(f"{model_full_name} ID vs OOD Variance in Steerability")
    fig.tight_layout()
    fig.savefig(f"figures/{model}_steerability_variance_id_vs_ood.pdf", bbox_inches='tight')


compute_steerability_variance_id_vs_ood(df)
