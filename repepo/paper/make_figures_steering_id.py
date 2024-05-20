# %%
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.formula.api import ols

sns.set_theme()


# %%
model = 'llama7b' 
# model = 'qwen'

# %%
# Load preprocessed data
df = pd.read_parquet(f'{model}_steerability.parquet.gzip')

# %%
# Compute steerability statistics

df['median_slope'] = df.groupby(['dataset_name', 'steering_label', 'dataset_label'])['slope'].transform('median')
df['mean_slope'] = df.groupby(['dataset_name', 'steering_label', 'dataset_label'])['slope'].transform('mean')
df['std_slope'] = df.groupby(['dataset_name', 'steering_label', 'dataset_label'])['slope'].transform('std')
df['kurtosis_slope'] = df.groupby(['dataset_name', 'steering_label', 'dataset_label'])['slope'].transform(pd.Series.kurt)
df['sign_slope'] = df['slope'] > 0
df['frac_anti_steerable'] = 1 - df.groupby(['dataset_name', 'steering_label', 'dataset_label'])['sign_slope'].transform('mean')
df['pos_option_is_A'] = df['test_example.positive.text'].str.endswith('(A)').astype(bool)

df['response_is_A_and_Yes'] = df['test_example.positive.text'].str.contains('\(A\):[ ]+Yes', regex = True)
df['response_is_B_and_Yes'] = df['test_example.positive.text'].str.contains('\(B\):[ ]+Yes', regex = True)
df['pos_option_is_Yes'] = (
    (df['response_is_A_and_Yes'] & df['pos_option_is_A'])
    | (df['response_is_B_and_Yes'] & ~df['pos_option_is_A'])
)

print(df.columns)
df.head()



# %%
# Plot: Per-Sample Steerability

def plot_per_sample_steerability(df):
    plot_df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline')
    ]
    order = plot_df[['dataset_name', 'median_slope']].drop_duplicates().sort_values('median_slope', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 20))
    sns.violinplot(plot_df, x='slope', y = 'dataset_name', hue='dataset_name', ax=ax, order = order['dataset_name'])
    ax.axvline(x = 0, color = 'black', linestyle = '--')
    fig.tight_layout()
    fig.savefig('figures/per_sample_steerability.png')

plot_per_sample_steerability(df)
# %%

def plot_per_sample_steerability_top_5_mid_5_bot_5(df):
    plot_df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline')
    ]
    order = plot_df[['dataset_name', 'median_slope']].drop_duplicates().sort_values('median_slope', ascending=False)

    # Plot only the top 5 datasets
    fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize=(10, 7), sharex=True)
    top_5 = order.head(5)
    plot_df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline') &
        (df['dataset_name'].isin(top_5['dataset_name']))
    ]
    sns.violinplot(plot_df, x='slope', y = 'dataset_name', hue='dataset_name', ax=ax[0], order = top_5['dataset_name'])
    ax[0].set_title('Top 5 datasets by median slope')
    ax[0].axvline(x = 0, color = 'black', linestyle = '--')

    # Plot the middle 5 datasets
    middle_5 = order[18:23]
    plot_df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline') &
        (df['dataset_name'].isin(middle_5['dataset_name']))
    ]
    ax[1].set_title('Middle 5 datasets by median slope')
    ax[1].axvline(x = 0, color = 'black', linestyle = '--')
    sns.violinplot(plot_df, x='slope', y = 'dataset_name', hue='dataset_name', ax=ax[1], order = middle_5['dataset_name'])

    # Plot only the bottom 5 datasets
    bottom_5 = order.tail(5)
    plot_df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline') &
        (df['dataset_name'].isin(bottom_5['dataset_name']))
    ]
    ax[2].set_title('Bottom 5 datasets by median slope')
    ax[2].axvline(x = 0, color = 'black', linestyle = '--')
    sns.violinplot(plot_df, x='slope', y = 'dataset_name', hue='dataset_name', ax=ax[2], order = bottom_5['dataset_name'])
    fig.show()
    fig.savefig('figures/per_sample_steerability_top5_bot5.png')

plot_per_sample_steerability_top_5_mid_5_bot_5(df)

# %%

def plot_per_sample_steerability_and_fraction_anti_steerable(df):

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 15), sharey=True)

    # Per sample steerability
    plot_df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline')
    ]
    order = plot_df[['dataset_name', 'median_slope']].drop_duplicates().sort_values('median_slope', ascending=False)
    # Rename 'slope' to 'steerability' 
    plot_df = plot_df.rename(columns = {'slope': 'steerability'})

    # Plot
    ax = axs[0]
    sns.violinplot(plot_df, x='steerability', y = 'dataset_name', hue='dataset_name', ax=ax, order = order['dataset_name'])
    ax.axvline(x = 0, color = 'black', linestyle = '--')
    ax.set_title("Per-sample steerability")

    # Fraction of anti-steerable examples
    plot_df['sign_slope_mean'] = plot_df.groupby('dataset_name')['sign_slope'].transform('mean')
    plot_df['frac_anti_steerable'] = 1 - plot_df['sign_slope_mean']
    plot_df = plot_df[['dataset_name', 'frac_anti_steerable']].drop_duplicates()

    # Plot
    ax = axs[1]
    sns.barplot(plot_df, y='dataset_name', x = 'frac_anti_steerable', ax=ax, order = order['dataset_name'])
    ax.set_title("Anti-steerable examples")

    # Finish
    
    fig.tight_layout()
    fig.savefig('figures/fraction_anti_steerable.png')

plot_per_sample_steerability_and_fraction_anti_steerable(df)

# %%
# Bar chart of slope vs response_is_A

def plot_slope_vs_response_is_A(df):

    plot_df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline')
    ]
    # Rename 
    plot_df = plot_df.rename(columns = {'slope': 'Steerability', 'dataset_name': 'Dataset'})
    plot_df['Positive Option'] = plot_df['pos_option_is_A'].apply(lambda x: 'A' if x else 'B')


    order = plot_df[['Dataset', 'median_slope']].drop_duplicates().sort_values('median_slope', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 15))
    sns.barplot(
        data=plot_df, 
        hue = 'Positive Option', 
        x='Steerability', 
        y='Dataset', 
        ax=ax, 
        order = order['Dataset'],
        errorbar = None
    )

    fig.tight_layout()
    fig.savefig('figures/slope_vs_pos_option_is_A.png')

plot_slope_vs_response_is_A(df)

# %% 
# Plot dataset statistics of pos_option_is_Yes

def plot_response_is_A(df):

    plot_df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline') & 
        (df['multiplier'] == 0)
    ] 

    # Rename 
    plot_df = plot_df.rename(columns = {'slope': 'Steerability', 'dataset_name': 'Dataset'})
    plot_df['Positive Option'] = plot_df['pos_option_is_A'].apply(lambda x: 'A' if x else 'B')
    order = plot_df[['Dataset', 'median_slope']].drop_duplicates().sort_values('median_slope', ascending=False)

    # Plot a stacked barplot of the fraction of A vs B responses in each dataset.
    count_df = plot_df.groupby(['Dataset', 'Positive Option']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 15))
    count_df.plot(kind='barh', stacked=True, ax = ax)
    ax.set_ylabel('Dataset')
    ax.set_xlabel('Count')
    ax.set_title('Count of Positive Options per Dataset')
    fig.tight_layout()
    plt.show()


plot_response_is_A(df)

# %%
def plot_slope_vs_response_is_Yes(df):

    # Filter by datasets where there is at least one Yes and one No
    # Count the number of "no" responses
    # yes_counts = df.groupby('dataset_name')['pos_option_is_Yes'].sum()
    # total_counts = df.groupby('dataset_name')['pos_option_is_Yes'].count()
    # # Filter by datasets where there is at least one Yes and one No
    # selected_datasets = total_counts[(yes_counts > 0) & (yes_counts < total_counts)]
    # print(selected_datasets)

    fig, ax = plt.subplots(figsize=(10, 15))

    plot_df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline') 
        # (df['dataset_name'].isin(selected_datasets.index))
    ]
    # Rename 
    plot_df = plot_df.rename(columns = {'slope': 'Steerability', 'dataset_name': 'Dataset'})

    plot_df['pos_A'] = plot_df['pos_option_is_A'].apply(lambda x: 'A' if x else 'B')
    plot_df['pos_Yes'] = plot_df['pos_option_is_Yes'].apply(lambda x: 'Yes' if x else 'No')
    plot_df['Positive Option'] = plot_df['pos_A'] + ' and ' + plot_df['pos_Yes']

    order = plot_df[['Dataset', 'median_slope']].drop_duplicates().sort_values('median_slope', ascending=False)
    hue_order = ['A and No', 'A and Yes', 'B and No', 'B and Yes']
    sns.barplot(
        data=plot_df, 
        hue = 'Positive Option', 
        x='Steerability', 
        y='Dataset', 
        ax=ax, 
        order = order['Dataset'],
        hue_order = hue_order,
        errorbar = None
    )
    fig.tight_layout()
    fig.savefig('figures/slope_vs_pos_option_is_Yes.png')

plot_slope_vs_response_is_Yes(df)

# %%
def plot_response_is_Yes(df):

    plot_df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline') & 
        (df['multiplier'] == 0)
    ] 

    # Rename 
    plot_df = plot_df.rename(columns = {'slope': 'Steerability', 'dataset_name': 'Dataset'})
    plot_df['pos_A'] = plot_df['pos_option_is_A'].apply(lambda x: 'A' if x else 'B')
    plot_df['pos_Yes'] = plot_df['pos_option_is_Yes'].apply(lambda x: 'Yes' if x else 'No')
    plot_df['Positive Option'] = plot_df['pos_A'] + ' and ' + plot_df['pos_Yes']
    order = plot_df[['Dataset', 'median_slope']].drop_duplicates().sort_values('median_slope', ascending=False)

    # Plot a stacked barplot of the fraction of A vs B responses in each dataset.
    count_df = plot_df.groupby(['Dataset', 'Positive Option']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 15))
    # Set order by order
    count_df = count_df.loc[order['Dataset']]
    count_df.plot(kind='barh', stacked=True, ax = ax)
    ax.set_ylabel('Dataset')
    ax.set_xlabel('Count')
    ax.set_title('Count of Positive Options per Dataset')
    fig.tight_layout()
    fig.savefig("figures/counts_of_positive_options_per_dataset.png")
    plt.show()

plot_response_is_Yes(df)

# %%
# The above two in the same figure
def plot_slope_and_counts_for_response_is_Yes(df):

    fig, axs = plt.subplots(
        nrows = 1, 
        ncols = 2, 
        width_ratios=[5, 1], 
        figsize=(10, 10), 
        sharey=True
    )
    ax = axs[0]

    plot_df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline') 
        # & (df['dataset_name'].isin(selected_datasets.index))
    ]
    # Rename 
    plot_df = plot_df.rename(columns = {'slope': 'Steerability', 'dataset_name': 'Dataset'})

    plot_df['pos_A'] = plot_df['pos_option_is_A'].apply(lambda x: 'A' if x else 'B')
    plot_df['pos_Yes'] = plot_df['pos_option_is_Yes'].apply(lambda x: 'Yes' if x else 'No')
    plot_df['Positive Option'] = plot_df['pos_A'] + ' and ' + plot_df['pos_Yes']

    order = plot_df[['Dataset', 'median_slope']].drop_duplicates().sort_values('median_slope', ascending=False)
    hue_order = ['A and No', 'A and Yes', 'B and No', 'B and Yes']
    sns.barplot(
        data=plot_df, 
        hue = 'Positive Option', 
        x='Steerability', 
        y='Dataset', 
        ax=ax, 
        order = order['Dataset'],
        hue_order = hue_order,
        errorbar = None
    )
    ax.set_title('Mean Steerability')

    ax = axs[1]
    plot_df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline') & 
        (df['multiplier'] == 0)
    ] 

    # Rename 
    plot_df = plot_df.rename(columns = {'slope': 'Steerability', 'dataset_name': 'Dataset'})
    plot_df['pos_A'] = plot_df['pos_option_is_A'].apply(lambda x: 'A' if x else 'B')
    plot_df['pos_Yes'] = plot_df['pos_option_is_Yes'].apply(lambda x: 'Yes' if x else 'No')
    plot_df['Positive Option'] = plot_df['pos_A'] + ' and ' + plot_df['pos_Yes']

    # Plot a stacked barplot of the fraction of A vs B responses in each dataset.
    count_df = plot_df.groupby(['Dataset', 'Positive Option']).size().unstack().fillna(0)
    count_df.plot(kind='barh', stacked=True, ax = ax)
    ax.set_ylabel('Dataset')
    ax.set_xlabel('Count')
    ax.set_title('Option Counts')
    ax.get_legend().remove()
    fig.tight_layout()
    fig.savefig("figures/plot_slope_and_counts_for_response_is_Yes.png")
    plt.show()

plot_slope_and_counts_for_response_is_Yes(df)


# %% 

def compute_variance(df, dataset_name):
    df = df[
        (df['dataset_name'] == dataset_name)
        & (df.multiplier == 0)
    ][['pos_option_is_A', 'pos_option_is_Yes', 'slope']]

    df['pos_option_is_A'] = df['pos_option_is_A'].astype(int)
    df['pos_option_is_Yes'] = df['pos_option_is_Yes'].astype(int)

    # Total variance of 'slope'
    total_variance = df['slope'].var()

    # Variance explained by 'pos_option_is_A'
    model_A = ols('slope ~ pos_option_is_A', data=df).fit()
    residuals_A = model_A.resid
    explained_variance_A = total_variance - residuals_A.var()

    # Variance explained by 'pos_option_is_Yes'
    model_Yes = ols('slope ~ pos_option_is_Yes', data=df).fit()
    residuals_Yes = model_Yes.resid
    explained_variance_Yes = total_variance - residuals_Yes.var()

    # Variance explained by both 'pos_option_is_A' and 'pos_option_is_Yes'
    model_both = ols('slope ~ pos_option_is_A + pos_option_is_Yes', data=df).fit()
    residuals_both = model_both.resid
    explained_variance_both = total_variance - residuals_both.var()


    marginal_explained_variance_yes = explained_variance_both - explained_variance_A
    unexplained_variance = total_variance - explained_variance_both
    return {
        'dataset_name': dataset_name,
        'total_variance': total_variance,
        'variance_explained_A': explained_variance_A,
        'variance_explained_Yes': explained_variance_Yes,
        'variance_explained_both': explained_variance_both,
        'marginal_variance_explained_Yes': marginal_explained_variance_yes,
        'unexplained_variance': unexplained_variance        
    }

def plot_variance(df):

    df = df[
        (df['steering_label'] == 'baseline') &
        (df['dataset_label'] == 'baseline') & 
        (df['multiplier'] == 0)
    ] 

    rows = []
    for dataset_name in df.dataset_name.unique():
        rows.append(compute_variance(df, dataset_name))
    variance_df = pd.DataFrame(rows)
    # Rename
    variance_df = variance_df[[
        'dataset_name', 
        'variance_explained_A', 
        'marginal_variance_explained_Yes', 
        'unexplained_variance'
    ]]
    variance_df = variance_df.rename(columns = {
        'dataset_name': 'Dataset',
        'variance_explained_A': 'Var Explained: A/B',
        'marginal_variance_explained_Yes': 'Marginal Var Explained: Yes/No',
        'unexplained_variance': 'Unexplained'
    })

    # Stacked barplot
    fig, ax = plt.subplots(figsize=(10, 15))
    variance_df = variance_df.set_index('Dataset')
    # Fix order to Unexplained, Marginal Var Explained, Var Explained
    variance_df = variance_df[['Unexplained', 'Marginal Var Explained: Yes/No', 'Var Explained: A/B']]
    variance_df.plot(kind='barh', stacked=True, ax = ax)
    fig.tight_layout()
    fig.savefig('figures/breakdown_variance_explained_by_spurious_factors.png')

plot_variance(df)

