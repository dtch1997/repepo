# %%
"""

Assumes you have run repepo.paper.make_figurse_steering_ood for both models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sns.set_theme()

# Load the steerability data
llama7b_df = pd.read_parquet("llama7b_steerability_summary.parquet.gzip")
llama7b_df = llama7b_df.drop_duplicates()
qwen_df = pd.read_parquet("qwen_steerability_summary.parquet.gzip")
qwen_df = qwen_df.drop_duplicates()

combined = llama7b_df.merge(qwen_df, on="dataset_name", suffixes=("_llama7b", "_qwen"))

# %%
# Correlation in gen gap
sns.regplot(data=combined, x="gap_qwen", y="gap_llama7b")

# %%
# Correlation in steerability
fig, ax = plt.subplots()
sns.regplot(data=combined, x="steerability_id_qwen", y="steerability_id_llama7b")
# Draw the x = y line
x = combined["steerability_id_qwen"]
y = combined["steerability_id_llama7b"]

min = x.min() if x.min() < y.min() else y.min()
max = x.max() if x.max() > y.max() else y.max()
ax.plot([min, max], [min, max], color="black", linestyle="--")
plt.xlabel("Qwen ID steerability")
plt.ylabel("Llama7b ID steerability")

fig.suptitle("Steerability ID for Qwen and Llama7b")
fig.savefig("figures/id_steerability_correlation.png")
plt.show()

# Print the spearman correlation
result = spearmanr(
    combined["steerability_id_qwen"], combined["steerability_id_llama7b"]
)
print(f"{result.statistic:.3f}")  # type: ignore

# %%
# Correlation in ood steerability
fig, ax = plt.subplots()
sns.regplot(data=combined, x="steerability_ood_qwen", y="steerability_ood_llama7b")
# Draw the x = y line
x = combined["steerability_ood_qwen"]
y = combined["steerability_ood_qwen"]

min = x.min() if x.min() < y.min() else y.min()
max = x.max() if x.max() > y.max() else y.max()
ax.plot([min, max], [min, max], color="black", linestyle="--")
plt.xlabel("Qwen OOD steerability")
plt.ylabel("Llama7b OOD steerability")

fig.suptitle("Steerability OOD for Qwen and Llama7b")
fig.savefig("figures/ood_steerability_correlation.png")
plt.show()

result = spearmanr(
    combined["steerability_ood_qwen"], combined["steerability_ood_llama7b"]
)
print(f"{result.statistic:.3f}")  # type: ignore
# %%
