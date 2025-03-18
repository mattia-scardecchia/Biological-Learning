import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base_dir = "data/grid-search-03-03"
folders = os.listdir(base_dir)


dfs = [
    pd.read_csv(os.path.join(base_dir, folder, "grid_search_results.csv"))
    for folder in folders
]
index_columns = [
    c
    for c in dfs[0].columns
    if "acc" not in c and c not in ["lambda_right", "lambda_y"]
]
df = pd.concat(dfs, axis=0).set_index(index_columns)
df = df.drop(columns=["lambda_right", "lambda_y"])
means = df.groupby(index_columns).mean()
stds = df.groupby(index_columns).std()

means.sort_values("max_train_acc", ascending=False, inplace=True)
means.to_csv(os.path.join(base_dir, "sorted_means.csv"))


# 1. Filter rows based on a threshold on max_train_acc
threshold_value = 0.80  # set your desired threshold here
filtered = means[means["max_train_acc"] > threshold_value]

# 2. Plot histograms of all 4 index variables
index_names = (
    means.index.names
)  # should be ['lr', 'threshold', 'lambda_left', 'lambda_x']
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.flatten()
for i, name in enumerate(index_names):
    # Get the index level values (converted to float if necessary)
    vals = filtered.index.get_level_values(name).astype(float)
    axs[i].hist(vals, bins=10, edgecolor="black")
    axs[i].set_title(f"Histogram of {name}")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, f"histograms-filtered{threshold_value}.png"))
plt.show()

# 3. Choose a reference 4-tuple using, for example, the median of each index in the filtered data
ref_values = {
    name: np.median(filtered.index.get_level_values(name).astype(float))
    for name in index_names
}
print("Reference values:", ref_values)

# 4. For each pair of indices, fix them to the reference value and plot a heatmap for the other two
# There are 6 pairs in total (choose 2 fixed indices out of 4)
fixed_pairs = list(itertools.combinations(index_names, 2))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, fixed_pair in zip(axes, fixed_pairs):
    # The free (varying) indices are the ones not in the fixed_pair
    free_pair = [name for name in index_names if name not in fixed_pair]

    # Build a condition: for each fixed index, its value must equal the reference value
    condition = np.ones(len(means), dtype=bool)
    for name in fixed_pair:
        condition &= (
            means.index.get_level_values(name).astype(float) == ref_values[name]
        )

    # Filter the DataFrame
    subdf = means[condition]

    # Reset index for pivoting
    subdf_reset = subdf.reset_index()
    # Pivot: rows = first free variable, columns = second free variable, values = max_train_acc
    pivot_table = subdf_reset.pivot(
        index=free_pair[0], columns=free_pair[1], values="max_train_acc"
    )

    # Plot the heatmap
    im = ax.imshow(pivot_table, aspect="auto", origin="lower", interpolation="none")
    ax.set_title(
        f"Fixed: {fixed_pair[0]}={ref_values[fixed_pair[0]]}, {fixed_pair[1]}={ref_values[fixed_pair[1]]}\nFree: {free_pair[0]} vs {free_pair[1]}\n"
    )
    ax.set_xlabel(free_pair[1])
    ax.set_ylabel(free_pair[0])

    # Set ticks to show the unique values from the pivot table
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_xticklabels(pivot_table.columns.astype(str), rotation=45)
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_yticklabels(pivot_table.index.astype(str))

    # Add a colorbar for each subplot
    fig.colorbar(im, ax=ax)

fig.suptitle("Effect of hyperparameters on max_train_acc")
plt.tight_layout()
plt.show()

# --- Step 5: Heatmaps averaging over all fixed pairs (for each free pair) ---
# For each combination: choose 2 indices to be fixed and the other 2 to vary (free).
fixed_pairs = list(itertools.combinations(index_names, 2))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, fixed_pair in zip(axes, fixed_pairs):
    free_pair = [name for name in index_names if name not in fixed_pair]
    df_reset = means.reset_index()
    group = df_reset.groupby(free_pair)["max_train_acc"].mean().reset_index()
    # Create a pivot table: rows = free_pair[0], columns = free_pair[1]
    pivot_table = group.pivot(
        index=free_pair[0], columns=free_pair[1], values="max_train_acc"
    )

    im = ax.imshow(pivot_table, aspect="auto", origin="lower", interpolation="none")
    ax.set_title(
        f"Averaged over all fixed ({fixed_pair[0]}, {fixed_pair[1]})\nFree: {free_pair[0]} vs {free_pair[1]}"
    )
    ax.set_xlabel(free_pair[1])
    ax.set_ylabel(free_pair[0])

    # Set tick labels based on the pivot table
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_xticklabels(pivot_table.columns.astype(str), rotation=45)
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_yticklabels(pivot_table.index.astype(str))

    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "heatmaps-averaged.png"))
plt.show()

# --- Step 6: Line plots fixing three variables (using reference values) ---
# For each index variable (free variable), fix the other three at their reference values.
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, free_var in enumerate(index_names):
    fixed_vars = [name for name in index_names if name != free_var]

    # Build condition: for each fixed variable, its value equals the reference value.
    condition = np.ones(len(means), dtype=bool)
    for var in fixed_vars:
        condition &= means.index.get_level_values(var).astype(float) == ref_values[var]

    subdf = means[condition]
    subdf_reset = subdf.reset_index()
    subdf_reset = subdf_reset.sort_values(by=free_var)

    ax = axes[i]
    ax.plot(
        subdf_reset[free_var].astype(float), subdf_reset["max_train_acc"], marker="o"
    )
    ax.set_title(
        f"Fixed {fixed_vars} = {[ref_values[var] for var in fixed_vars]}\nEffect of {free_var}"
    )
    ax.set_xlabel(free_var)
    ax.set_ylabel("max_train_acc")

plt.tight_layout()
plt.show()

# --- Step 7: Line plots averaging over all fixed choices for three variables ---
# For each index variable, group by that variable (averaging over all combinations of the other three)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, free_var in enumerate(index_names):
    df_reset = means.reset_index()
    group = df_reset.groupby(free_var)["max_train_acc"].mean().reset_index()
    group = group.sort_values(by=free_var)

    ax = axes[i]
    ax.plot(group[free_var].astype(float), group["max_train_acc"], marker="o")
    ax.set_title(
        f"Averaged over all fixed {[name for name in index_names if name != free_var]}\nEffect of {free_var}"
    )
    ax.set_xlabel(free_var)
    ax.set_ylabel("Average max_train_acc")

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "line-plots-averaged.png"))
plt.show()
