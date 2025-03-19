import itertools
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------- Configuration --------------------
BASE_DIR = "data/grid-search-18-03/sparsity 1.0"
THRESHOLD_VALUE = 0.5
ACCURACY_TYPE = "max_eval_acc"
HIST_BINS = 10


# -------------------- Helper Functions --------------------
def create_subplots(n, figsize_per_plot=(5, 4)):
    """Create subplots dynamically based on n plots."""
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    return fig, axes


# -------------------- Data Ingestion & Processing --------------------
def load_data(base_dir: str):
    """Load CSV files from folders and return concatenated DataFrame with a multi-index."""
    folders = [
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
    ]
    dfs = [
        pd.read_csv(os.path.join(base_dir, folder, "grid_search_results.csv"))
        for folder in folders
    ]
    # Determine index columns as those not containing 'acc' and not in these lambda columns
    index_columns = [
        c
        for c in dfs[0].columns
        if "acc" not in c and c not in ["lambda_right", "lambda_y"]
    ]
    df = pd.concat(dfs, axis=0).set_index(index_columns)
    df = df.drop(columns=["lambda_right", "lambda_y"])
    return df, index_columns


def compute_stats(df: pd.DataFrame, index_columns: list):
    """Compute mean and std over the grouped data and sort by ACCURACY_TYPE."""
    means = df.groupby(index_columns).mean()
    stds = df.groupby(index_columns).std()
    return means, stds


def filter_by_threshold(means: pd.DataFrame, threshold: float, accuracy_type: str):
    """Return the subset of data with the specified accuracy greater than threshold."""
    return means[means[accuracy_type] > threshold]


# -------------------- Visualization Functions --------------------
def plot_histograms(
    filtered: pd.DataFrame, index_names: list, base_dir: str, threshold: float
):
    """Plot histograms for each index variable in the filtered data."""
    n = len(index_names)
    fig, axes = create_subplots(n, figsize_per_plot=(5, 4))
    for i, name in enumerate(index_names):
        vals = filtered.index.get_level_values(name).astype(float)
        axes[i].hist(vals, bins=HIST_BINS, edgecolor="black")
        axes[i].set_title(f"Histogram of {name}")
    # Hide unused subplots if any
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.suptitle(f"Histograms of Index Variables | {ACCURACY_TYPE} > {threshold}")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "histograms-filtered.png"))
    plt.close()


def get_reference_values(filtered: pd.DataFrame, index_names: list):
    """Calculate the median of each index variable from filtered data."""
    ref = {}
    for name in index_names:
        vals = np.sort(filtered.index.get_level_values(name).astype(float))
        ref[name] = vals[len(vals) // 2]  # Upper median
    return ref


def plot_heatmaps_with_fixed(means: pd.DataFrame, ref_values: dict, index_names: list):
    """
    For an arbitrary number of index variables, fix all but two (using provided reference values)
    and plot a heatmap for the remaining free pair.
    """
    n = len(index_names)
    if n < 2:
        print("Not enough indices to create a heatmap.")
        return
    # For each combination of fixed variables of length (n-2), the remaining two are free.
    fixed_combinations = list(itertools.combinations(index_names, n - 2))
    fig, axes = create_subplots(len(fixed_combinations), figsize_per_plot=(5, 4))
    for ax, fixed_vars in zip(axes, fixed_combinations):
        free_vars = [name for name in index_names if name not in fixed_vars]
        # Expect free_vars to have exactly 2 elements
        if len(free_vars) != 2:
            continue
        condition = np.ones(len(means), dtype=bool)
        for var in fixed_vars:
            condition &= (
                means.index.get_level_values(var).astype(float) == ref_values[var]
            )
        subdf = means[condition]
        pivot_table = subdf.reset_index().pivot(
            index=free_vars[0], columns=free_vars[1], values=ACCURACY_TYPE
        )
        im = ax.imshow(pivot_table, aspect="auto", origin="lower", interpolation="none")
        fixed_str = ", ".join(f"{var}={ref_values[var]}" for var in fixed_vars)
        ax.set_title(f"{fixed_str}\nFree: {free_vars[0]} vs {free_vars[1]}")
        ax.set_xlabel(free_vars[1])
        ax.set_ylabel(free_vars[0])
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns.astype(str), rotation=45)
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index.astype(str))
        fig.colorbar(im, ax=ax)
    plt.suptitle(
        f"{ACCURACY_TYPE} having fixed all other variables to a decent choice | {ACCURACY_TYPE} > {THRESHOLD_VALUE}"
    )
    plt.tight_layout()
    plt.close()


def plot_heatmaps_averaged(means: pd.DataFrame, index_names: list, base_dir: str):
    """
    For each combination of two indices (free pair), average over all the remaining variables
    and plot the resulting heatmap.
    """
    free_pairs = list(itertools.combinations(index_names, 2))
    fig, axes = create_subplots(len(free_pairs), figsize_per_plot=(5, 4))
    for ax, free_pair in zip(axes, free_pairs):
        fixed_vars = [name for name in index_names if name not in free_pair]
        df_reset = means.reset_index()
        group = df_reset.groupby(list(free_pair))[ACCURACY_TYPE].mean().reset_index()
        pivot_table = group.pivot(
            index=free_pair[0], columns=free_pair[1], values=ACCURACY_TYPE
        )
        im = ax.imshow(pivot_table, aspect="auto", origin="lower", interpolation="none")
        ax.set_title(f"{free_pair[0]} vs {free_pair[1]}")
        ax.set_xlabel(free_pair[1])
        ax.set_ylabel(free_pair[0])
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns.astype(str), rotation=45)
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index.astype(str))
        fig.colorbar(im, ax=ax)
    fig.suptitle(
        f"{ACCURACY_TYPE} averaged over all values of all other variables | {ACCURACY_TYPE} > {THRESHOLD_VALUE}"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "heatmaps-averaged.png"))
    plt.close()


def plot_line_plots_fixed(means: pd.DataFrame, ref_values: dict, index_names: list):
    """
    For each index variable, plot a line graph showing the effect of that variable,
    while fixing the other variables at their reference values.
    """
    n = len(index_names)
    fig, axes = create_subplots(n, figsize_per_plot=(6, 4))
    for i, free_var in enumerate(index_names):
        fixed_vars = [name for name in index_names if name != free_var]
        condition = np.ones(len(means), dtype=bool)
        for var in fixed_vars:
            condition &= (
                means.index.get_level_values(var).astype(float) == ref_values[var]
            )
        subdf = means[condition]
        subdf_reset = subdf.reset_index().sort_values(by=free_var)
        ax = axes[i]
        ax.plot(
            subdf_reset[free_var].astype(float), subdf_reset[ACCURACY_TYPE], marker="o"
        )
        ax.set_title(
            f"{fixed_vars} = {[ref_values[var] for var in fixed_vars]}\nEffect of {free_var}"
        )
        ax.set_xlabel(free_var)
        ax.set_ylabel(ACCURACY_TYPE)
    plt.suptitle(
        f"{ACCURACY_TYPE} having fixed all other variables to a decent choice | {ACCURACY_TYPE} > {THRESHOLD_VALUE}"
    )
    plt.tight_layout()
    plt.close()


def plot_line_plots_averaged(means: pd.DataFrame, index_names: list, base_dir: str):
    """
    For each index variable, plot a line graph showing the effect of that variable,
    averaging over all other indices.
    """
    n = len(index_names)
    fig, axes = create_subplots(n, figsize_per_plot=(6, 4))
    for i, free_var in enumerate(index_names):
        df_reset = means.reset_index()
        group = (
            df_reset.groupby(free_var)[ACCURACY_TYPE]
            .mean()
            .reset_index()
            .sort_values(by=free_var)
        )
        ax = axes[i]
        ax.plot(group[free_var].astype(float), group[ACCURACY_TYPE], marker="o")
        other_vars = [name for name in index_names if name != free_var]
        ax.set_title(f"Effect of {free_var}")
        ax.set_xlabel(free_var)
        ax.set_ylabel(f"Average {ACCURACY_TYPE}")
    plt.suptitle(
        f"{ACCURACY_TYPE} averaged over all values of all other variables | {ACCURACY_TYPE} > {THRESHOLD_VALUE}"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "line-plots-averaged.png"))
    plt.close()


# -------------------- Main Routine --------------------
def main():
    # Load and process data
    df, index_names = load_data(BASE_DIR)
    means, _ = compute_stats(df, index_names)
    means.sort_values(ACCURACY_TYPE, ascending=False, inplace=True)
    means.to_csv(os.path.join(BASE_DIR, "sorted_means.csv"))

    # Filter data based on threshold
    filtered = filter_by_threshold(means, THRESHOLD_VALUE, ACCURACY_TYPE)

    # Plot histograms for filtered data
    plot_histograms(filtered, index_names, BASE_DIR, THRESHOLD_VALUE)

    # Compute reference values from filtered data
    ref_values = get_reference_values(filtered, index_names)
    print("Reference values:", ref_values)

    # Plot heatmaps with fixed variables (fixing all but 2 at ref values)
    plot_heatmaps_with_fixed(filtered, ref_values, index_names)

    # Plot heatmaps averaged over all but two variables
    plot_heatmaps_averaged(filtered, index_names, BASE_DIR)

    # Plot line plots with other variables fixed at reference values
    plot_line_plots_fixed(filtered, ref_values, index_names)

    # Plot line plots averaging over the other variables
    plot_line_plots_averaged(filtered, index_names, BASE_DIR)


if __name__ == "__main__":
    main()
