import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------- Configuration --------------------
BASE_DIR = "data/grid-search-03-03"
THRESHOLD_VALUE = 0.10
ACCURACY_TYPE = "max_train_acc"
HIST_BINS = 10


# -------------------- Data Ingestion & Processing --------------------
def load_data(base_dir: str):
    """Load CSV files from folders and return concatenated DataFrame with a multi-index."""
    folders = [
        folder
        for folder in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, folder))
    ]
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
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for i, name in enumerate(index_names):
        vals = filtered.index.get_level_values(name).astype(float)
        axs[i].hist(vals, bins=HIST_BINS, edgecolor="black")
        axs[i].set_title(f"Histogram of {name}")

    plt.suptitle(f"Histograms of Index Variables | {ACCURACY_TYPE} > {threshold}")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"histograms-filtered{threshold}.png"))
    plt.show()


def get_reference_values(filtered: pd.DataFrame, index_names: list):
    """Calculate the median of each index variable from filtered data."""
    out = {}
    for name in index_names:
        sorted_vals = np.sort(filtered.index.get_level_values(name).astype(float))
        out[name] = sorted_vals[len(sorted_vals) // 2]  # Upper median
    return out


def plot_heatmaps_with_fixed_pairs(
    means: pd.DataFrame, ref_values: dict, index_names: list
):
    """Plot heatmaps for each fixed pair of indices using the reference values."""
    fixed_pairs = list(itertools.combinations(index_names, 2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, fixed_pair in zip(axes, fixed_pairs):
        free_pair = [name for name in index_names if name not in fixed_pair]
        condition = np.ones(len(means), dtype=bool)
        for name in fixed_pair:
            condition &= (
                means.index.get_level_values(name).astype(float) == ref_values[name]
            )
        subdf = means[condition]
        pivot_table = subdf.reset_index().pivot(
            index=free_pair[0], columns=free_pair[1], values=ACCURACY_TYPE
        )
        im = ax.imshow(pivot_table, aspect="auto", origin="lower", interpolation="none")
        ax.set_title(
            f"Fixed: {fixed_pair[0]}={ref_values[fixed_pair[0]]}, "
            f"{fixed_pair[1]}={ref_values[fixed_pair[1]]}\nFree: {free_pair[0]} vs {free_pair[1]}"
        )
        ax.set_xlabel(free_pair[1])
        ax.set_ylabel(free_pair[0])
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns.astype(str), rotation=45)
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index.astype(str))
        fig.colorbar(im, ax=ax)

    plt.suptitle(f"Heatmaps with Fixed Pairs | {ACCURACY_TYPE} > {THRESHOLD_VALUE}")
    plt.tight_layout()
    plt.show()


def plot_heatmaps_averaged(means: pd.DataFrame, index_names: list, base_dir: str):
    """Plot heatmaps averaged over all fixed pairs."""
    fixed_pairs = list(itertools.combinations(index_names, 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for ax, fixed_pair in zip(axes, fixed_pairs):
        free_pair = [name for name in index_names if name not in fixed_pair]
        df_reset = means.reset_index()
        group = df_reset.groupby(free_pair)[ACCURACY_TYPE].mean().reset_index()
        pivot_table = group.pivot(
            index=free_pair[0], columns=free_pair[1], values=ACCURACY_TYPE
        )
        im = ax.imshow(pivot_table, aspect="auto", origin="lower", interpolation="none")
        ax.set_title(
            f"Averaged over all values of ({fixed_pair[0]}, {fixed_pair[1]})\nFree: {free_pair[0]} vs {free_pair[1]}"
        )
        ax.set_xlabel(free_pair[1])
        ax.set_ylabel(free_pair[0])
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns.astype(str), rotation=45)
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index.astype(str))
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "heatmaps-averaged.png"))
    plt.show()


def plot_line_plots_fixed(means: pd.DataFrame, ref_values: dict, index_names: list):
    """For each index variable, plot a line graph with the other three fixed at reference values."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

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
            f"Fixed {fixed_vars} = {[float(ref_values[var]) for var in fixed_vars]}\nEffect of {free_var}"
        )
        ax.set_xlabel(free_var)
        ax.set_ylabel(ACCURACY_TYPE)

    plt.suptitle(
        f"Line Plots with Fixed Variables | {ACCURACY_TYPE} > {THRESHOLD_VALUE}"
    )
    plt.tight_layout()
    plt.show()


def plot_line_plots_averaged(means: pd.DataFrame, index_names: list, base_dir: str):
    """For each index variable, plot a line graph averaging over the other three indices."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

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
        ax.set_title(
            f"Averaged over {[name for name in index_names if name != free_var]}\nEffect of {free_var}"
        )
        ax.set_xlabel(free_var)
        ax.set_ylabel(f"Average {ACCURACY_TYPE}")

    plt.suptitle(
        f"Line Plots Averaged Over Variables | {ACCURACY_TYPE} > {THRESHOLD_VALUE}"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "line-plots-averaged.png"))
    plt.show()


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

    # Plot heatmaps for fixed pairs
    plot_heatmaps_with_fixed_pairs(means, ref_values, index_names)

    # Plot heatmaps averaged over fixed pairs
    plot_heatmaps_averaged(means, index_names, BASE_DIR)

    # Plot line plots with three variables fixed at reference values
    plot_line_plots_fixed(means, ref_values, index_names)

    # Plot line plots averaging over the other three variables
    plot_line_plots_averaged(means, index_names, BASE_DIR)


if __name__ == "__main__":
    main()
