import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
data_dir = os.path.join(
    "experiments", "lambda_advantage", "lambda", "2025-07-08-01-57-24"
)
out_dir = os.path.join("figures", "lambda_advantage")
os.makedirs(out_dir, exist_ok=True)

# Load CSVs for each lambda_input_skip value
dfs = []
for entry in os.listdir(data_dir):
    path = os.path.join(data_dir, entry)
    if os.path.isdir(path) and "lambda_input_skip=" in entry:
        m = re.search(r"lambda_input_skip=([0-9.]+)", entry)
        if not m:
            continue
        skip_val = float(m.group(1))
        csv_path = os.path.join(path, "grid_search_results.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            df["lambda_input_skip"] = skip_val
            dfs.append(df)

# Combine data
all_df = pd.concat(dfs, ignore_index=True)


# Plotting for 'max' and 'final' accuracies
def compute_stats(df, metric_prefix):
    # Group by lambda_l and compute mean and SEM across seeds
    stats = df.groupby("lambda_l")[f"{metric_prefix}_acc"].agg(
        ["mean", lambda x: x.sem()]
    )
    stats.columns = ["mean", "sem"]
    return stats


for final_or_max in ["max", "final"]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for skip_val, color in zip(
        sorted(all_df["lambda_input_skip"].unique()), ["tab:blue", "tab:orange"]
    ):
        subset = all_df[all_df["lambda_input_skip"] == skip_val]
        # Training stats
        train_stats = compute_stats(subset, f"{final_or_max}_train")
        # Evaluation stats
        eval_stats = compute_stats(subset, f"{final_or_max}_eval")

        # Plot mean with SEM as error bars
        axes[0].errorbar(
            train_stats.index,
            train_stats["mean"],
            yerr=train_stats["sem"],
            label=f"$\\lambda_x={skip_val}$",
            color=color,
            marker="o",
        )
        axes[1].errorbar(
            eval_stats.index,
            eval_stats["mean"],
            yerr=eval_stats["sem"],
            label=f"$\\lambda_x={skip_val}$",
            color=color,
            marker="o",
        )

    # Labels and legends
    axes[0].set_title(f"{final_or_max.capitalize()} Train Accuracy vs $\lambda$")
    axes[0].set_xlabel("$\lambda$")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
    axes[0].grid(which="both")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].set_title(f"{final_or_max.capitalize()} Eval Accuracy vs $\lambda$")
    axes[1].set_xlabel("$\lambda$")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
    axes[1].grid(which="both")
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    # Save figure
    out_path = os.path.join(out_dir, f"{final_or_max}_accuracy.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

print("Plots saved to figures/lambda_advantage")
