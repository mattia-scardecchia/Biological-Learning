import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# base directory of experiments
base_dir = "experiments/JD0"
out_dir = "figures/JD0"
os.makedirs(out_dir, exist_ok=True)

# collect (lambda_wback, csv_path) entries by walking the tree
csv_entries = []
for root, _, files in os.walk(base_dir):
    if "grid_search_results.csv" in files:
        m = re.search(r"lambda_wback=([0-9\.]+)", root)
        if m:
            lam = float(m.group(1))
            csv_entries.append((lam, os.path.join(root, "grid_search_results.csv")))
# sort by lambda_wback
csv_entries.sort(key=lambda x: x[0])

# loop over both modes
for mode in ("final", "max"):
    # plotting setup: 2 rows (train/eval), one column per lambda
    n_cols = len(csv_entries)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))

    for col, (lam, csv_path) in enumerate(csv_entries):
        df = pd.read_csv(csv_path)
        train_col = f"{mode}_train_acc"
        eval_col = f"{mode}_eval_acc"

        # pivot to matrix
        pivot_train = df.pivot(
            index="lr_J", columns="threshold_hidden", values=train_col
        )
        pivot_eval = df.pivot(index="lr_J", columns="threshold_hidden", values=eval_col)
        lambda_wback_string = r"$\lambda_\mathrm{wback}$"

        # top: train heatmap
        ax_t = axes[0, col]
        sns.heatmap(pivot_train, ax=ax_t, cbar=(col == n_cols - 1), vmin=0, vmax=1)
        ax_t.scatter(df["threshold_hidden"], df["lr_J"], color="white", s=20)
        ax_t.set_title(f"{mode} train acc ({lambda_wback_string}={lam})")
        if col == 0:
            ax_t.set_ylabel("lr")
        ax_t.set_xlabel("threshold")

        # bottom: eval
        ax_e = axes[1, col]
        sns.heatmap(pivot_eval, ax=ax_e, cbar=(col == n_cols - 1), vmin=0, vmax=1)
        ax_e.scatter(df["threshold_hidden"], df["lr_J"], color="white", s=20)
        ax_e.set_title(f"{mode} eval acc ({lambda_wback_string}={lam})")
        if col == 0:
            ax_e.set_ylabel("lr")
        ax_e.set_xlabel("threshold")

    # layout and save
    fig.suptitle("Entangled MNIST (N=100), single hidden layer, $J_D$ = 0")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"grid_search_acc_{mode}.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")
