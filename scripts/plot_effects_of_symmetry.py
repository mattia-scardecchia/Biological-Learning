import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----- 1) Load the data from disk -----
csv_path = "/Users/mat/Desktop/Files/Code/Biological-Learning/multirun/sym/summary.csv"
df = pd.read_csv(csv_path, converters={"lambda_input_skip": ast.literal_eval})
df["lambda_input_skip"] = df["lambda_input_skip"].str[0]


# ----- 2) Helper to build shortened legend labels -----
def make_label(row):
    if not row["symmetric_J_init"]:
        return "asym"
    flags = [
        row["symmetric_threshold_internal_couplings"],
        row["symmetric_update_internal_couplings"],
        row["symmetrize_internal_couplings"],
    ]
    if not any(flags):
        return "sym0"
    rule = flags.index(True) + 1
    return f"sym{rule}"


df["label"] = df.apply(make_label, axis=1)

# ----- 3) Define desired bar order and mapping text -----
order = ["asym", "sym0", "sym1", "sym2", "sym3"]
mapping_text = (
    "asym : asymmetric init, asymmetric update\n"
    "sym0 : symmetric init, asymmetric update\n"
    "sym1 : symmetric init, conditioned symmetric update\n"
    "sym2 : symmetric init, symmetric update\n"
    "sym3 : symmetric init, asymmetric update + forceful symmetrization\n"
)

# ----- 4) Ensure output directory exists -----
out_dir = "./figures"
os.makedirs(out_dir, exist_ok=True)

# ----- 5) Plotting per J_D with ordered bars, grids, zoomed y-range, and saving -----
for j_d in sorted(df["J_D"].unique()):
    sub = df[df["J_D"] == j_d]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    fig.suptitle(
        f"One tenth of Entangled MNIST (N=100). Single hidden layer, J_D = {j_d}",
        fontsize=16,
    )

    for i, double in enumerate([False, True]):
        for j, lam in enumerate([1.0, 5.0]):
            ax = axes[i, j]
            block = (
                sub[
                    (sub["double_dynamics"] == double)
                    & (sub["lambda_input_skip"] == lam)
                ]
                .set_index("label")
                .reindex(order)
            )

            train_vals = block["max_train_acc"].values
            eval_vals = block["max_eval_acc"].values
            x = np.arange(len(order))
            width = 0.35

            ax.bar(x - width / 2, train_vals, width, label="train")
            ax.bar(x + width / 2, eval_vals, width, label="eval")

            # zoom y-axis to data ±10%
            all_vals = np.concatenate([train_vals, eval_vals])
            vmin, vmax = all_vals.min(), all_vals.max()
            margin = (vmax - vmin) * 0.1
            ax.set_ylim(vmin - margin, vmax + margin)

            ax.set_xticks(x)
            ax.set_xticklabels(order, rotation=45, ha="right")
            ax.set_title(f"double_dynamics={double}, $λ_x$={lam}")
            ax.grid(True, axis="y", linestyle="--", alpha=0.6)
            ax.legend()

    # reserve bottom space for mapping text
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    # add the mapping box below the subplots
    fig.text(
        0.36,
        0.1,
        mapping_text,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"),
    )

    # save figure
    fname = os.path.join(out_dir, f"JD_{j_d:.1f}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot for J_D={j_d} to {fname}")
