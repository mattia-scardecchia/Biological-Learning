#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# map method names to folders / init flags
METHOD_MAP = {
    "asym": ("perceptron", False),
    "sym0": ("perceptron", True),
    "sym1": ("sym1", None),
    "sym2": ("sym2", None),
    "sym3": ("sym3", None),
}

JVALS = [0.0, 0.5]


def load_method(base_dir, method, final_or_max):
    folder, want_sym = METHOD_MAP[method]
    run_dir = os.listdir(os.path.join(base_dir, folder))[-1]
    df = pd.read_csv(os.path.join(base_dir, folder, run_dir, "grid_search_results.csv"))
    if want_sym is not None:
        df = df[df["symmetric_J_init"] == want_sym]
    df["method"] = method
    return df


def compute_stats(df, final_or_max):
    stats = (
        df.groupby(["J_D", "lambda_input_skip", "double_dynamics", "method"])
        .agg(
            train_mean=(f"{final_or_max}_train_acc", "mean"),
            train_std=(f"{final_or_max}_train_acc", "std"),
            eval_mean=(f"{final_or_max}_eval_acc", "mean"),
            eval_std=(f"{final_or_max}_eval_acc", "std"),
            n=("seed", "count"),
        )
        .reset_index()
    )
    stats["train_sem"] = stats["train_std"] / np.sqrt(stats["n"])
    stats["eval_sem"] = stats["eval_std"] / np.sqrt(stats["n"])
    return stats


def plot_stats(stats, final_or_max, output_dir):
    lam_vals = sorted(stats["lambda_input_skip"].unique())
    methods = list(METHOD_MAP.keys())
    x = np.arange(len(methods))
    os.makedirs(output_dir, exist_ok=True)

    for dd in [False, True]:
        fig, axes = plt.subplots(
            len(JVALS), len(lam_vals), figsize=(12, 8), sharey=True
        )
        for i, Jval in enumerate(JVALS):
            for j, lam in enumerate(lam_vals):
                ax = axes[i, j]
                block = stats[
                    (stats["double_dynamics"] == dd)
                    & (stats["J_D"] == Jval)
                    & (stats["lambda_input_skip"] == lam)
                ]

                train = [block[block.method == m].train_mean.values[0] for m in methods]
                train_e = [
                    block[block.method == m].train_sem.values[0] for m in methods
                ]
                eval_ = [block[block.method == m].eval_mean.values[0] for m in methods]
                eval_e = [block[block.method == m].eval_sem.values[0] for m in methods]

                ax.bar(
                    x - 0.15, train, width=0.3, yerr=train_e, capsize=5, label="train"
                )
                ax.bar(x + 0.15, eval_, width=0.3, yerr=eval_e, capsize=5, label="eval")
                ax.set_xticks(x)
                ax.set_xticklabels(methods)
                ax.set_title(f"$J_D$={Jval}, λₓ={lam}")
                ax.set_yticks(np.arange(0, 1.0, 0.1), minor=True)
                ax.grid(axis="y", linestyle="--", alpha=0.5, which="both")

                if i == 0:
                    ax.set_xlabel("Methods")
                if j == 0:
                    ax.set_ylabel(f"{final_or_max} accuracy")

        plt.legend()
        fig.suptitle(
            f"Entangled MNIST (N=100). Single hidden layer,\ndouble_dynamics = {dd}",
            y=0.95,
        )
        mapping_text = (
            "asym : asymmetric init, asymmetric update\n"
            "sym0 : symmetric init, asymmetric update\n"
            "sym1 : symmetric init, double-conditioned symmetric update\n"
            "sym2 : symmetric init, symmetric update\n"
            "sym3 : symmetric init, asymmetric update + forceful symmetrization"
        )
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        fig.text(
            0.36,
            0.1,
            mapping_text,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"),
        )

        filename = f"{final_or_max}_dd_{dd}.png"
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)


def main(base_dir, final_or_max):
    dfs = [load_method(base_dir, m, final_or_max) for m in METHOD_MAP]
    df = pd.concat(dfs, ignore_index=True)
    stats = compute_stats(df, final_or_max)
    output_dir = os.path.join("figures", "symmetric_rules")
    plot_stats(stats, final_or_max, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate symmetric rules plots")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="experiments/symmetric_rules",
        help="Root experiments folder (default: experiments/symmetric_rules)",
    )
    args = parser.parse_args()

    for final_or_max in ["final", "max"]:
        main(args.base_dir, final_or_max)
