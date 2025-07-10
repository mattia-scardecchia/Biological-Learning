import json
import os
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def is_value(s):
    """Return True if s represents a boolean, int, or float."""
    if s in ("True", "False"):
        return True
    if re.fullmatch(r"-?\d+", s):
        return True
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return True
    return False


def parse_value(s):
    """Convert s to bool, int, float, or fallback to string."""
    if s == "True":
        return True
    if s == "False":
        return False
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    return s


def parse_folder_name(name):
    """
    Given a folder name like:
      plots_J_D_0.0_lambda_input_skip_2.0_double_dynamics_False_...
    return a dict of config parameters.
    """
    parts = name.split("_")[1:]  # skip 'plots'
    cfg = {}
    i = 0
    while i < len(parts):
        # collect key parts until a value is reached
        key_parts = []
        while i < len(parts) and not is_value(parts[i]):
            key_parts.append(parts[i])
            i += 1
        key = "_".join(key_parts)
        if i < len(parts):
            value = parse_value(parts[i])
            i += 1
        else:
            value = None
        cfg[key] = value
    return cfg


# Root directory containing all 'plots_...' folders
root_dir = Path("experiments/sym_evolution/2025-07-03-10-48-06")

rows = []
for folder in root_dir.iterdir():
    if folder.is_dir() and folder.name.startswith("plots_"):
        stats_path = folder / "stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
            # Extract symmetricity_history
            sym_hist = stats.get("symmetricity_history", [])
            # Parse config from folder name
            cfg = parse_folder_name(folder.name)
            # Attach the time series
            cfg["sym-hist"] = sym_hist
            rows.append(cfg)

print(f"Found {len(rows)} configurations.")

if rows:
    print("Example entry:")
    print(json.dumps(rows[0], indent=2))

# 1) Group by config (excluding seed) and compute mean and SEM
grouped = {}
for cfg in rows:
    key = (
        cfg["J_D"],
        cfg["lambda_input_skip"],
        cfg["double_dynamics"],
        cfg["symmetric_J_init"],
    )
    grouped.setdefault(key, []).append(cfg["sym-hist"])

stats_rows = []
for (J_D, lam, dd, sym_init), hists in grouped.items():
    # Convert to array: shape (n_runs, time)
    data = np.array([[h[t][0] for t in range(len(h))] for h in hists])
    mean_hist = np.mean(data, axis=0)
    std_hist = np.std(data, axis=0)
    sem_hist = std_hist / np.sqrt(data.shape[0])
    stats_rows.append(
        {
            "J_D": J_D,
            "lambda_input_skip": lam,
            "double_dynamics": dd,
            "symmetric_J_init": sym_init,
            "mean_hist": mean_hist,
            "sem_hist": sem_hist,
        }
    )

# 2) Plot per double_dynamics with SEM bands
colors = {2.0: "blue", 5.0: "orange"}
linestyles = {0.0: "-", 0.5: "--"}
for dd in [False, True]:
    sym_inits = sorted(
        {r["symmetric_J_init"] for r in stats_rows if r["double_dynamics"] == dd}
    )
    fig, axes = plt.subplots(
        nrows=1, ncols=len(sym_inits), sharey=True, figsize=(16, 8)
    )
    if len(sym_inits) == 1:
        axes = [axes]
    for ax, sym_init in zip(axes, sym_inits):
        for r in stats_rows:
            if r["double_dynamics"] != dd or r["symmetric_J_init"] != sym_init:
                continue
            x = np.arange(len(r["mean_hist"]))
            mean = r["mean_hist"]
            sem = r["sem_hist"]
            label = f"λ={r['lambda_input_skip']}, J={r['J_D']}"
            ax.plot(
                x,
                mean,
                label=label,
                color=colors[r["lambda_input_skip"]],
                linestyle=linestyles[r["J_D"]],
            )
            ax.fill_between(x, mean - sem, mean + sem, alpha=0.3)
        ax.set_title(f"symmetric_J_init = {sym_init}")
        ax.set_ylabel("Symmetricity")
        ax.legend()
        ax.grid()
        ax.set_xlabel("Epoch")
    fig.suptitle(
        f"Entangle MNIST (N=100). Single hidden layer,\ndouble_dynamics = {dd}", y=0.92
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.join("figures", "symmetricity_evolution"), exist_ok=True)
    fig.savefig(
        os.path.join("figures", "symmetricity_evolution", f"dd_{dd}.png"),
        dpi=300,
    )
    plt.close(fig)
