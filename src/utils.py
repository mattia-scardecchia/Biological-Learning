from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt


def plot_fixed_points_similarity_heatmap(
    fixed_points: Dict[int, List[np.ndarray]],
    with_flip_invariance: bool = False,
):
    """
    :param fixed_points: dict with integer keys, each containing a list of fixed points
    """
    fig, axes = plt.subplots(1, len(fixed_points), figsize=(30, 10))
    for idx, ax in zip(fixed_points, axes):
        vectors = fixed_points[idx]
        T = len(vectors)
        sims = np.zeros((T, T))
        for t in range(T):
            for s in range(T):
                sims[t, s] = np.mean(vectors[t] == vectors[s])
                if with_flip_invariance:
                    sims[t, s] = max(sims[t, s], 1 - sims[t, s])
        cax = ax.matshow(sims, cmap="seismic", vmin=0, vmax=1)
        fig.colorbar(cax, ax=ax)
        ax.set_title(f"Layer {idx}." if idx < len(fixed_points) - 1 else "Readout.")
        ax.set_xlabel("Step")
        ax.set_ylabel("Step")
    fig.suptitle(
        "Similarity heatmap between internal representations within each layer"
    )
    fig.tight_layout()
    return fig


def plot_accuracy_by_class_barplot(accuracy_by_class: Dict[int, float]):
    fig, ax = plt.subplots()
    ax.bar(list(accuracy_by_class.keys()), list(accuracy_by_class.values()))
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by class")
    return fig
