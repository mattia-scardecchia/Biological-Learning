from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt

DTYPE = np.float32


def sign(x: float):
    return 2 * int(x > 0) - 1


def theta(x: float):
    return int(x > 0)


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
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid()
    global_avg = np.mean(list(accuracy_by_class.values()))
    ax.set_title(f"Accuracy by class (Global Avg: {global_avg:.2f})")
    return fig


def plot_accuracy_history(train_acc_history, eval_acc_history=None, eval_epochs=None):
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    train_epochs = np.arange(1, len(train_acc_history) + 1)
    ax.plot(train_epochs, train_acc_history, label="Train")
    if eval_acc_history is not None:
        assert eval_epochs is not None
        ax.plot(eval_epochs, eval_acc_history, label="Eval")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Evolution of accuracy during training")
    ax.grid()
    ax.legend()
    return fig
