from typing import Dict, List

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from src.data import get_balanced_dataset

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
    :param fixed_points: dict with integer keys, each representing a different layer.
    The values are lists of numpy arrays of shape (N,), one for each input.
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


def plot_representation_similarity_among_inputs(representations, epoch, layer_skip=1):
    """
    For a fixed epoch, plot a heatmap for each layer (or every kth layer) that shows the similarity
    (1 - normalized Hamming distance) between representations of all input pairs.

    Parameters:
        representations: dict
            Dictionary with integer keys. Each value is a numpy array of shape (T, L, N). T is the number of epochs,
            L is the number of layers and N is the number of neurons per layer.
        epoch: int
            The epoch index to use.
        layer_skip: int, optional (default=1)
            Plot one every k layers.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    # Sort input keys for consistent ordering.
    input_keys = sorted(representations.keys())
    num_inputs = len(input_keys)
    # Get number of layers from any representation.
    _, L, N = representations[input_keys[0]].shape
    selected_layers = list(range(0, L, layer_skip))

    # Create a subplot for each selected layer (in one row).
    fig, axes = plt.subplots(
        1, len(selected_layers), figsize=(5 * len(selected_layers), 4)
    )
    if len(selected_layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, selected_layers):
        # Build similarity matrix (num_inputs x num_inputs)
        sim_matrix = np.zeros((num_inputs, num_inputs))
        for i, key_i in enumerate(input_keys):
            rep_i = representations[key_i][epoch, layer, :]  # vector of length N
            for j, key_j in enumerate(input_keys):
                rep_j = representations[key_j][epoch, layer, :]
                # Compute normalized Hamming distance: fraction of mismatched bits
                hamming = np.mean(rep_i != rep_j)
                sim_matrix[i, j] = 1 - hamming
        sns.heatmap(
            sim_matrix, ax=ax, cmap="seismic", vmin=0, vmax=1, cbar=(ax == axes[-1])
        )  # show colorbar only on last subplot
        ax.set_title(f"Epoch {epoch}, Layer {layer}")
        ax.set_xlabel("Input")
        ax.set_ylabel("Input")

    plt.tight_layout()
    return fig


def plot_representations_similarity_among_layers(
    representations,
    input_key=None,
    num_epochs=3,
    average_inputs=False,
):
    """
    For each selected epoch, plot a heatmap that shows, for every pair of layers, the similarity
    (1 - normalized Hamming distance) between the representations.

    If average_inputs is False, a single input is used (specified by input_key).
    If average_inputs is True, similarity is computed for each input and then averaged across all inputs.

    Parameters:
        representations: dict
            Dictionary with integer keys. Each value is a numpy array of shape (T, L, N).
        input_key: int, optional
            The key corresponding to the input to consider (only used when average_inputs is False).
        num_epochs: int, optional (default=3)
            The approximate number of epochs to sample. The function determines epoch_skip as max(1, T//num_epochs).
        average_inputs: bool, optional (default=False)
            If True, average the similarity matrices across all inputs; otherwise, use a single input.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    if average_inputs:
        # Get a list of all input keys and use one to determine the shape.
        assert input_key is None, (
            "input_key should be None when averaging across inputs."
        )
        input_keys = sorted(representations.keys())
        rep0 = representations[input_keys[0]]  # shape: (T, L, N)
    else:
        if input_key is None:
            raise ValueError(
                "input_key must be provided if not averaging across inputs."
            )
        rep0 = representations[input_key]

    T, L, N = rep0.shape
    epoch_skip = max(1, T // num_epochs)
    selected_epochs = list(range(0, T, epoch_skip))

    # Create a subplot for each selected epoch (in one row).
    fig, axes = plt.subplots(
        1, len(selected_epochs), figsize=(5 * len(selected_epochs), 4)
    )
    if len(selected_epochs) == 1:
        axes = [axes]

    for ax, epoch in zip(axes, selected_epochs):
        sim_matrix = np.zeros((L, L))
        for l in range(L):
            for m in range(L):
                if average_inputs:
                    sims = []
                    for key in input_keys:
                        rep = representations[key]  # shape: (T, L, N)
                        rep_l = rep[epoch, l, :]
                        rep_m = rep[epoch, m, :]
                        hamming = np.mean(rep_l != rep_m)
                        sims.append(1 - hamming)
                    sim_matrix[l, m] = np.mean(sims)
                else:
                    rep = representations[input_key]
                    rep_l = rep[epoch, l, :]
                    rep_m = rep[epoch, m, :]
                    hamming = np.mean(rep_l != rep_m)
                    sim_matrix[l, m] = 1 - hamming

        sns.heatmap(
            sim_matrix, ax=ax, cmap="seismic", vmin=0, vmax=1, cbar=(ax == axes[-1])
        )
        non_diagonal = ~np.eye(L, dtype=bool) if L > 1 else np.ones((L, L), dtype=bool)
        ax.set_title(
            f"Epoch {epoch}. max: {np.max(sim_matrix[non_diagonal]):.2f}, avg: {np.mean(sim_matrix[non_diagonal]):.2f}, min: {np.min(sim_matrix[non_diagonal]):.2f}"
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")

    plt.tight_layout()
    return fig


def plot_time_series(tensor):
    """
    Plot each column of a (T, N) tensor as a separate time series in N horizontal subplots.

    Parameters:
        tensor (numpy.ndarray or torch.Tensor): 2D array of shape (T, N)
    """
    # Convert torch tensor to numpy array if needed.
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    T, N = tensor.shape
    fig, axes = plt.subplots(1, N, figsize=(5 * N, 4), squeeze=False)

    for i in range(N):
        axes[0, i].plot(tensor[:, i])
        axes[0, i].set_title(f"Series {i}")
        axes[0, i].set_xlabel("Time")
        axes[0, i].set_ylabel("Value")

    plt.tight_layout()
    plt.show()


def load_synthetic_dataset(
    N,
    P,
    C,
    p,
    eval_samples_per_class,
    rng,
    train_data_dir,
    test_data_dir,
    device,
):
    train_inputs, train_targets, train_metadata, train_class_prototypes = (
        get_balanced_dataset(
            N,
            P,
            C,
            p,
            train_data_dir,
            None,
            rng,
            shuffle=False,
            load_if_available=True,
            dump=True,
        )
    )
    eval_inputs, eval_targets, eval_metadata, eval_class_prototypes = (
        get_balanced_dataset(
            N,
            eval_samples_per_class,
            C,
            p,
            test_data_dir,
            train_class_prototypes,
            rng,
            shuffle=False,
            load_if_available=True,
            dump=True,
        )
    )
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32).to(device)
    train_targets = torch.tensor(train_targets, dtype=torch.float32).to(device)
    eval_inputs = torch.tensor(eval_inputs, dtype=torch.float32).to(device)
    eval_targets = torch.tensor(eval_targets, dtype=torch.float32).to(device)

    return (
        train_inputs,
        train_targets,
        eval_inputs,
        eval_targets,
        train_metadata,
        train_class_prototypes,
        eval_metadata,
        eval_class_prototypes,
    )
