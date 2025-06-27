import math
from typing import Dict, List

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

DTYPE = torch.float32


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
        non_diagonal_mask = ~np.eye(num_inputs, dtype=bool)
        avg_sim_off_diagonal = np.mean(sim_matrix[non_diagonal_mask])
        ax.set_title(f"Epoch {epoch}, Layer {layer}. Avg: {avg_sim_off_diagonal:.2f}")
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
        assert (
            input_key is None
        ), "input_key should be None when averaging across inputs."
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


def create_subplots(n_subplots, row_height=3, col_width=8):
    """
    Create subplots in a grid with two rows (if n_subplots > 1).
    """
    rows = 2 if n_subplots > 1 else 1
    cols = math.ceil(n_subplots / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(col_width * cols, row_height * rows))
    if n_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    return fig, axes


def plot_couplings_histograms(logs, time_indexes, bins=30):
    """
    Plot histograms for the evolution of coupling distributions at specified time indexes.

    Parameters:
        logs (dict): Dictionary with the following keys and expected numpy array shapes:
            - "W_forth": np.array of shape (T, C, N)
            - "W_back": np.array of shape (T, N, C)
            - "internal_couplings": np.array of shape (T, L, N, N)
            - "left_couplings": np.array of shape (T, L, N, N)
            - "right_couplings": np.array of shape (T, L, N, N)
        time_indexes (list of int): List of time step indices at which to plot histograms.
        bins (int, optional): Number of bins for the histogram plots (default is 50).

    Legend for shapes:
        T: Number of time steps.
        L: Number of layers.
        N: Number of neurons per layer.
        C: Number of classes.

    Behavior:
        - Creates one figure for "internal_couplings" with one subplot per layer.
        - Creates one figure for "left_couplings" combined with "W_forth":
              one subplot per layer for "left_couplings" and an extra subplot for "W_forth".
        - Creates one figure for "right_couplings" combined with "W_back":
              one subplot per layer for "right_couplings" and an extra subplot for "W_back".
        - In each subplot, histograms are plotted for the given time indexes with a legend indicating the time.
    """
    figs = {}

    # 1. Internal Couplings
    internal = logs["internal_couplings"]  # shape: (T, L, N, N)
    T, L, _, _ = internal.shape
    fig_int, axes_int = create_subplots(L)
    for l in range(L):
        ax = axes_int[l]
        for t in time_indexes:
            data = internal[t, l].flatten()
            ax.hist(data, bins=bins, alpha=0.3, label=f"t={t}", density=True)
        ax.set_title(f"Layer {l}")
        ax.legend()
        ax.grid(True)
    fig_int.suptitle("Internal Couplings (density)")
    fig_int.tight_layout()
    figs["internal"] = fig_int

    # 2. Left Couplings + W_forth
    left = logs["left_couplings"]  # shape: (T, L-1, N, N)
    W_forth = logs["W_forth"]  # shape: (T, C, N)
    N = left.shape[-1]
    non_diagonal = ~np.eye(N, dtype=bool)
    fig_left, axes_left = create_subplots(L)
    for l in range(L - 1):
        ax = axes_left[l]
        for t in time_indexes:
            data = left[t, l][non_diagonal].flatten()
            ax.hist(data, bins=bins, alpha=0.3, label=f"t={t}", density=True)
        ax.set_title(f"Layer {l + 1}")
        ax.legend()
        ax.grid(True)
    ax = axes_left[L - 1]
    for t in time_indexes:
        data = W_forth[t].flatten()
        ax.hist(data, bins=bins, alpha=0.3, label=f"t={t}", density=True)
    ax.set_title("Readout layer (W_forth)")
    ax.legend()
    ax.grid(True)
    fig_left.suptitle("Left Couplings (density)")
    fig_left.tight_layout()
    figs["left"] = fig_left

    # 3. Right Couplings + W_back
    right = logs["right_couplings"]  # shape: (T, L-1, N, N)
    W_back = logs["W_back"]  # shape: (T, N, C)
    fig_right, axes_right = create_subplots(L)
    for l in range(L - 1):
        ax = axes_right[l]
        for t in time_indexes:
            data = right[t, l].flatten()
            ax.hist(data, bins=bins, alpha=0.3, label=f"t={t}", density=True)
        ax.set_title(f"Layer {l}")
        ax.legend()
        ax.grid(True)
    ax = axes_right[L - 1]
    for t in time_indexes:
        data = W_back[t].flatten()
        ax.hist(data, bins=bins, alpha=0.3, label=f"t={t}", density=True)
    ax.set_title(f"Layer {L} (W_back)")
    ax.legend()
    ax.grid(True)
    fig_right.suptitle("Right Couplings (density)")
    fig_right.tight_layout()
    figs["right"] = fig_right

    return figs


def plot_couplings_distro_evolution(logs):
    """
    Plot the evolution over time of the mean and standard deviation for each coupling distribution.

    Parameters:
        logs (dict): Dictionary with the following keys and expected numpy array shapes:
            - "W_forth": np.array of shape (T, C, N)
            - "W_back": np.array of shape (T, N, C)
            - "internal_couplings": np.array of shape (T, L, N, N)
            - "left_couplings": np.array of shape (T, L, N, N)
            - "right_couplings": np.array of shape (T, L, N, N)

    Legend for shapes:
        T: Number of time steps.
        L: Number of layers.
        N: Number of neurons per layer.
        C: Number of classes.

    Behavior:
        - For each key, computes the mean and standard deviation across all dimensions except the time dimension.
        - For "internal_couplings": Plots one subplot per layer showing the evolution (over time) of mean with error bars representing std.
        - For "left_couplings" and "W_forth": Plots one subplot per layer for left couplings and an extra subplot for W_forth.
        - For "right_couplings" and "W_back": Plots one subplot per layer for right couplings and an extra subplot for W_back.
    """
    figs = {}

    # 1. Internal Couplings Evolution
    internal = logs["internal_couplings"]  # shape: (T, L, N, N)
    T, L, _, _ = internal.shape
    fig_int, axes_int = create_subplots(L)
    for l in range(L):
        means, stds = [], []
        for t in range(T):
            data = internal[t, l].flatten()
            means.append(np.mean(data))
            stds.append(np.std(data))
        ax = axes_int[l]
        ax.errorbar(np.arange(T), means, yerr=stds, fmt="-o")
        ax.set_title(f"Layer {l}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
    fig_int.suptitle("Internal Couplings - Evolution of Mean and Std")
    fig_int.tight_layout()
    figs["internal"] = fig_int

    # 2. Left Couplings + W_forth Evolution
    left = logs["left_couplings"]  # shape: (T, L-1, N, N)
    W_forth = logs["W_forth"]  # shape: (T, C, N)
    fig_left, axes_left = create_subplots(L)
    keep_mask = ~np.eye(
        left.shape[-1], dtype=bool
    )  # exclude diagonal cause ferromagnetic couplings are big and ruin the stddev
    for l in range(L - 1):
        means, stds = [], []
        for t in range(T):
            data = left[t, l][keep_mask].flatten()
            means.append(np.mean(data))
            stds.append(np.std(data))
        ax = axes_left[l]
        ax.errorbar(np.arange(T), means, yerr=stds, fmt="-o")
        ax.set_title(f"Layer {l + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
    means, stds = [], []
    for t in range(T):
        data = W_forth[t].flatten()
        means.append(np.mean(data))
        stds.append(np.std(data))
    ax = axes_left[L - 1]
    ax.errorbar(np.arange(T), means, yerr=stds, fmt="-o")
    ax.set_title("W_forth Evolution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True)
    fig_left.suptitle("Left Couplings - Evolution of Mean and Std")
    fig_left.tight_layout()
    figs["left"] = fig_left

    # 3. Right Couplings + W_back Evolution
    right = logs["right_couplings"]  # shape: (T, L-1, N, N)
    W_back = logs["W_back"]  # shape: (T, N, C)
    fig_right, axes_right = create_subplots(L)
    for l in range(L - 1):
        means, stds = [], []
        for t in range(T):
            data = right[t, l].flatten()
            means.append(np.mean(data))
            stds.append(np.std(data))
        ax = axes_right[l]
        ax.errorbar(np.arange(T), means, yerr=stds, fmt="-o")
        ax.set_title(f"Layer {l}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
    means, stds = [], []
    for t in range(T):
        data = W_back[t].flatten()
        means.append(np.mean(data))
        stds.append(np.std(data))
    ax = axes_right[L - 1]
    ax.errorbar(np.arange(T), means, yerr=stds, fmt="-o")
    ax.set_title("W_back Evolution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True)
    fig_right.suptitle("Right Couplings - Evolution of Mean and Std")
    fig_right.tight_layout()
    figs["right"] = fig_right

    return figs


def relaxation_trajectory_double_dynamics(
    classifier, x: torch.Tensor, y: torch.Tensor, steps: List[int], state=None
):
    """
    saves the state and unsat percentage at each step in step list
    """
    states = []
    unsats = []
    overlaps = []
    dyn_steps = max(steps) // 2
    if state is None:
        state = classifier.initialize_state(x, y, "zeros")
        state_prev = state.clone()
    for step in range(dyn_steps):
        state, _, unsat = classifier.relax(
            state_prev,
            max_steps=1,
            ignore_right=0,
        )
        overlap = (state_prev[:, 1] * state[:, 1]).sum(dim=-1) / state_prev.shape[-1]
        overlaps.append(overlap)  # B
        state_prev = state
        if step + 1 in steps:
            # Store the state and unsat status
            states.append(state.clone())
            unsats.append(unsat.clone())
    for step in range(dyn_steps, dyn_steps * 2):
        state, _, unsat = classifier.relax(
            state_prev,
            max_steps=1,
            ignore_right=1,
        )
        overlap = (state_prev[:, 1] * state[:, 1]).sum(dim=-1) / state_prev.shape[-1]
        overlaps.append(overlap)  # B
        state_prev = state
        if step + 1 in steps:
            # Store the state and unsat status
            states.append(state.clone())
            unsats.append(unsat.clone())
    states = torch.stack(states, dim=0)  # T, B, L, N
    states = states.permute(1, 0, 2, 3)  # B, T, L, N
    unsats = torch.stack(unsats, dim=0)  # T, B, L, N
    unsats = unsats.permute(1, 0, 2, 3)  # B, T, L, N
    overlaps = torch.stack(overlaps, dim=0)  # T', B
    return states, unsats, overlaps
