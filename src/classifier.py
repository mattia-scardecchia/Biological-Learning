import logging
import math
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

DTYPE = np.float32


def initialize_state(N: int, rng: Optional[np.random.Generator] = None):
    rng = np.random.default_rng() if rng is None else rng
    S = np.array(rng.choice([-1, 1], size=(N,)), dtype=DTYPE)
    return S


def initialize_readout_state(C: int, rng: Optional[np.random.Generator] = None):
    rng = np.random.default_rng() if rng is None else rng
    return np.zeros((C,), dtype=DTYPE)


def initialize_couplings(N: int, J_D: float, rng: Optional[np.random.Generator] = None):
    rng = np.random.default_rng() if rng is None else rng
    J = np.array(rng.normal(0, 1 / np.sqrt(N), size=(N, N)), dtype=DTYPE)
    np.fill_diagonal(J, J_D)
    return J


def initialize_readout_couplings(
    N: int, C: int, rng: Optional[np.random.Generator] = None
):
    rng = np.random.default_rng() if rng is None else rng
    W = np.array(rng.choice([-1, 1], size=(C, N)), dtype=DTYPE)
    return W


def sign(x: float):
    return 2 * int(x > 0) - 1


def theta(x: float):
    return int(x > 0)


class Classifier:
    """Holds the classifier state, and handles training and inference."""

    def __init__(
        self,
        num_layers: int,
        N: int,
        C: int,
        lambda_left: float,
        lambda_right: float,
        lambda_x: float,
        lambda_y: float,
        J_D: float,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initializes the classifier.
        :param num_layers: number of layers.
        :param N: number of neurons per layer.
        :param C: number of classes.
        :param lambda_left: strength of coupling with previous layer.
        :param lambda_right: strength of coupling with next layer.
        :param lambda_x: strength of coupling with input.
        :param lambda_y: strength of coupling with target.
        :param J_D: self-interaction strength.
        :param rng: random number generator for initialization.
        """
        self.num_layers = num_layers
        self.N = N
        self.C = C
        self.lambda_left = lambda_left
        self.lambda_right = lambda_right
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.J_D = J_D

        rng = np.random.default_rng() if rng is None else rng
        self.initialize_state(rng)
        self.initialize_couplings(rng)
        self.activations = [sign for _ in range(self.num_layers)] + [theta]
        # self.activations = [sign for _ in range(self.num_layers + 1)]

    def initialize_state(
        self, rng: Optional[np.random.Generator] = None, x: Optional[np.ndarray] = None
    ):
        """Initializes the state of the network. If x is provided, initialize all
        non-readout layers to x. Otherwise, sample."""
        rng = np.random.default_rng() if rng is None else rng
        if x is not None:
            self.layers = [x.copy() for _ in range(self.num_layers)]
        else:
            self.layers = [
                initialize_state(self.N, rng) for _ in range(self.num_layers)
            ]
        self.layers.append(
            initialize_readout_state(self.C, rng)
        )  # num_layers + 1, N (layers[-1] is the readout layer)

    def initialize_couplings(self, rng: Optional[np.random.Generator] = None):
        """Initializes the couplings of the network."""
        rng = np.random.default_rng() if rng is None else rng
        self.couplings = [
            initialize_couplings(self.N, self.J_D, rng) for _ in range(self.num_layers)
        ]  # num_layers, N, N
        self.W = initialize_readout_couplings(self.N, self.C, rng)  # C, N

    def internal_field(self, layer_idx: int, neuron_idx: int):
        """Field due to interaction within each layer."""
        if layer_idx == self.num_layers:
            return 0  # readout layer has no internal field
        return np.dot(self.couplings[layer_idx][neuron_idx, :], self.layers[layer_idx])

    def left_field(
        self, layer_idx: int, neuron_idx: int, x: Optional[np.ndarray] = None
    ):
        """Field due to interaction with previous layer, or with left external field."""
        if layer_idx == 0:
            if x is None:
                return 0
            return self.lambda_x * x[neuron_idx]
        if layer_idx == self.num_layers:
            return np.dot(
                self.W[neuron_idx, :], self.layers[self.num_layers - 1]
            ) / np.sqrt(self.N)
        return self.lambda_left * self.layers[layer_idx - 1][neuron_idx]

    def right_field(
        self, layer_idx: int, neuron_idx: int, y: Optional[np.ndarray] = None
    ):
        """Field due to interaction with next layer, or with right external field."""
        if layer_idx == self.num_layers - 1:
            return np.dot(
                self.W[:, neuron_idx], self.layers[self.num_layers]
            )  # NOTE: no multiplier because readout is sparse (?)
        if layer_idx == self.num_layers:
            if y is None:
                return 0
            return self.lambda_y * sign(y[neuron_idx])  # NOTE: assume y is one-hot
        return self.lambda_right * self.layers[layer_idx + 1][neuron_idx]

    def local_field(
        self,
        layer_idx: int,
        neuron_idx: int,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        ignore_right=False,
    ):
        """Computes the local field perceived by a neuron. \\
        :param layer_idx: layer index. Can pass y for the readout layer. \\
        :param neuron_idx: neuron index. \\
        :param x: input (left external field). If None, no left external field. \\
        :param y: one-hot target (right external field). If None, no right external field. \\
        :param ignore_right: if True, ignore the interaction from the right (next layer, or external field). \\
        :return: the local field perceived by the neuron.
        """
        internal_field = self.internal_field(layer_idx, neuron_idx)
        left_field = self.left_field(layer_idx, neuron_idx, x)
        right_field = 0 if ignore_right else self.right_field(layer_idx, neuron_idx, y)
        return internal_field + left_field + right_field

    def relax(
        self,
        max_steps: int,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Relaxes the network to a stable state. \\
        :param x: input (left external field). If None, omit it. \\
        :param y: one-hot target (right external field). If None, omit it. \\
        :param max_steps: maximum number of sweeps over the network for updates. \\
        :param rng: random number generator for update order. \\
        :return: the number of full sweeps taken to converge (or max_steps if not converged).
        """
        step, made_update = 0, True
        rng = np.random.default_rng() if rng is None else rng
        while made_update and step < max_steps:
            made_update = False
            for layer_idx in range(self.num_layers + 1):
                perm = rng.permutation(len(self.layers[layer_idx]))
                for neuron_idx in perm:
                    local_field = self.local_field(layer_idx, neuron_idx, x, y)
                    update = self.activations[layer_idx](local_field)
                    made_update = made_update or (
                        update != self.layers[layer_idx][neuron_idx]
                    )
                    self.layers[layer_idx][neuron_idx] = update
            step += 1
        return step  # NOTE: if we converge during the (max_steps - 1)th sweep, we do not detect it

    def apply_perceptron_rule(self, lr: float, threshold: float, x: np.ndarray):
        """Applies the perceptron learning rule to the network in its current state.
        :param lr: learning rate.
        :param threshold: stability threshold.
        :param x: input.
        """
        count = 0
        for layer_idx in range(self.num_layers):
            for neuron_idx in range(self.N):
                local_field = self.local_field(
                    layer_idx, neuron_idx, x=x, y=None, ignore_right=True
                )
                local_state = self.layers[layer_idx][neuron_idx]
                if local_field * local_state > threshold:
                    continue
                count += 1
                self.couplings[layer_idx][neuron_idx, :] += (
                    lr * local_state * self.layers[layer_idx][:]
                )  # NOTE: this does not interfere with the local field computation in subsequent updates within the same sweep
                self.couplings[layer_idx][neuron_idx, neuron_idx] = self.J_D
        return count

    def train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_steps: int,
        lr: float,
        threshold: float,
        rng: Optional[np.random.Generator] = None,
    ):
        """Performs a single training step.
        :param x: input.
        :param y: one-hot target.
        :param lr: learning rate.
        :param threshold: stability threshold.
        """
        rng = np.random.default_rng() if rng is None else rng
        self.initialize_state(rng, x)
        num_sweeps = self.relax(max_steps, x, y, rng)
        if num_sweeps == max_steps:
            logging.warning(f"Did not detect convergence in {max_steps} full sweeps.")
        updated_count = self.apply_perceptron_rule(lr, threshold, x)
        return num_sweeps, updated_count

    def inference(
        self,
        inputs: np.ndarray,
        max_steps: int,
        rng: Optional[np.random.Generator] = None,
        repeat: int = 1,
    ):
        """Performs inference with the network.
        :param repeat: do inference independently multiple times.
        :return: the prediction as an array of shape [repeat, num_inputs, num_classes].
        """
        rng = np.random.default_rng() if rng is None else rng
        predictions = np.zeros((repeat, inputs.shape[0], self.C), dtype=DTYPE)
        for i in range(repeat):
            for j, x in enumerate(inputs):
                self.initialize_state(rng, x)
                self.relax(max_steps, x, None, rng)
                predictions[i, j, :] = self.layers[-1].copy()
        return predictions

    def evaluate(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        max_steps: int,
        rng: Optional[np.random.Generator] = None,
        repeat: int = 1,
    ):
        """
        Evaluates the network on a dataset. Compute overall accuracy and per-class accuracy,
        both for individual predictions and through majority voting. \\
        :return: a dictionary with the following keys:
        - overall_accuracy: float,
        - majority_accuracy: float,
        - accuracy_by_class: dict,            # keys: class index, value: accuracy
        - majority_accuracy_by_class: dict,   # keys: class index, value: accuracy
        """
        predictions = self.inference(inputs, max_steps, rng, repeat)
        _, num_inputs, num_classes = predictions.shape
        predicted_individual = np.argmax(predictions, axis=2)  # repeat, num_inputs
        ground_truth = np.argmax(targets, axis=1)  # num_inputs,
        majority_votes = np.empty(num_inputs, dtype=np.int64)  # num_inputs,
        for j in range(num_inputs):
            counts = np.bincount(predicted_individual[:, j], minlength=num_classes)
            majority_votes[j] = np.argmax(counts)

        acc = (predicted_individual == ground_truth[None, :]).mean()
        majority_acc = (majority_votes == ground_truth).mean()
        acc_by_class = {}
        majority_acc_by_class = {}
        for cls in range(num_classes):
            cls_indices = np.where(ground_truth == cls)[0]
            if len(cls_indices) == 0:
                acc_by_class[cls] = None
                majority_acc_by_class[cls] = None
                continue
            acc_by_class[cls] = (predicted_individual[:, cls_indices] == cls).mean()
            majority_acc_by_class[cls] = (majority_votes[cls_indices] == cls).mean()
        return {
            "overall_accuracy": acc,
            "majority_accuracy": majority_acc,
            "accuracy_by_class": acc_by_class,
            "majority_accuracy_by_class": majority_acc_by_class,
        }

    def train_epoch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        max_steps: int,
        lr: float,
        threshold: float,
        rng: Optional[np.random.Generator] = None,
    ):
        """Trains the network for one epoch."""
        rng = np.random.default_rng() if rng is None else rng
        perm = rng.permutation(inputs.shape[0])
        sweep_nums, updates_counts = [], []
        for i in perm:
            num_sweeps, updated_count = self.train_step(
                inputs[i], targets[i], max_steps, lr, threshold, rng
            )
            sweep_nums.append(num_sweeps)
            updates_counts.append(updated_count)
        return sweep_nums, updates_counts

    def train_loop(
        self,
        num_epochs: int,
        inputs: np.ndarray,
        targets: np.ndarray,
        max_steps: int,
        lr: float,
        threshold: float,
        rng: Optional[np.random.Generator] = None,
    ):
        """Trains the network for multiple epochs.
        :param num_epochs: number of epochs.
        """
        for epoch in range(num_epochs):
            sweep_nums, update_counts = self.train_epoch(
                inputs, targets, max_steps, lr, threshold, rng
            )
            metrics = self.evaluate(inputs, targets, max_steps, rng)
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}:\n"
                f"train accuracy: {metrics['overall_accuracy']:.3f}\n"
                f"Average number of full sweeps: {np.mean(sweep_nums):.3f}\n"
                f"Average fraction of perceptron rule updates per full sweep: {np.mean(update_counts) / (self.num_layers * self.N + self.C):.3f}\n"
            )

    def plot_fields_histograms(self, x=None, y=None):
        """
        Plots histograms of the various field types (internal, left, right)
        at each layer. \\
        The plot is arranged in two rows of subplots, one per layer (including
        readout); in each subplot, the three histograms are overlaid.
        """
        total_layers = self.num_layers + 1
        n_cols = math.ceil(total_layers / 2)
        fig1, axs1 = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
        fig2, axs2 = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
        axs1, axs2 = axs1.flatten(), axs2.flatten()

        for layer_idx in range(total_layers):
            internal, left, right, total = [], [], [], []
            for neuron_idx in range(len(self.layers[layer_idx])):
                internal.append(self.internal_field(layer_idx, neuron_idx))
                left.append(self.left_field(layer_idx, neuron_idx, x))
                right.append(self.right_field(layer_idx, neuron_idx, y))
                total.append(internal[-1] + left[-1] + right[-1])
            ax = axs1[layer_idx]
            ax.hist(internal, bins=30, alpha=0.6, label="Internal", color="blue")
            ax.hist(left, bins=30, alpha=0.6, label="Left", color="green")
            ax.hist(right, bins=30, alpha=0.6, label="Right", color="red")
            ax.set_title(
                f"Layer {layer_idx}" if layer_idx < self.num_layers else "Readout"
            )
            ax.legend()
            ax = axs2[layer_idx]
            ax.hist(total, bins=30, alpha=0.6, label="Total", color="black")
            ax.set_title(
                f"Layer {layer_idx}" if layer_idx < self.num_layers else "Readout"
            )
            ax.legend()

        for j in range(total_layers, len(axs1)):
            axs1[j].axis("off")
            axs2[j].axis("off")
        fig1.tight_layout(rect=(0, 0, 1, 0.97))
        fig2.tight_layout(rect=(0, 0, 1, 0.97))
        return fig1, fig2
