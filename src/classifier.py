import logging
from typing import Optional

import numpy as np

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
        self.activations = [np.sign for _ in range(self.num_layers)] + [
            lambda x: int(x > 0)
        ]

    def initialize_state(self, rng: Optional[np.random.Generator] = None):
        """Initializes the state of the network."""
        rng = np.random.default_rng() if rng is None else rng
        self.layers = [initialize_state(self.N, rng) for _ in range(self.num_layers)]
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
            )  # TODO: multiplier?
        return self.lambda_left * self.layers[layer_idx - 1][neuron_idx]

    def right_field(
        self, layer_idx: int, neuron_idx: int, y: Optional[np.ndarray] = None
    ):
        """Field due to interaction with next layer, or with right external field."""
        if layer_idx == self.num_layers - 1:
            return np.dot(
                self.W[:, neuron_idx], self.layers[self.num_layers]
            )  # TODO: multiplier?
        if layer_idx == self.num_layers:
            if y is None:
                return 0
            return self.lambda_y * y[neuron_idx]
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
        x: np.ndarray,
        y: np.ndarray,
        max_steps: int,
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
        for layer_idx in range(self.num_layers):
            for neuron_idx in range(self.N):
                local_field = self.local_field(
                    layer_idx, neuron_idx, x=x, y=None, ignore_right=True
                )
                local_state = self.layers[layer_idx][neuron_idx]
                if local_field * local_state > threshold:
                    continue
                self.couplings[layer_idx][neuron_idx, :] += (
                    lr * local_state * self.layers[layer_idx][:]
                )  # NOTE: this does not interfere with the local field computation in subsequent updates within the same sweep
                self.couplings[layer_idx][neuron_idx, neuron_idx] = self.J_D

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
        self.initialize_state(rng)
        steps = self.relax(x, y, max_steps, rng)
        if steps == max_steps:
            logging.warning(f"Did not detect convergence in {max_steps} full sweeps.")
        self.apply_perceptron_rule(lr, threshold, x)

    def inference(self, x: np.ndarray, repeat: int = 1):
        """Performs inference with the network.
        :param x: input.
        :param repeat: do inference independently multiple times.
        :return: the prediction as an array of shape [repeat, num_classes].
        """
        pass

    def train_epoch(self):
        """Trains the network for one epoch."""
        pass

    def train_loop(self, num_epochs: int):
        """Trains the network for multiple epochs.
        :param num_epochs: number of epochs.
        """
        pass
