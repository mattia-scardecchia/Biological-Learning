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
        self.layers = [initialize_state(N, rng) for _ in range(num_layers)]
        self.layers.append(
            initialize_readout_state(C, rng)
        )  # num_layers + 1, N (layers[-1] is the readout layer)
        self.couplings = [
            initialize_couplings(N, J_D, rng) for _ in range(num_layers)
        ]  # num_layers, N, N
        self.W = initialize_readout_couplings(N, C, rng)  # C, N

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
        """Computes the local field perceived by a neuron.
        :param layer_idx: layer index. Can pass y for the readout layer.
        :param neuron_idx: neuron index.
        :param x: input (left external field). If None, omit it.
        :param y: one-hot target (right external field). If None, omit it.
        :param ignore_right: if True, ignore the interaction from the right (next layer, or external field).
        :return: the local field perceived by the neuron.
        """
        internal_field = self.internal_field(layer_idx, neuron_idx)
        left_field = self.left_field(layer_idx, neuron_idx, x)
        right_field = 0 if ignore_right else self.right_field(layer_idx, neuron_idx, y)
        return internal_field + left_field + right_field

    def relax(self, x: np.ndarray, y: np.ndarray):
        """Relaxes the network to a stable state.
        :param x: input (left external field). If None, omit it.
        :param y: one-hot target (right external field). If None, omit it.
        """
        pass

    def apply_perceptron_rule(self, lr: float, threshold: float):
        """Applies the perceptron learning rule to the network in its current state.
        :param lr: learning rate.
        :param threshold: stability threshold.
        """
        pass

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float, threshold: float):
        """Performs a single training step.
        :param x: input.
        :param y: one-hot target.
        :param lr: learning rate.
        :param threshold: stability threshold.
        """
        pass

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
