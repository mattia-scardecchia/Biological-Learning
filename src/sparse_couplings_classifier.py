from typing import Optional

import numpy as np

from src.classifier import Classifier, initialize_readout_weights
from src.utils import DTYPE, sign, theta


def initialize_J(N, J_D, sparsity_level, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    J = np.array(
        rng.normal(0, 1 / np.sqrt(N * sparsity_level), size=(N, N)),
        dtype=DTYPE,
    )
    np.fill_diagonal(J, J_D)
    return J


class SparseCouplingsClassifier(Classifier):
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
        sparse_readout: bool = True,
        sparsity_level: float = 1.0,
    ):
        """Initializes the classifier. \\
        :param num_layers: number of layers. \\
        :param N: number of neurons per layer. \\
        :param C: number of classes. \\
        :param lambda_left: strength of coupling with previous layer. \\
        :param lambda_right: strength of coupling with next layer. \\
        :param lambda_x: strength of coupling with input. \\
        :param lambda_y: strength of coupling with target. \\
        :param J_D: self-interaction strength. \\
        :param rng: random number generator for initialization. \\
        :param sparse_readout: if true, use 0/1 states for the readout layer. Otherwise, use -1/1 states. \\
        :param sparsity_level: fraction of active couplings.
        """
        self.num_layers = num_layers
        self.N = N
        self.C = C
        self.lambda_left = lambda_left
        self.lambda_right = lambda_right
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.J_D = J_D
        self.sparse_readout = sparse_readout
        self.sparsity_level = sparsity_level

        rng = np.random.default_rng() if rng is None else rng
        self.active_couplings = [
            rng.binomial(
                1,
                self.sparsity_level,
                (self.N, self.N),
            )
            for _ in range(self.num_layers)
        ]
        for i in range(self.num_layers):
            for j in range(self.N):
                self.active_couplings[i][j, j] = 1  # Keep J_D active

        self.initialize_state(rng)
        self.initialize_couplings(rng)
        if sparse_readout:
            self.activations = [sign for _ in range(self.num_layers)] + [theta]
        else:
            self.activations = [sign for _ in range(self.num_layers + 1)]

    def enforce_sparsity(self):
        """Sets inactive couplings to zero."""
        for i in range(self.num_layers):
            self.couplings[i] *= self.active_couplings[i]

    def initialize_couplings(self, rng: Optional[np.random.Generator] = None):
        """Initializes the couplings of the network."""
        rng = np.random.default_rng() if rng is None else rng
        self.couplings = [
            initialize_J(self.N, self.J_D, self.sparsity_level, rng)
            for _ in range(self.num_layers)
        ]  # num_layers, N, N
        self.enforce_sparsity()
        self.W = initialize_readout_weights(self.N, self.C, rng)  # C, N

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
                    lr
                    * local_state
                    * self.layers[layer_idx][:]
                    * self.active_couplings[layer_idx][neuron_idx, :]
                )  # NOTE: this does not interfere with the local field computation in subsequent updates within the same sweep
                self.couplings[layer_idx][neuron_idx, neuron_idx] = self.J_D
        return count
