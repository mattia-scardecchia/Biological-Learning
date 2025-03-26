import logging
import math
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt


def initialize_layer(
    batch_size: int, layer_width: int, device: str, generator: torch.Generator
):
    state = torch.randint(
        0,
        2,
        (batch_size, layer_width),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return state * 2 - 1


class BatchMeIfYouCan:
    """
    BatchMeIfYouCan

    A dangerously parallel operator for when you’ve got too many dot products
    and not enough patience. Executes large-scale batch computations with
    reckless efficiency using all the cores, threads, and dark magic available.

    Features:
    - Massively parallel batch processing
    - Matrix multiplications at light speed
    - Makes your CPU sweat and your RAM cry
    - Not responsible for any melted laptops

    Usage:
        Just give it data. It’ll handle the rest. Fast. Loud. Proud.

    Warning:
        Not for the faint of FLOP. May cause overheating, data loss, or
        existential dread. Use at your own risk.

    """

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
        device: str = "cpu",
        seed: int = None,
    ):
        """
        Initializes the classifier.
        :param num_layers: number of hidden layers.
        :param N: number of neurons per hidden layer.
        :param C: number of neurons in the readout layer.
        :param lambda_left: coupling strength with the previous layer.
        :param lambda_right: coupling strength with the next layer.
        :param lambda_x: strength of coupling with the input.
        :param lambda_y: strength of coupling with the target.
        :param J_D: self-interaction strength (diagonal of internal couplings).
        :param device: 'cpu' or 'cuda'.
        :param seed: optional random seed.
        """
        self.num_layers = num_layers
        self.N = N
        self.C = C
        self.lambda_left = lambda_left
        self.lambda_right = lambda_right
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.J_D = J_D
        self.device = device
        self.generator = torch.Generator(device=self.device)
        self.cpu_generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)
            self.cpu_generator.manual_seed(seed)

        self.couplings = self.initialize_couplings()  # num_layers, N, N
        self.diagonal_mask = torch.stack(
            [torch.eye(N, device=device, dtype=torch.bool)] * num_layers
        )
        self.W_forth = self.initialize_readout_weights()  # C, N
        self.W_back = self.W_forth.clone()  # C, N
        self.W_forth /= math.sqrt(N)
        self.W_back /= math.sqrt(C)

        logging.info(f"Initialized classifier on device: {self.device}")
        logging.info(f"Parameters: N={N}, C={C}, num_layers={num_layers}")

    def initialize_couplings(self):
        """
        Initializes the internal couplings within each layer."
        """
        Js = []
        for _ in range(self.num_layers):
            J = torch.randn(
                self.N, self.N, device=self.device, generator=self.generator
            )
            J /= math.sqrt(self.N)
            J.fill_diagonal_(self.J_D)
            Js.append(J)
        return torch.stack(Js)

    def initialize_readout_weights(self):
        """
        Initializes the readout weight matrix.
        """
        weights = torch.randint(
            0,
            2,
            (self.C, self.N),
            device=self.device,
            dtype=torch.float32,
            generator=self.generator,
        )
        return weights * 2 - 1

    def initialize_neurons_state(
        self, batch_size: int, x: Optional[torch.Tensor] = None
    ):
        """
        Initializes the state of the neurons within each layer, and
        in the readout layer
        """
        if x is not None:
            states = [x.clone() for _ in range(self.num_layers)]
        else:
            states = [
                initialize_layer(batch_size, self.N, self.device, self.generator)
                for _ in range(self.num_layers)
            ]
        readout = initialize_layer(batch_size, self.C, self.device, self.generator)
        return torch.stack(states), readout

    def sign(self, input: torch.Tensor):
        """
        Sign activation function. Returns +1 if x > 0, -1 if x < 0, 0 if x == 0.
        """
        # This used to return 1 at x == 0. The reason was that, in the computation
        # of the right field at the readout layer, where neurons feel the influence
        # of the one-hot encoded target y, we needed to have -1 at 0 otherwise all
        # components of the right field would be positive. Now, instead, we handle that
        # case without calling sign, so we can use the optimized built-in function.
        # This ofc has the problem of potentially introducing 0s in the state...
        return torch.sign(input)

    def internal_field(self, states: torch.Tensor):
        """
        For each neuron in the network, excluding those in the readout layer,
        computes the internal field.
        :param states: tensor of shape (num_layers, batch_size, N).
        """
        return torch.matmul(states, self.couplings)

    def left_field(self, states: torch.Tensor, x: Optional[torch.Tensor] = None):
        """
        For each neuron in the network, computes the left field.
        """
        if x is None:
            x = torch.zeros(states[0].shape, device=self.device)
        hidden_left = torch.cat(
            [x.clone().unsqueeze(0) * self.lambda_x, states[:-1] * self.lambda_left]
        )
        readout_left = torch.matmul(states[-1], self.W_forth.T)
        return hidden_left, readout_left

    def right_field(
        self,
        states: torch.Tensor,
        readout: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        """
        For each neuron in the network, computes the right field.
        """
        if y is None:
            y = torch.zeros(readout.shape, device=self.device)
        hidden_right = torch.cat(
            [
                states[1:] * self.lambda_right,
                torch.matmul(readout, self.W_back).unsqueeze(0),
            ]
        )
        readout_right = (2 * y.clone() - 1) * self.lambda_y
        return hidden_right, readout_right

    def local_field(
        self,
        states: torch.Tensor,
        readout: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        ignore_right: bool = False,
    ):
        """
        For each neuron in the network, including the readout layer,
        computes the local field.
        """
        hidden_internal = self.internal_field(states)
        hidden_right, readout_right = self.right_field(states, readout, y)
        hidden_left, readout_left = self.left_field(states, x)
        if ignore_right:
            hidden = hidden_internal + hidden_left
            readout = readout_left
        else:
            hidden = hidden_internal + hidden_left + hidden_right
            readout = readout_left + readout_right
        return hidden, readout

    def relax(
        self,
        states: torch.Tensor,
        readout: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        max_steps: int = 100,
        ignore_right: bool = False,
    ):
        steps = 0
        unsats = []
        while steps < max_steps:
            steps += 1
            hidden_field, readout_field = self.local_field(
                states, readout, x, y, ignore_right=ignore_right
            )
            unsat = torch.mean(hidden_field * states < 0, dtype=torch.float).item()
            unsats.append(unsat)
            new_states = self.sign(hidden_field)
            new_readout = self.sign(readout_field)
            # logging.debug(
            #     f"Relaxation step {steps}: flip rate = {1 - torch.mean(new_states == states, dtype=torch.float).item()}"
            # )
            if torch.equal(readout, new_readout) and torch.equal(states, new_states):
                break
            states, readout = new_states, new_readout
        # if steps == max_steps:
        #     logging.warning("Relaxation did not converge")
        return states, readout, steps

    def perceptron_rule_update(
        self,
        states: torch.Tensor,
        readout: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        lr: float,
        threshold: float,
    ):
        hidden_field, _ = self.local_field(states, readout, x, y, True)
        is_unstable = ((hidden_field * states) <= threshold).float()
        delta_J = lr * torch.matmul((is_unstable * states).transpose(1, 2), states)
        delta_J[self.diagonal_mask] = 0
        self.couplings += delta_J
        num_updates = is_unstable.sum().item()
        logging.debug(f"Number of perceptron rule updates: {num_updates}")
        return num_updates

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        max_steps: int,
        lr: float,
        threshold: float,
    ):
        states, readout = self.initialize_neurons_state(x.shape[0], x)
        states, readout, num_sweeps = self.relax(states, readout, x, y, max_steps)
        num_updates = self.perceptron_rule_update(states, readout, x, y, lr, threshold)
        return num_sweeps, num_updates

    def inference(self, x: torch.Tensor, max_steps: int):
        initial_states, initial_readout = self.initialize_neurons_state(x.shape[0], x)
        states, readout, _ = self.relax(
            initial_states, initial_readout, x, max_steps=max_steps, ignore_right=True
        )
        logits = torch.matmul(states[-1], self.W_forth.T)
        return logits, states, readout

    def evaluate(self, x: torch.Tensor, y: torch.Tensor, max_steps: int):
        logits, states, readout = self.inference(x, max_steps)
        predictions = torch.argmax(logits, dim=1)
        ground_truth = torch.argmax(y, dim=1)
        accuracy = (predictions == ground_truth).float().mean().item()
        accuracy_by_class = {}
        for cls in range(self.C):
            cls_mask = ground_truth == cls
            accuracy_by_class[cls] = (
                (predictions[cls_mask] == cls).float().mean().item()
            )
        fixed_points = {idx: states[idx] for idx in range(self.num_layers)}
        fixed_points[self.num_layers] = readout
        return {
            "overall_accuracy": accuracy,
            "accuracy_by_class": accuracy_by_class,
            "fixed_points": fixed_points,
            "logits": logits,
        }

    def train_epoch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        max_steps: int,
        lr: float,
        threshold: float,
        batch_size: int,
    ):
        """
        Trains the network for one epoch over the training set.
        Shuffles the dataset and processes mini-batches.
        :param inputs: tensor of shape (num_samples, N).
        :param targets: tensor of shape (num_samples, C).
        :param max_steps: maximum relaxation sweeps.
        :param lr: learning rate.
        :param threshold: stability threshold.
        :param batch_size: mini-batch size.
        :return: tuple (list of sweeps per batch, list of update counts per batch)
        """
        num_samples = inputs.shape[0]
        idxs_perm = torch.randperm(num_samples, generator=self.cpu_generator)
        sweeps_list, updates_list = [], []
        for i in range(0, num_samples, batch_size):
            batch_idxs = idxs_perm[i : i + batch_size]
            x = inputs[batch_idxs]
            y = targets[batch_idxs]
            sweeps, updates = self.train_step(x, y, max_steps, lr, threshold)
            sweeps_list.append(sweeps)
            updates_list.append(updates)
        return sweeps_list, updates_list

    @torch.inference_mode()
    def train_loop(
        self,
        num_epochs: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        max_steps: int,
        lr: float,
        threshold: float,
        batch_size: int,
        eval_interval: Optional[int] = None,
        eval_inputs: Optional[torch.Tensor] = None,
        eval_targets: Optional[torch.Tensor] = None,
    ):
        """
        Trains the network for multiple epochs.
        Logs training accuracy and (optionally) validation accuracy.
        :param num_epochs: number of epochs.
        :param inputs: training inputs, shape (num_samples, N).
        :param targets: training targets, shape (num_samples, C).
        :param max_steps: maximum relaxation sweeps.
        :param lr: learning rate.
        :param threshold: stability threshold.
        :param batch_size: mini-batch size.
        :param eval_interval: evaluation interval in epochs.
        :param eval_inputs: validation inputs.
        :param eval_targets: validation targets.
        :return: tuple (train accuracy history, eval accuracy history)
        """
        if eval_interval is None:
            eval_interval = num_epochs + 1  # never evaluate
        train_acc_history = []
        eval_acc_history = []
        representations = defaultdict(list)  # input, time, layer
        for epoch in range(num_epochs):
            sweeps, updates = self.train_epoch(
                inputs, targets, max_steps, lr, threshold, batch_size
            )
            train_metrics = self.evaluate(inputs, targets, max_steps)
            avg_sweeps = torch.tensor(sweeps).float().mean().item()
            avg_updates = torch.tensor(updates).float().mean().item()
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}:\n"
                f"Train Acc: {train_metrics['overall_accuracy']:.3f}\n"
                f"Avg number of full sweeps: {avg_sweeps:.3f}\n"
                f"Avg fraction of updates per sweep: {avg_updates / ((self.num_layers * self.N + self.C) * batch_size):.3f}"
            )
            train_acc_history.append(train_metrics["overall_accuracy"])
            if (
                (epoch + 1) % eval_interval == 0
                and eval_inputs is not None
                and eval_targets is not None
            ):
                eval_metrics = self.evaluate(eval_inputs, eval_targets, max_steps)
                logging.info(f"Val Acc: {eval_metrics['overall_accuracy']:.3f}\n")
                eval_acc_history.append(eval_metrics["overall_accuracy"])
                for i in range(len(eval_inputs)):
                    representations[i].append(
                        [
                            eval_metrics["fixed_points"][idx][i]
                            for idx in range(self.num_layers)
                        ]
                    )
        representations = {i: np.array(reps) for i, reps in representations.items()}
        return train_acc_history, eval_acc_history, representations

    def plot_couplings_histograms(self):
        """
        Plots histograms of the internal coupling values for each hidden layer.
        :return: matplotlib figure.
        """
        ncols = math.ceil((self.num_layers + 2) / 2)
        fig, axs = plt.subplots(2, ncols, figsize=(5 * ncols, 8))
        axs = axs.flatten()
        for layer_idx in range(self.num_layers):
            couplings_np = self.couplings[layer_idx].cpu().detach().numpy().flatten()
            ax = axs[layer_idx]
            ax.hist(couplings_np, bins=30, alpha=0.6, label="Couplings", color="purple")
            ax.set_title(f"Layer {layer_idx}")
            ax.grid()
            ax.legend()
        for ax, title, W in zip(
            axs[self.num_layers : self.num_layers + 2],
            ["Readout Weights (Forth)", "Readout Weights (Back)"],
            [self.W_forth, self.W_back],
        ):
            ax.hist(
                W.cpu().detach().numpy().flatten(),
                bins=30,
                alpha=0.6,
                label="Readout Weights",
                color="blue",
            )
            ax.set_title(title)
            ax.grid()
            ax.legend()
        for j in range(self.num_layers + 2, len(axs)):
            axs[j].axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig

    def plot_fields_histograms(
        self,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        relax_first: bool = True,
        max_steps: int = 100,
    ):
        """
        Plots histograms of the various field components (internal, left, right, total)
        at each layer.
        :param x: input tensor of shape (batch, N).
        :param y: target tensor of shape (batch, C).
        :param relax_first: whether to relax the network before plotting.
        :param max_steps: maximum relaxation sweeps for obtaining a fixed point.
        :return: tuple of figures (fields figure, total fields figure)
        """
        batch_size = x.shape[0] if x is not None else 1
        states, readout = self.initialize_neurons_state(batch_size, x)
        if relax_first:
            states, readout, _ = self.relax(states, readout, x, y, max_steps)
        total_layers = self.num_layers + 1
        ncols = math.ceil(total_layers / 2)
        fig_fields, axs_fields = plt.subplots(2, ncols, figsize=(5 * ncols, 8))
        fig_total, axs_total = plt.subplots(2, ncols, figsize=(5 * ncols, 8))
        axs_fields = axs_fields.flatten()
        axs_total = axs_total.flatten()
        hidden_internal = self.internal_field(states)
        hidden_right, readout_right = self.right_field(states, readout, y)
        hidden_left, readout_left = self.left_field(states, x)
        hidden_total = hidden_internal + hidden_left + hidden_right
        readout_total = readout_left + readout_right

        for layer_idx in range(total_layers):
            if layer_idx == self.num_layers:
                internal = np.zeros_like(readout_left).flatten()
                left = readout_left.cpu().detach().numpy().flatten()
                right = readout_right.cpu().detach().numpy().flatten()
                total = readout_total.cpu().detach().numpy().flatten()
            else:
                internal = hidden_internal[layer_idx].cpu().detach().numpy().flatten()
                left = hidden_left[layer_idx].cpu().detach().numpy().flatten()
                right = hidden_right[layer_idx].cpu().detach().numpy().flatten()
                total = hidden_total[layer_idx].cpu().detach().numpy().flatten()
            ax = axs_fields[layer_idx]
            ax.hist(internal, bins=30, alpha=0.6, label="Internal", color="blue")
            ax.hist(left, bins=30, alpha=0.6, label="Left", color="green")
            ax.hist(right, bins=30, alpha=0.6, label="Right", color="red")
            title = f"Layer {layer_idx}" if layer_idx < self.num_layers else "Readout"
            ax.set_title(title)
            ax.grid()
            ax.legend()
            ax_total = axs_total[layer_idx]
            ax_total.hist(total, bins=30, alpha=0.6, label="Total", color="black")
            ax_total.set_title(title)
            ax_total.grid()
            ax_total.legend()

        for j in range(total_layers, len(axs_fields)):
            axs_fields[j].axis("off")
            axs_total[j].axis("off")
        fig_fields.tight_layout(rect=(0, 0, 1, 0.97))
        fig_total.tight_layout(rect=(0, 0, 1, 0.97))
        return fig_fields, fig_total
