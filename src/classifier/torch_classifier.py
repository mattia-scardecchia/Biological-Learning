import logging
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch


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


class TorchClassifier:
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

        self.couplings = [
            self.initialize_J() for _ in range(num_layers)
        ]  # num_layers, N, N
        self.W_forth = self.initialize_readout_weights()  # C, N
        self.W_back = self.W_forth.clone()  # C, N
        logging.info(
            f"TorchClassifier initialized with {num_layers} hidden layers, N={N}, C={C} on {device}"
        )

    def initialize_state(
        self, batch_size: int, x: torch.Tensor = None, y: torch.Tensor = None
    ):
        """
        Initializes states for all layers (hidden and readout) for a given batch.
        If x is provided, all hidden layers are set to clones of x;
        otherwise, they are randomly initialized to -1 or +1.
        For the readout layer, if y is provided, it is used (clamped) else randomly initialized.
        :param batch_size: number of samples.
        :param x: input tensor of shape (batch_size, N) or None.
        :param y: target tensor of shape (batch_size, C) or None.
        :return: list of state tensors for each layer (first num_layers for hidden layers, last for readout).
        """
        if x is not None:
            states = [x.clone() for _ in range(self.num_layers)]
        else:
            states = [
                initialize_layer(batch_size, self.N, self.device, self.generator)
                for _ in range(self.num_layers)
            ]
        states.append(initialize_layer(batch_size, self.C, self.device, self.generator))
        return states

    def initialize_J(self):
        """
        Initializes an internal coupling matrix of shape (N, N) with normal values
        (std = 1/sqrt(N)) and sets its diagonal to J_D.
        """
        J = torch.randn(
            (self.N, self.N), device=self.device, generator=self.generator
        ) / math.sqrt(self.N)
        J.fill_diagonal_(self.J_D)
        return J

    def initialize_readout_weights(self):
        """
        Initializes the readout weight matrix (W_forth) of shape (C, N) with values -1 or 1.
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

    def internal_field(self, layer_idx: int, states: list):
        """
        Computes the internal field for a given layer.
        For hidden layers, it is a matrix multiplication between the state and the transposed coupling matrix.
        For the readout layer, it is zero.
        """
        if layer_idx == self.num_layers:
            return torch.zeros_like(states[layer_idx])  # readout layer
        return torch.matmul(states[layer_idx], self.couplings[layer_idx].t())

    def left_field(self, layer_idx: int, states: list, x: torch.Tensor = None):
        """Field due to interaction with previous layer, or with left external field."""
        if layer_idx == 0:
            if x is None:
                return torch.zeros_like(states[layer_idx])
            return self.lambda_x * x
        if layer_idx == self.num_layers:
            return torch.matmul(
                states[self.num_layers - 1], self.W_forth.t()
            ) / math.sqrt(self.N)
        return self.lambda_left * states[layer_idx - 1]

    def right_field(self, layer_idx: int, states: list, y: torch.Tensor = None):
        """
        Computes the right field.
        - For the last hidden layer, it returns the projection from the readout state via W_back.
        - For the readout layer, if y is provided, it returns lambda_y * activation(y).
        - For intermediate layers, it returns lambda_right * state of the next layer.
        """
        if layer_idx == self.num_layers - 1:
            prod = torch.matmul(states[self.num_layers], self.W_back)
            return prod / math.sqrt(self.C)
        if layer_idx == self.num_layers:
            if y is None:
                return torch.zeros_like(states[layer_idx])
            return self.lambda_y * (2 * y - 1)  # NOTE: assume y is one-hot encoded
        return self.lambda_right * states[layer_idx + 1]

    def local_field(
        self,
        layer_idx: int,
        states: list,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        ignore_right: bool = False,
    ):
        """
        Computes the local field perceived by neurons in a given layer.
        It sums the internal, left, and right fields.
        """
        internal = self.internal_field(layer_idx, states)
        left = self.left_field(layer_idx, states, x)
        right = (
            torch.zeros_like(states[layer_idx])
            if ignore_right
            else self.right_field(layer_idx, states, y)
        )
        return internal + left + right

    def relax(
        self,
        states: list,
        x: torch.Tensor,
        y: torch.Tensor = None,
        max_steps: int = 100,
        ignore_right: bool = False,
    ):
        """
        Synchronously relaxes the network to a stable state.
        Uses initialize_state to set up the hidden layers (with x if provided)
        and the readout layer (with y if provided).
        :param states: list of initial state tensors for each layer.
        :param x: input tensor of shape (batch, N).
        :param y: target tensor of shape (batch, C). If provided, the readout layer is clamped.
        :param max_steps: maximum number of synchronous sweeps.
        :return: tuple (states, steps) where states is a list of state tensors for each layer.
        """
        steps = 0
        # unsats = []
        while steps < max_steps:
            steps += 1
            # unsat, tot = 0, 0
            new_states = []
            for layer_idx in range(self.num_layers + 1):
                local_field = self.local_field(
                    layer_idx, states, x, y, ignore_right=ignore_right
                )
                # unsat += ((local_field * states[layer_idx]) < 0).sum().item()
                # tot += local_field.numel()
                new_state = self.sign(local_field)
                new_states.append(new_state)
            # unsats.append(unsat / tot)
            if all(torch.equal(old, new) for old, new in zip(states, new_states)):
                break
            states = new_states
        logging.debug(f"Relaxation converged in {steps} steps")
        return states, steps

    def perceptron_rule_update(
        self, states: list, x: torch.Tensor, lr: float, threshold: float
    ):
        """
        Applies the perceptron learning rule in a batched, vectorized manner.
        Updates internal coupling matrices and readout weights.
        :param states: list of state tensors from a relaxed network.
        :param x: input tensor of shape (batch, N).
        :param lr: learning rate.
        :param threshold: stability threshold.
        :return: total number of updates performed.
        """
        total_updates = 0

        for layer_idx in range(self.num_layers):
            local_field = self.local_field(
                layer_idx, states, x, y=None, ignore_right=True
            )
            s = states[layer_idx]
            is_unstable = (local_field * s) <= threshold
            total_updates += is_unstable.sum().item()
            delta_J = lr * torch.matmul((is_unstable.float() * s).t(), s)
            self.couplings[layer_idx] = self.couplings[layer_idx] + delta_J
            self.couplings[layer_idx].fill_diagonal_(self.J_D)

        logging.debug(f"Perceptron rule updates: {total_updates}")
        return total_updates

        # # Update readout weights --> needs checking
        # s_last = states[self.num_layers - 1]
        # s_readout = states[self.num_layers]
        # lf_last = self.local_field(
        #     self.num_layers - 1, states, x, y=None, ignore_right=True
        # )
        # cond_last = (lf_last * s_last) <= threshold
        # total_updates += cond_last.sum().item()
        # delta_W_back = lr * torch.matmul(s_readout.t(), cond_last.float() * s_last)
        # self.W_back = self.W_back + delta_W_back

        # lf_readout = self.local_field(
        #     self.num_layers, states, x, y=None, ignore_right=True
        # )
        # cond_readout = (lf_readout * s_readout) <= threshold
        # total_updates += cond_readout.sum().item()
        # delta_W_forth = lr * torch.matmul(
        #     (cond_readout.float() * s_readout).t(), s_last
        # )
        # self.W_forth = self.W_forth + delta_W_forth

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        max_steps: int,
        lr: float,
        threshold: float,
    ):
        """
        Performs one training step: relaxes the network on input x (with target y) and applies the perceptron rule.
        :param x: input tensor of shape (batch, N).
        :param y: target tensor of shape (batch, C).
        :param max_steps: maximum relaxation sweeps.
        :param lr: learning rate.
        :param threshold: stability threshold.
        :return: tuple (sweeps, num_updates)
        """
        initial_states = self.initialize_state(x.shape[0], x, y)
        final_states, num_sweeps = self.relax(initial_states, x, y, max_steps)
        num_updates = self.perceptron_rule_update(final_states, x, lr, threshold)
        return num_sweeps, num_updates

    def inference(self, x: torch.Tensor, max_steps: int):
        """
        Performs inference on a batch of inputs.
        :param x: input tensor of shape (batch, N).
        :param max_steps: maximum relaxation sweeps.
        :return: tuple (logits, states) where logits is the output of the readout layer.
        """
        initial_states = self.initialize_state(x.shape[0], x)
        final_states, _ = self.relax(
            initial_states, x, y=None, max_steps=max_steps, ignore_right=True
        )
        logits = self.left_field(self.num_layers, final_states)
        return logits, final_states

    def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor, max_steps: int):
        """
        Evaluates the network on a batch of inputs.
        :param inputs: tensor of shape (num_samples, N).
        :param targets: tensor of shape (num_samples, C) (one-hot encoded).
        :param max_steps: maximum relaxation sweeps.
        :return: tuple (accuracy, logits)
        """
        logits, fixed_points = self.inference(inputs, max_steps)
        predictions = torch.argmax(logits, dim=1)
        ground_truth = torch.argmax(targets, dim=1)
        accuracy = (predictions == ground_truth).float().mean().item()
        accuracy_by_class = {}
        for cls in range(self.C):
            cls_mask = ground_truth == cls
            accuracy_by_class[cls] = (
                (predictions[cls_mask] == cls).float().mean().item()
            )
        return {
            "overall_accuracy": accuracy,
            "accuracy_by_class": accuracy_by_class,
            "predictions": predictions,
            "fixed_points": {
                idx: fixed_points[idx].cpu().numpy() for idx in range(len(fixed_points))
            },
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
        indices = torch.randperm(num_samples, generator=self.cpu_generator)
        sweeps_list = []
        updates_list = []
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_inputs = inputs[batch_idx]
            batch_targets = targets[batch_idx]
            sweeps, updates = self.train_step(
                batch_inputs, batch_targets, max_steps, lr, threshold
            )
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
        eval_interval: int = None,
        eval_inputs: torch.Tensor = None,
        eval_targets: torch.Tensor = None,
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

    def plot_fields_histograms(
        self, x: torch.Tensor, y: torch.Tensor = None, max_steps: int = 100
    ):
        """
        Plots histograms of the various field components (internal, left, right, total)
        at each layer.
        :param x: input tensor of shape (batch, N).
        :param y: target tensor of shape (batch, C) (optional).
        :param max_steps: maximum relaxation sweeps for obtaining a fixed point.
        :return: tuple of figures (fields figure, total fields figure)
        """
        initial_states = self.initialize_state(x.shape[0], x, y)
        final_states, _ = self.relax(initial_states, x, y, max_steps)
        total_layers = self.num_layers + 1
        n_cols = math.ceil(total_layers / 2)
        fig_fields, axs_fields = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
        fig_total, axs_total = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
        axs_fields = axs_fields.flatten()
        axs_total = axs_total.flatten()

        for l in range(total_layers):
            internal = self.internal_field(l, final_states)
            left = self.left_field(l, final_states, x)
            right = self.right_field(l, final_states, y)
            total_field = internal + left + right

            internal_np = internal.cpu().detach().numpy().flatten()
            left_np = left.cpu().detach().numpy().flatten()
            right_np = right.cpu().detach().numpy().flatten()
            total_np = total_field.cpu().detach().numpy().flatten()

            ax = axs_fields[l]
            ax.hist(internal_np, bins=30, alpha=0.6, label="Internal", color="blue")
            ax.hist(left_np, bins=30, alpha=0.6, label="Left", color="green")
            ax.hist(right_np, bins=30, alpha=0.6, label="Right", color="red")
            title = f"Layer {l}" if l < self.num_layers else "Readout"
            ax.set_title(title)
            ax.legend()

            ax_total = axs_total[l]
            ax_total.hist(total_np, bins=30, alpha=0.6, label="Total", color="black")
            ax_total.set_title(title)
            ax_total.legend()

        # Turn off extra axes if any.
        for j in range(total_layers, len(axs_fields)):
            axs_fields[j].axis("off")
            axs_total[j].axis("off")

        fig_fields.tight_layout(rect=(0, 0, 1, 0.97))
        fig_total.tight_layout(rect=(0, 0, 1, 0.97))
        return fig_fields, fig_total

    def plot_couplings_histograms(self):
        """
        Plots histograms of the internal coupling values for each hidden layer.
        :return: matplotlib figure.
        """
        num_plots = self.num_layers
        fig, axs = plt.subplots(2, num_plots, figsize=(5 * num_plots, 8))
        axs = axs.flatten()

        for l in range(self.num_layers):
            couplings_np = self.couplings[l].cpu().detach().numpy().flatten()
            ax = axs[l]
            ax.hist(couplings_np, bins=30, alpha=0.6, label="Couplings", color="purple")
            ax.set_title(f"Layer {l}")
            ax.legend()

        # Turn off extra axes if any.
        for j in range(self.num_layers, len(axs)):
            axs[j].axis("off")

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig
