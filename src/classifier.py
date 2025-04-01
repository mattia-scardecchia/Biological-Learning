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
        J_D: float,
        lambda_left: list[float],
        lambda_right: list[float],
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Initializes the classifier.
        :param num_layers: number of hidden layers.
        :param N: number of neurons per hidden layer.
        :param C: number of neurons in the readout layer.
        :param lambda_left: coupling strength with the previous layer. First element is lambda_x.
        :param lambda_right: coupling strength with the next layer. Last element is lambda_y.
        :param J_D: self-interaction strength (diagonal of internal couplings).
        :param device: 'cpu' or 'cuda'.
        :param seed: optional random seed.
        """
        self.num_layers = num_layers
        self.N = N
        self.C = C
        assert len(lambda_left) == len(lambda_right) == num_layers + 1
        self.lambda_left = torch.tensor(lambda_left, device=device)
        self.lambda_right = torch.tensor(lambda_right, device=device)
        self.J_D = torch.tensor(J_D, device=device)
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
        logging.info(
            f"Parameters: N={N}, C={C}, num_layers={num_layers}, J_D={J_D}, lambda_left={lambda_left}, lambda_right={lambda_right}"
        )

        # self.fixed_noise = torch.stack(
        #     [
        #         initialize_layer(1, self.N, self.device, self.generator)
        #         for _ in range(self.num_layers)
        #     ]
        # )  # num_layers, 1, N

        # self.fixed_noise = (
        #     initialize_layer(1, self.N, self.device, self.generator)
        #     .unsqueeze(0)
        #     .expand(self.num_layers, -1, -1)
        # )  # num_layers, 1, N

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

        # weights = torch.randn(
        #     self.C, self.N, device=self.device, generator=self.generator
        # )
        # return weights

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

        # assert x is not None
        # # states = torch.stack([x.sign().clone() for _ in range(self.num_layers)])
        # # mask = torch.rand_like(states) < 0.05
        # # states[mask] = self.fixed_noise.expand(-1, batch_size, -1)[mask]
        # states = self.fixed_noise.expand(-1, batch_size, -1).clone()
        # readout = initialize_layer(batch_size, self.C, self.device, self.generator)
        # return states, readout

        # assert x is not None
        # probas = torch.linspace(0, 0.5, self.num_layers)
        # residual = torch.stack([self.sign(x).clone() for _ in range(self.num_layers)])
        # mask = torch.rand_like(residual) < probas[:, None, None]
        # states = torch.where(
        #     mask, self.fixed_noise.expand(-1, batch_size, -1), residual
        # )
        # readout = initialize_layer(batch_size, self.C, self.device, self.generator)
        # return states, readout

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

    def internal_field_layer(self, states: torch.Tensor, layer_idx: int):
        return torch.matmul(states[layer_idx], self.couplings[layer_idx].T)

    def internal_field(self, states: torch.Tensor):
        """
        For each neuron in the network, excluding those in the readout layer,
        computes the internal field.
        :param states: tensor of shape (num_layers, batch_size, N).
        """
        return torch.matmul(states, self.couplings.transpose(1, 2))

    def left_field_layer(
        self, states: torch.Tensor, layer_idx: int, x: Optional[torch.Tensor] = None
    ):
        match layer_idx:
            case 0:
                if x is None:
                    return 0
                return x * self.lambda_left[0]
            case self.num_layers:
                return torch.matmul(states[-1], self.W_forth.T) * self.lambda_left[-1]
            case _:
                return states[layer_idx - 1] * self.lambda_left[layer_idx]

    def left_field(self, states: torch.Tensor, x: Optional[torch.Tensor] = None):
        """
        For each neuron in the network, computes the left field.
        """
        if x is None:
            x = torch.zeros(states[0].shape, device=self.device)
        hidden_left = torch.cat(
            [
                x.unsqueeze(0) * self.lambda_left[0],
                states[:-1] * self.lambda_left[1:-1, None, None],
            ]
        )
        readout_left = torch.matmul(states[-1], self.W_forth.T) * self.lambda_left[-1]
        return hidden_left, readout_left

    def right_field_layer(
        self,
        states: torch.Tensor,
        layer_idx: int,
        readout: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        if layer_idx == self.num_layers:
            if y is None:
                return 0
            return (2 * y - 1) * self.lambda_right[-1]
        elif layer_idx == self.num_layers - 1:
            return torch.matmul(readout, self.W_back) * self.lambda_right[-2]
        else:
            return states[layer_idx + 1] * self.lambda_right[layer_idx]

    def right_field(
        self,
        states: torch.Tensor,
        readout: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        """
        For each neuron in the network, computes the right field.
        """
        hidden_right = torch.cat(
            [
                states[1:] * self.lambda_right[:-2, None, None],
                torch.matmul(readout, self.W_back).unsqueeze(0) * self.lambda_right[-2],
            ]
        )
        readout_right = (2 * y - 1) * self.lambda_right[-1] if y is not None else 0
        return hidden_right, readout_right

    def local_field_layer(
        self,
        states: torch.Tensor,
        readout: torch.Tensor,
        layer_idx: int,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        ignore_right: bool = False,
    ):
        internal = (
            self.internal_field_layer(states, layer_idx)
            if layer_idx < self.num_layers
            else torch.tensor(0, device=self.device)
        )
        left = self.left_field_layer(states, layer_idx, x)
        right = self.right_field_layer(states, layer_idx, readout, y)
        if ignore_right:
            return internal + left
        return internal + left + right

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
        while steps < max_steps:
            steps += 1
            hidden_field, readout_field = self.local_field(
                states, readout, x, y, ignore_right=ignore_right
            )
            new_states = self.sign(hidden_field)
            new_readout = self.sign(readout_field)
            # if torch.equal(readout, new_readout) and torch.equal(states, new_states):
            #     break
            states, readout = new_states, new_readout
        # print(self.num_unsat_neurons(states, readout, x, y) / states.numel())
        return states, readout, steps

    # def relax(
    #     self,
    #     states: torch.Tensor,
    #     readout: torch.Tensor,
    #     x: Optional[torch.Tensor] = None,
    #     y: Optional[torch.Tensor] = None,
    #     max_steps: int = 100,
    #     ignore_right: bool = False,
    # ):
    #     steps = 0
    #     while steps < max_steps:
    #         steps += 1
    #         for layer_idx in range(self.num_layers)[::-1]:
    #             hidden_field = self.local_field_layer(
    #                 states, readout, layer_idx, x, y, ignore_right=ignore_right
    #             )
    #             new_state = self.sign(hidden_field)
    #             states[layer_idx] = new_state
    #         readout_field = self.local_field_layer(
    #             states, readout, self.num_layers, x, y, ignore_right=ignore_right
    #         )
    #         new_readout = self.sign(readout_field)
    #         readout = new_readout
    #     print(self.num_unsat_neurons(states, readout, x, y) / states.numel())
    #     return states, readout, steps

    def perceptron_rule_update(
        self,
        states: torch.Tensor,
        readout: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        lr: torch.Tensor,
        threshold: torch.Tensor,
        weight_decay: torch.Tensor,
    ):
        # update J
        hidden_field, readout_field = self.local_field(states, readout, x, y, True)
        is_unstable = ((hidden_field * states) <= threshold[:-1, None, None]).float()
        delta_J = (
            lr[:-2, None, None]
            * torch.matmul((is_unstable * states).transpose(1, 2), states)
            / math.sqrt(self.N)
            / math.sqrt(x.shape[0])
        )
        self.couplings = (
            self.couplings
            * (
                1
                - weight_decay[:-2, None, None]
                * lr[:-2, None, None]
                / math.sqrt(self.N)
            )
            + delta_J
        )
        self.couplings[self.diagonal_mask] = self.J_D
        fraction_updates = is_unstable.mean().item()

        # update W_back
        delta_W_back = (
            (lr[-2] * torch.matmul((is_unstable[-1] * states[-1]).T, readout).T)
            / math.sqrt(self.C)
            / math.sqrt(x.shape[0])
        )
        self.W_back = (
            self.W_back * (1 - weight_decay[-2] * lr[-2] / math.sqrt(self.C))
            + delta_W_back
        )

        # update W_forth
        is_unstable_readout = (readout_field * readout <= threshold[-1]).float()
        delta_W_forth = (
            lr[-1]
            * torch.matmul((is_unstable_readout * readout).T, states[-1])
            / math.sqrt(self.N)
            / math.sqrt(x.shape[0])
        )
        self.W_forth = (
            self.W_forth * (1 - weight_decay[-1] * lr[-1] / math.sqrt(self.N))
            + delta_W_forth
        )

        return fraction_updates

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        max_steps: int,
        lr: torch.Tensor,
        threshold: torch.Tensor,
        weight_decay: torch.Tensor,
    ):
        states, readout = self.initialize_neurons_state(x.shape[0], x)
        states, readout, num_sweeps = self.relax(states, readout, x, y, max_steps)
        fraction_updates = self.perceptron_rule_update(
            states, readout, x, y, lr, threshold, weight_decay
        )
        return num_sweeps, fraction_updates

    def inference(self, x: torch.Tensor, max_steps: int):
        initial_states, initial_readout = self.initialize_neurons_state(x.shape[0], x)
        states, readout, _ = self.relax(
            initial_states,
            initial_readout,
            x,
            y=None,
            max_steps=max_steps,
            ignore_right=True,
        )
        # logits = torch.matmul(states[-1], self.W_forth.T)
        logits = self.left_field_layer(states, self.num_layers, x)
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
        lr: torch.Tensor,
        threshold: torch.Tensor,
        weight_decay: torch.Tensor,
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
            sweeps, fraction_updates = self.train_step(
                x, y, max_steps, lr, threshold, weight_decay
            )
            sweeps_list.append(sweeps)
            updates_list.append(fraction_updates)
        return sweeps_list, updates_list

    @torch.inference_mode()
    def train_loop(
        self,
        num_epochs: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        max_steps: int,
        lr: torch.Tensor,
        threshold: torch.Tensor,
        weight_decay: torch.Tensor,
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
        assert len(threshold) == self.num_layers + 1
        assert len(weight_decay) == self.num_layers + 2
        assert len(lr) == self.num_layers + 2
        if eval_interval is None:
            eval_interval = num_epochs + 1  # never evaluate
        train_acc_history = []
        eval_acc_history = []
        representations = defaultdict(list)  # input, time, layer
        for epoch in range(num_epochs):
            sweeps, fraction_updates = self.train_epoch(
                inputs, targets, max_steps, lr, threshold, weight_decay, batch_size
            )
            train_metrics = self.evaluate(inputs, targets, max_steps)
            avg_sweeps = torch.tensor(sweeps).float().mean().item()
            avg_updates = torch.tensor(fraction_updates).float().mean().item()
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}:\n"
                f"Train Acc: {train_metrics['overall_accuracy']:.3f}\n"
                f"Avg number of full sweeps: {avg_sweeps:.3f}\n"
                f"Avg fraction of updates per sweep: {avg_updates:.3f}"
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
        representations = {
            i: np.array([[t.cpu() for t in sublist] for sublist in reps])
            for i, reps in representations.items()
        }
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
            couplings_np = self.couplings[layer_idx].cpu().numpy().flatten()
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
                W.cpu().numpy().flatten(),
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
        relax_first: bool = False,
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
                internal = np.zeros_like(readout_left.cpu().numpy()).flatten()
                left = readout_left.cpu().numpy().flatten()
                right = readout_right.cpu().numpy().flatten()
                total = readout_total.cpu().numpy().flatten()
            else:
                internal = hidden_internal[layer_idx].cpu().numpy().flatten()
                left = hidden_left[layer_idx].cpu().numpy().flatten()
                right = hidden_right[layer_idx].cpu().numpy().flatten()
                total = hidden_total[layer_idx].cpu().numpy().flatten()
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

    def num_unsat_neurons(
        self,
        states,
        readout,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ):
        hidden_field, readout_field = self.local_field(states, readout, x, y)
        is_unstable = ((hidden_field * states) <= 0).float()
        return is_unstable.sum().item()
