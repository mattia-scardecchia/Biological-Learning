import logging
import math

import matplotlib.pyplot as plt
import torch


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

        if seed is not None:
            torch.manual_seed(seed)

        # Create a dedicated generator for reproducibility.
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

        # Initialize internal coupling matrices (one per hidden layer)
        self.couplings = [self.initialize_J() for _ in range(num_layers)]
        # Initialize readout weights (W_forth and symmetric W_back)
        self.W_forth = self.initialize_readout_weights()
        self.W_back = self.W_forth.clone()

        logging.info(
            f"TorchClassifier initialized with {num_layers} hidden layers, N={N}, C={C} on {device}"
        )

    def initialize_state(self, batch_size: int, size: int):
        """
        Initializes a state tensor with values -1 or +1.
        :param batch_size: number of samples in the batch.
        :param size: number of neurons.
        :return: tensor of shape (batch_size, size) on self.device.
        """
        state = torch.randint(
            0,
            2,
            (batch_size, size),
            device=self.device,
            dtype=torch.float32,
            generator=self.generator,
        )
        return state * 2 - 1

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

    def activation(self, x: torch.Tensor):
        """
        Sign activation function. Returns +1 if x>=0, else -1.
        """
        pos = torch.tensor(1.0, device=self.device, dtype=x.dtype)
        neg = torch.tensor(-1.0, device=self.device, dtype=x.dtype)
        return torch.where(x >= 0, pos, neg)

    def local_field(
        self,
        layer_idx: int,
        states: list,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        ignore_right: bool = False,
    ):
        """
        Computes the local field for a given layer (batched).
        :param layer_idx: index of the layer (0..num_layers for readout).
        :param states: list of state tensors for each layer.
        :param x: input tensor of shape (batch, N).
        :param y: target tensor of shape (batch, C).
        :param ignore_right: if True, omit the contribution from the next layer/external field.
        :return: tensor of local fields, shape matching the state at layer_idx.
        """
        # Internal field (only for hidden layers)
        if layer_idx < self.num_layers:
            internal = torch.matmul(states[layer_idx], self.couplings[layer_idx].t())
        else:
            internal = torch.zeros_like(states[layer_idx])

        # Left field
        if layer_idx == 0:
            left = (
                self.lambda_x * x
                if x is not None
                else torch.zeros_like(states[layer_idx])
            )
        elif layer_idx == self.num_layers:
            left = torch.matmul(
                states[self.num_layers - 1], self.W_forth.t()
            ) / math.sqrt(self.N)
        else:
            left = self.lambda_left * states[layer_idx - 1]

        # Right field
        if ignore_right:
            right = torch.zeros_like(states[layer_idx])
        else:
            if layer_idx == self.num_layers - 1:
                right = torch.matmul(
                    states[self.num_layers], self.W_back.t()
                ) / math.sqrt(self.C)
            elif layer_idx == self.num_layers:
                right = (
                    self.lambda_y * self.activation(y)
                    if y is not None
                    else torch.zeros_like(states[layer_idx])
                )
            else:
                right = self.lambda_right * states[layer_idx + 1]

        return internal + left + right

    def relax(self, x: torch.Tensor, y: torch.Tensor = None, max_steps: int = 100):
        """
        Synchronously relaxes the network to a stable state.
        If x is provided, all hidden layers are initialized to x.
        :param x: input tensor of shape (batch, N).
        :param y: target tensor of shape (batch, C). If provided, the readout layer is clamped.
        :param max_steps: maximum number of synchronous sweeps.
        :return: tuple (states, steps) where states is a list of state tensors for each layer.
        """
        batch_size = x.shape[0]
        # Initialize hidden layers: if x is provided, use it for all layers; otherwise, random init.
        if x is not None:
            states = [x.clone() for _ in range(self.num_layers)]
        else:
            states = [
                self.initialize_state(batch_size, self.N)
                for _ in range(self.num_layers)
            ]

        # Initialize readout layer. If target provided, use it; otherwise, random.
        if y is not None:
            states.append(y.clone().to(self.device).float())
        else:
            states.append(self.initialize_state(batch_size, self.C))

        steps = 0
        while steps < max_steps:
            steps += 1
            new_states = []
            for l in range(self.num_layers + 1):
                lf = self.local_field(l, states, x, y, ignore_right=False)
                new_state = self.activation(lf)
                if l == self.num_layers and y is not None:
                    new_state = y.clone().to(self.device).float()
                new_states.append(new_state)
            # Check convergence: all layers must be exactly equal.
            if all(torch.equal(old, new) for old, new in zip(states, new_states)):
                break
            states = new_states

        logging.info(f"Relaxation converged in {steps} steps")
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
        batch_size = x.shape[0]
        total_updates = 0

        # Update internal couplings for hidden layers.
        for l in range(self.num_layers):
            lf = self.local_field(l, states, x, y=None, ignore_right=True)
            s = states[l]
            cond = (lf * s) <= threshold
            total_updates += cond.sum().item()
            delta_J = lr * torch.matmul((cond.float() * s).t(), s)
            self.couplings[l] = self.couplings[l] + delta_J
            self.couplings[l].fill_diagonal_(self.J_D)

        # Update readout weights.
        s_last = states[self.num_layers - 1]
        s_readout = states[self.num_layers]
        lf_last = self.local_field(
            self.num_layers - 1, states, x, y=None, ignore_right=True
        )
        cond_last = (lf_last * s_last) <= threshold
        total_updates += cond_last.sum().item()
        delta_W_back = lr * torch.matmul(s_readout.t(), cond_last.float() * s_last)
        self.W_back = self.W_back + delta_W_back

        lf_readout = self.local_field(
            self.num_layers, states, x, y=None, ignore_right=True
        )
        cond_readout = (lf_readout * s_readout) <= threshold
        total_updates += cond_readout.sum().item()
        delta_W_forth = lr * torch.matmul(
            (cond_readout.float() * s_readout).t(), s_last
        )
        self.W_forth = self.W_forth + delta_W_forth

        logging.info(f"Perceptron rule updates: total updates = {total_updates}")
        return total_updates

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
        states, sweeps = self.relax(x, y, max_steps)
        num_updates = self.perceptron_rule_update(states, x, lr, threshold)
        return sweeps, num_updates

    def inference(self, x: torch.Tensor, max_steps: int):
        """
        Performs inference on a batch of inputs.
        :param x: input tensor of shape (batch, N).
        :param max_steps: maximum relaxation sweeps.
        :return: tuple (logits, states) where logits is the output of the readout layer.
        """
        states, steps = self.relax(x, y=None, max_steps=max_steps)
        logits = self.local_field(
            self.num_layers, states, x, y=None, ignore_right=False
        )
        return logits, states

    def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor, max_steps: int):
        """
        Evaluates the network on a batch of inputs.
        :param inputs: tensor of shape (num_samples, N).
        :param targets: tensor of shape (num_samples, C) (one-hot encoded).
        :param max_steps: maximum relaxation sweeps.
        :return: tuple (accuracy, logits)
        """
        logits, _ = self.inference(inputs, max_steps)
        predictions = torch.argmax(logits, dim=1)
        ground_truth = torch.argmax(targets, dim=1)
        accuracy = (predictions == ground_truth).float().mean().item()
        logging.info(f"Evaluation accuracy: {accuracy:.3f}")
        return accuracy, logits

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
        indices = torch.randperm(num_samples, generator=self.generator)
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
        for epoch in range(num_epochs):
            sweeps, updates = self.train_epoch(
                inputs, targets, max_steps, lr, threshold, batch_size
            )
            accuracy, _ = self.evaluate(inputs, targets, max_steps)
            avg_sweeps = torch.tensor(sweeps).float().mean().item()
            avg_updates = torch.tensor(updates).float().mean().item()
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}: train accuracy: {accuracy:.3f}, "
                f"avg sweeps: {avg_sweeps:.3f}, avg updates: {avg_updates:.3f}"
            )
            train_acc_history.append(accuracy)
            if (
                (epoch + 1) % eval_interval == 0
                and eval_inputs is not None
                and eval_targets is not None
            ):
                val_accuracy, _ = self.evaluate(eval_inputs, eval_targets, max_steps)
                logging.info(f"Validation accuracy: {val_accuracy:.3f}")
                eval_acc_history.append(val_accuracy)
        return train_acc_history, eval_acc_history

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
        # Obtain relaxed states from the network.
        states, _ = self.relax(x, y, max_steps)
        total_layers = self.num_layers + 1
        n_cols = math.ceil(total_layers / 2)
        fig_fields, axs_fields = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
        fig_total, axs_total = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
        axs_fields = axs_fields.flatten()
        axs_total = axs_total.flatten()

        # For each layer, compute the components.
        for l in range(total_layers):
            # Internal field.
            if l < self.num_layers:
                internal = torch.matmul(states[l], self.couplings[l].t())
            else:
                internal = torch.zeros_like(states[l])
            # Left field.
            if l == 0:
                left = self.lambda_x * x
            elif l == self.num_layers:
                left = torch.matmul(
                    states[self.num_layers - 1], self.W_forth.t()
                ) / math.sqrt(self.N)
            else:
                left = self.lambda_left * states[l - 1]
            # Right field.
            if l == self.num_layers - 1:
                right = torch.matmul(
                    states[self.num_layers], self.W_back.t()
                ) / math.sqrt(self.C)
            elif l == self.num_layers:
                right = (
                    self.lambda_y * self.activation(y)
                    if y is not None
                    else torch.zeros_like(states[l])
                )
            else:
                right = self.lambda_right * states[l + 1]

            total_field = internal + left + right

            # Move to CPU and convert to numpy.
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
