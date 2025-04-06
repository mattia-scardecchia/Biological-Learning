import math
from typing import Optional

import torch
import torch.nn.functional as F


def sample_readout_weights(N, C, device, generator):
    W = torch.randint(
        0,
        2,
        (N, C),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return 2 * W - 1


def sample_couplings(N, device, generator, J_D, ferromagnetic: bool = False):
    if ferromagnetic:
        return torch.diag(torch.ones(N, device=device) * J_D)
    J = torch.randn(N, N, device=device, generator=generator)
    J /= torch.sqrt(torch.tensor(N, device=device))
    J.fill_diagonal_(J_D.item())
    return J


def sample_state(N, batch_size, device, generator):
    S = torch.randint(
        0,
        2,
        (batch_size, N),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return 2 * S - 1


class BatchMeIfUCan:
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
        lr: torch.Tensor,
        threshold: torch.Tensor,
        weight_decay: torch.Tensor,
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
        assert len(lambda_left) == len(lambda_right) == num_layers + 1
        self.L = num_layers
        self.N = N
        self.C = C
        self.lambda_left = torch.tensor(lambda_left, device=device)
        self.lambda_right = torch.tensor(lambda_right, device=device)
        self.J_D = torch.tensor(J_D, device=device)

        self.root_N = torch.sqrt(torch.tensor(N, device=device))
        self.root_C = torch.sqrt(torch.tensor(C, device=device))

        self.device = device
        self.generator = torch.Generator(device=self.device)
        self.cpu_generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)
            self.cpu_generator.manual_seed(seed)

        self.couplings = self.initialize_couplings(ferromagnetic=True)  # L+1, N, 3N
        self.is_learnable = self.build_is_learnable_mask(learn_cross_couplings=False)
        self.lr = self.build_lr_tensor(lr)
        self.weight_decay = self.build_weight_decay_tensor(weight_decay)
        self.threshold = threshold.to(self.device)
        self.ignore_right_mask = self.build_ignore_right_mask()

    def initialize_couplings(self, ferromagnetic: bool = False):
        couplings_buffer = []

        # First Layer
        couplings_buffer.append(
            torch.eye(self.N, device=self.device) * self.lambda_left[0]
        )
        couplings_buffer.append(
            sample_couplings(self.N, self.device, self.generator, self.J_D)
        )
        couplings_buffer.append(
            sample_couplings(
                self.N,
                self.device,
                self.generator,
                self.lambda_right[0],
                ferromagnetic,
            )
        )

        # Middle Layers
        for idx in range(1, self.L - 1):
            couplings_buffer.append(
                sample_couplings(
                    self.N,
                    self.device,
                    self.generator,
                    self.lambda_left[idx],
                    ferromagnetic,
                )
            )
            couplings_buffer.append(
                sample_couplings(self.N, self.device, self.generator, self.J_D)
            )
            couplings_buffer.append(
                sample_couplings(
                    self.N,
                    self.device,
                    self.generator,
                    self.lambda_right[idx],
                    ferromagnetic,
                )
            )

        # Last Layer
        couplings_buffer.append(
            sample_couplings(
                self.N,
                self.device,
                self.generator,
                self.lambda_left[self.L - 1],
                ferromagnetic,
            )
        )
        couplings_buffer.append(
            sample_couplings(self.N, self.device, self.generator, self.J_D)
        )
        W_initial = sample_readout_weights(self.N, self.C, self.device, self.generator)
        W_back = W_initial.clone() / self.root_C
        couplings_buffer.append(
            F.pad(
                W_back,
                (0, self.N - self.C, 0, 0),
                mode="constant",
                value=0,
            )  # (N, C) -> (N, N)
        )

        # Readout Layer
        W_forth = W_initial.clone().T / self.root_N
        couplings_buffer.append(
            F.pad(
                W_forth,
                (0, 0, 0, self.N - self.C),
                mode="constant",
                value=0,
            )  # (C, N) -> (N, N)
        )

        couplings_buffer.append(torch.zeros((self.N, self.N), device=self.device))
        id = torch.eye(self.C, device=self.device) * self.lambda_right[-1]
        couplings_buffer.append(
            F.pad(
                id,
                (0, self.N - self.C, 0, self.N - self.C),
                mode="constant",
                value=0,
            )  # (C, C) -> (N, N)
        )

        # Get the correct layout
        couplings = (
            torch.stack(couplings_buffer)
            .reshape(self.L + 1, 3, self.N, self.N)
            .transpose(1, 2)
            .reshape(self.L + 1, self.N, 3 * self.N)
        )
        # more_transparent_couplings = torch.stack(
        #     [
        #         torch.cat(couplings_buffer[i * 3 : (i + 1) * 3], dim=1)
        #         for i in range(self.L + 1)
        #     ]
        # )

        return couplings.to(self.device)

    def build_is_learnable_mask(self, learn_cross_couplings: bool = True):
        N, L, C = self.N, self.L, self.C

        mask = torch.ones_like(self.couplings)
        mask[-1, :, N:] = 0
        mask[-1, C:N, :N] = 0
        mask[-2, :, 2 * N + C :] = 0
        mask[0, :, :N] = 0

        for idx in range(L):
            mask[idx, :, N : 2 * N].fill_diagonal_(0)
            if idx > 0:
                mask[idx, :, :N].fill_diagonal_(0)
            if idx < L - 1:
                mask[idx, :, 2 * N :].fill_diagonal_(0)
            if not learn_cross_couplings:
                if idx > 0:
                    mask[idx, :, :N] = 0
                if idx < L - 1:
                    mask[idx, :, 2 * N : 3 * N] = 0

        return mask.to(self.device).to(torch.bool)

    def build_lr_tensor(self, lr):
        N, L, C = self.N, self.L, self.C
        lr_tensor = torch.zeros_like(self.couplings)  # (L+1, N, 3N)
        for idx in range(L):
            lr_tensor[idx, :, :] = lr[idx] / math.sqrt(N)
        lr_tensor[L - 1, :, 2 * N : 2 * N + C] = lr[-2] / math.sqrt(C)  # overwrite
        lr_tensor[L, :C, :N] = lr[-1] / math.sqrt(N)
        lr_tensor[self.is_learnable == 0] = 0
        return lr_tensor.to(self.device)

    def build_weight_decay_tensor(self, weight_decay):
        N, L, C = self.N, self.L, self.C
        weight_decay_tensor = torch.zeros_like(self.couplings, device=self.device)
        for idx in range(L):
            weight_decay_tensor[idx, :, :] = weight_decay[idx]
        weight_decay_tensor[L - 1, :, 2 * N : 2 * N + C] = weight_decay[-2]
        weight_decay_tensor[L, :C, :N] = weight_decay[-1]
        weight_decay_tensor *= self.lr
        return weight_decay_tensor

    def build_ignore_right_mask(self):
        N = self.N
        mask = torch.ones_like(self.couplings).unsqueeze(0).repeat(2, 1, 1, 1)
        mask[1, :, :, 2 * N : 3 * N] = 0
        return mask.to(self.device)

    def initialize_state(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        :param x: shape (batch_size, N)
        :param y: shape (batch_size, C)
        :return: shape (batch_size, L+3, N)
        """
        batch_size = x.shape[0]
        y_hat = sample_state(self.C, batch_size, self.device, self.generator)
        y_hat = F.pad(
            y_hat,
            (0, self.N - self.C, 0, 0),
            mode="constant",
            value=0,
        )  # (B, C) -> (B, N)
        y_padded = F.pad(
            2 * y - 1,
            (0, self.N - self.C, 0, 0),
            mode="constant",
            value=0,
        )  # (B, C) -> (B, N)
        state = torch.cat(
            [
                x.unsqueeze(1).repeat(1, self.L + 1, 1),
                y_hat.unsqueeze(1),
                y_padded.unsqueeze(1),
            ],
            dim=1,
        )  # NOTE: repeat copies the data
        return state

    def local_field(
        self,
        state: torch.Tensor,
        ignore_right: int = 1,
    ):
        """
        :param state: shape (B, L+3, N)
        :return: shape (B, L+1, N)
        """
        state_unfolded = (
            state.unfold(1, 3, 1).transpose(-2, -1).flatten(2)
        )  # Shape: (B, L+1, 3*N)
        return torch.einsum(
            "lni,bli->bln",
            self.couplings * self.ignore_right_mask[ignore_right],
            state_unfolded,
        )

    def perceptron_rule(
        self,
        state: torch.Tensor,
    ):
        fields = self.local_field(state, ignore_right=1)  # shape (B, L+1, N)
        neurons = state[:, 1:-1, :]  # shape (B, L+1, N)
        S_unfolded = state.unfold(1, 3, 1).transpose(-2, -1)  # shape (B, L+1, 3, N)
        is_unstable = (fields * neurons) <= self.threshold[None, :, None]
        delta = (
            self.lr
            * torch.einsum("bli,blcj->licj", neurons * is_unstable, S_unfolded).flatten(
                2
            )
            / math.sqrt(state.shape[0])
        )
        self.couplings = self.couplings * (1 - self.weight_decay) + delta
        return is_unstable

    def relax(
        self,
        state: torch.Tensor,
        max_steps: int,
        ignore_right: int,
    ):
        sweeps = 0
        while sweeps < max_steps:
            sweeps += 1
            fields = self.local_field(state, ignore_right=ignore_right)
            state[:, 1:-1, :] = torch.sign(fields)
        unsat = self.fraction_unsat(state)
        return state, sweeps, unsat

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        max_sweeps: int,
    ):
        state = self.initialize_state(x, y)
        final_state, num_sweeps, unsat = self.relax(state, max_sweeps, ignore_right=0)
        made_update = self.perceptron_rule(final_state)
        return {
            "sweeps": num_sweeps,
            "hidden_updates": made_update[:, :-1, :],
            "readout_updates": made_update[:, -1, : self.C],
            "hidden_unsat": unsat[:, :-1, :],
            "readout_unsat": unsat[:, -1, : self.C],
        }

    def inference(
        self,
        x: torch.Tensor,
        max_sweeps: int,
    ):
        state = self.initialize_state(
            x, torch.zeros((x.shape[0], self.C), device=self.device)
        )
        final_state, num_sweeps, unsat = self.relax(state, max_sweeps, ignore_right=1)
        logits = final_state[:, -3] @ self.couplings[-1, : self.C, : self.N].T
        states, readout = final_state[:, 1:-2], final_state[:, -2]
        return logits, states, readout

    @property
    def W_back(self):
        return self.couplings[-2, :, 2 * self.N : 2 * self.N + self.C]

    @property
    def W_forth(self):
        return self.couplings[-1, : self.C, : self.N]

    @property
    def internal_couplings(self):
        return self.couplings[:-1, :, self.N : 2 * self.N]

    @property
    def left_couplings(self):
        return self.couplings[1:-1, :, : self.N]

    @property
    def right_couplings(self):
        return self.couplings[0:-2, :, 2 * self.N : 3 * self.N]

    @property
    def input_couplings(self):
        return self.couplings[0, :, : self.N]

    @property
    def output_couplings(self):
        return self.couplings[-1, : self.C, 2 * self.N : 2 * self.N + self.C]

    @staticmethod
    def split_state(state):
        x = state[:, 0, :]
        S = state[:, 1:-2, :]
        y_hat = state[:, -2, :]
        y = state[:, -1, :]
        return x, S, y_hat, y

    def field_breakdown(self, state, x, y):
        internal = torch.einsum(
            "lni,bli->bln", self.couplings[:, :, self.N : 2 * self.N], state[:, 1:-1, :]
        )
        left = torch.einsum(
            "lni,bli->bln", self.couplings[:, :, : self.N], state[:, 0:-2, :]
        )
        right = torch.einsum(
            "lni,bli->bln",
            self.couplings[:, :, 2 * self.N : 3 * self.N],
            state[:, 2:, :],
        )
        total = internal + left + right
        return {
            "internal": internal,
            "left": left,
            "right": right,
            "total": total,
        }

    def fraction_unsat(self, state, ignore_right: int = 0):
        fields = self.local_field(state, ignore_right=ignore_right)
        is_unsat = (fields * state[:, 1:-1, :]) <= 0
        return is_unsat
