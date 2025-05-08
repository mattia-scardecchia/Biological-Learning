import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

from src.utils import DTYPE

# @torch.compile(mode="max-autotune")
# def relax_kernel(state, couplings, mask, steps):
#     for _ in range(steps):
#         state_unfolded = state.unfold(1, 3, 1).transpose(-2, -1).flatten(2)
#         fields = torch.einsum(
#             "lni,bli->bln",
#             couplings * mask,
#             state_unfolded,
#         )
#         state[:, 1:-1] = torch.sign(fields)
#     return state


def sample_readout_weights(N, C, device, generator):
    # W = torch.randint(
    #     0,
    #     2,
    #     (N, C),
    #     device=device,
    #     dtype=DTYPE,
    #     generator=generator,
    # )
    # return 2 * W - 1
    W = torch.randn(
        (N, C),
        device=device,
        dtype=DTYPE,
        generator=generator,
    )
    return W


def sample_couplings(
    N,
    H,
    device,
    generator,
    J_D_1,
    J_D_2,
    ferromagnetic: bool = False,
):
    """
    Main diagonal
    """
    if ferromagnetic:
        J = torch.zeros((H, H), device=device, dtype=DTYPE)
    else:
        J = torch.randn(H, H, device=device, generator=generator, dtype=DTYPE)
        J /= torch.sqrt(torch.tensor(H, device=device, dtype=DTYPE))
    for i in range(N):
        J[i, i] = J_D_1
    for i in range(N, H):
        J[i, i] = J_D_2
    return J


def sample_state(N, batch_size, device, generator):
    S = torch.randint(
        0,
        2,
        (batch_size, N),
        device=device,
        dtype=DTYPE,
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
        H: int,
        C: int,
        J_D: float,
        lambda_left: list[float],
        lambda_right: list[float],
        lambda_internal: float,
        lambda_fc: float,
        lr: torch.Tensor,
        threshold: torch.Tensor,
        weight_decay: torch.Tensor,
        init_mode: str,
        init_noise: float,
        fc_left: bool,
        fc_right: bool,
        fc_input: bool,
        symmetric_W: bool,
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
        assert not (lambda_fc == 0 and (fc_left or fc_right))
        assert N <= H
        self.L = num_layers
        self.N = N
        self.H = H
        self.C = C
        self.lambda_left = torch.tensor(lambda_left, device=device)
        self.lambda_right = torch.tensor(lambda_right, device=device)
        self.J_D = torch.tensor(J_D, device=device)
        self.fc_left = fc_left
        self.fc_right = fc_right
        self.fc_input = fc_input
        self.lambda_internal = torch.tensor(lambda_internal, device=device)
        self.lambda_fc = torch.tensor(lambda_fc, device=device)
        self.symmetric_W = symmetric_W
        self.init_mode = init_mode
        self.init_noise = init_noise
        assert not ("noisy" in self.init_mode and self.init_noise == 0)

        self.root_H = torch.sqrt(torch.tensor(H, device=device))
        self.root_C = torch.sqrt(torch.tensor(C, device=device))

        self.device = device
        self.generator = torch.Generator(device=self.device)
        self.cpu_generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)
            self.cpu_generator.manual_seed(seed)

        self.couplings = self.initialize_couplings(
            fc_left=fc_left, fc_right=fc_right
        )  # L+1, H, 3H
        self.symmetrize_W()
        self.prepare_tensors(lr, weight_decay, threshold)

        logging.info(f"Initialized {self} on device: {self.device}")
        logging.info(
            f"Parameters:\n"
            f"N={N},\n"
            f"H={H},\n"
            f"C={C},\n"
            f"num_layers={num_layers},\n"
            f"J_D={J_D},\n"
            f"lambda_left={lambda_left},\n"
            f"lambda_right={lambda_right},\n"
            f"lr={lr},\n"
            f"threshold={threshold},\n"
            f"weight_decay={weight_decay}\n"
            f"lambda_internal={lambda_internal},\n"
            f"lambda_fc={lambda_fc},\n"
            f"init_mode={init_mode},\n"
            f"fc_left={fc_left},\n"
            f"fc_right={fc_right},\n"
            f"symmetric_W={symmetric_W},\n"
            f"device={device},\n"
            f"seed={seed},\n"
        )

    def prepare_tensors(self, lr, weight_decay, threshold):
        self.is_learnable = self.build_is_learnable_mask(self.fc_left, self.fc_right)
        self.lr = self.build_lr_tensor(lr)
        self.weight_decay = self.build_weight_decay_tensor(weight_decay)
        self.threshold = threshold.to(self.device)
        self.ignore_right_mask = self.build_ignore_right_mask()

    def initialize_couplings(self, fc_left: bool, fc_right: bool):
        couplings_buffer = []
        # fc_left = fc_right = 0  # hack to set ferromagnetic to True everywhere

        # First Layer
        if self.fc_input:
            J_x = (
                sample_couplings(
                    self.N,
                    self.H,
                    self.device,
                    self.generator,
                    self.lambda_left[0] / self.lambda_fc,
                    0,
                    not fc_left,
                )
                * self.lambda_fc
            )
        else:
            J_x = (
                torch.eye(self.H, device=self.device, dtype=DTYPE) * self.lambda_left[0]
            )
            for i in range(self.N, self.H):
                J_x[i, i] = 0
        couplings_buffer.append(J_x)
        couplings_buffer.append(
            sample_couplings(
                self.N,
                self.H,
                self.device,
                self.generator,
                self.J_D,
                self.J_D,
                False,
            )
            * self.lambda_internal
        )
        if self.L > 1:  # il L == 1, right couplings will be set later (W_back)
            couplings_buffer.append(
                sample_couplings(
                    self.N,
                    self.H,
                    self.device,
                    self.generator,
                    self.lambda_right[-1] / self.lambda_fc,
                    self.lambda_right[0] / self.lambda_fc,
                    not fc_right,
                )
                * self.lambda_fc
            )

        # Middle Layers
        for idx in range(1, self.L - 1):
            couplings_buffer.append(
                sample_couplings(
                    self.N,
                    self.H,
                    self.device,
                    self.generator,
                    self.lambda_left[0] / self.lambda_fc,
                    self.lambda_left[idx] / self.lambda_fc,
                    not fc_left,
                )
                * self.lambda_fc
            )
            couplings_buffer.append(
                sample_couplings(
                    self.N,
                    self.H,
                    self.device,
                    self.generator,
                    self.J_D,
                    self.J_D,
                    False,
                )
                * self.lambda_internal
            )
            couplings_buffer.append(
                sample_couplings(
                    self.N,
                    self.H,
                    self.device,
                    self.generator,
                    self.lambda_right[-1] / self.lambda_fc,
                    self.lambda_right[idx] / self.lambda_fc,
                    not fc_right,
                )
                * self.lambda_fc
            )

        # Last Layer
        if self.L > 1:  # il L == 1, left couplings have been set before
            couplings_buffer.append(
                sample_couplings(
                    self.N,
                    self.H,
                    self.device,
                    self.generator,
                    self.lambda_left[0] / self.lambda_fc,
                    self.lambda_left[self.L - 1] / self.lambda_fc,
                    not fc_left,
                )
                * self.lambda_fc
            )
            couplings_buffer.append(
                sample_couplings(
                    self.N,
                    self.H,
                    self.device,
                    self.generator,
                    self.J_D,
                    self.J_D,
                    False,
                )
                * self.lambda_internal
            )
        W_initial = sample_readout_weights(self.H, self.C, self.device, self.generator)
        W_back = W_initial.clone() * self.lambda_right[-2] / self.root_C
        couplings_buffer.append(
            F.pad(
                W_back,
                (0, self.H - self.C, 0, 0),
                mode="constant",
                value=0,
            )  # (H, C) -> (H, H)
        )

        # Readout Layer
        W_forth = W_initial.clone().T * self.lambda_left[-1] / self.root_H
        couplings_buffer.append(
            F.pad(
                W_forth,
                (0, 0, 0, self.H - self.C),
                mode="constant",
                value=0,
            )  # (H, C) -> (H, H)
        )
        couplings_buffer.append(torch.zeros((self.H, self.H), device=self.device))
        id = torch.eye(self.C, device=self.device) * self.lambda_right[-1]
        couplings_buffer.append(
            F.pad(
                id,
                (0, self.H - self.C, 0, self.H - self.C),
                mode="constant",
                value=0,
            )  # (C, C) -> (H, H)
        )

        # Get the correct layout
        # couplings = (
        #     torch.stack(couplings_buffer)
        #     .reshape(self.L + 1, 3, self.N, self.N)
        #     .transpose(1, 2)
        #     .reshape(self.L + 1, self.N, 3 * self.N)
        # )
        couplings = torch.stack(
            [
                torch.cat(couplings_buffer[i * 3 : (i + 1) * 3], dim=1)
                for i in range(self.L + 1)
            ]
        )

        return couplings.to(self.device)

    def build_is_learnable_mask(self, fc_left: bool, fc_right: bool):
        H, N, L, C = self.H, self.N, self.L, self.C

        mask = torch.ones_like(self.couplings)
        mask[-1, :, H:] = 0
        mask[-1, C:H, :H] = 0
        mask[-2, :, 2 * H + C :] = 0
        mask[0, :, :H] = 0

        for idx in range(L):
            mask[idx, :, H : 2 * H].fill_diagonal_(0)
            if idx > 0:
                mask[idx, :N, :N].fill_diagonal_(0)
                if not fc_left:
                    mask[idx, :, :H] = 0
            if idx < L - 1:
                mask[idx, :N, 2 * H : 2 * H + N].fill_diagonal_(0)
                if not fc_right:
                    mask[idx, :, 2 * H : 3 * H] = 0

        return mask.to(self.device).to(torch.bool)

    def build_lr_tensor(self, lr):
        H, L, C = self.H, self.L, self.C
        lr_tensor = torch.zeros_like(self.couplings)  # (L+1, H, 3H)
        for idx in range(L):
            lr_tensor[idx, :, :] = lr[idx] / math.sqrt(H)
            if idx < L:
                lr_tensor[idx, :, H : 2 * H] *= self.lambda_internal
                if idx < L - 1:
                    lr_tensor[idx, :, 2 * H : 3 * H] *= (
                        self.lambda_fc
                    )  # NOTE: this creates problems if we unfreeze the ferromagnetic diagonal
                if idx > 0:
                    lr_tensor[idx, :, :H] *= (
                        self.lambda_fc
                    )  # NOTE: this creates problems if we unfreeze the ferromagnetic diagonal
        lr_tensor[L - 1, :, 2 * H : 2 * H + C] = (
            lr[-2] * self.lambda_right[-2] / math.sqrt(C)
        )  # overwrite W_back
        lr_tensor[L, :C, :H] = lr[-1] * self.lambda_left[-1] / math.sqrt(H)  # W_forth
        lr_tensor[self.is_learnable == 0] = 0
        return lr_tensor.to(self.device)

    def build_weight_decay_tensor(self, weight_decay):
        H, L, C = self.H, self.L, self.C
        weight_decay_tensor = torch.zeros_like(self.couplings, device=self.device)
        for idx in range(L):
            weight_decay_tensor[idx, :, :] = weight_decay[idx]
        weight_decay_tensor[L - 1, :, 2 * H : 2 * H + C] = weight_decay[-2]
        weight_decay_tensor[L, :C, :H] = weight_decay[-1]
        weight_decay_tensor *= self.lr
        return weight_decay_tensor

    def build_ignore_right_mask(self):
        H, L = self.H, self.L
        mask = torch.ones_like(self.couplings).unsqueeze(0).repeat(2, 1, 1, 1)
        mask[1, :, :, 2 * H : 3 * H] = 0
        # mask[1, L - 1, :, 2 * H : 3 * H] = 1  # keep W_back with ignore_right=1
        return mask.to(self.device)

    def initialize_state(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: str,
    ):
        """
        :param x: shape (batch_size, N)
        :param y: shape (batch_size, C)
        :return: shape (batch_size, L+3, H)
        """
        H, N, C, L = self.H, self.N, self.C, self.L
        batch_size = x.shape[0]
        x, y = x.to(self.device, DTYPE), y.to(self.device, DTYPE)
        x_padded = F.pad(x, (0, H - N, 0, 0), "constant", 0).unsqueeze(
            1
        )  # (B, N) -> (B, 1, H)
        if mode == "input":
            neurons = x_padded.repeat(1, L, 1)
            # y_hat = sample_state(C, batch_size, self.device, self.generator)
            # y_hat = torch.zeros((batch_size, C), device=self.device, dtype=DTYPE)
            y_hat = y.clone()
        elif mode == "zeros":
            neurons = torch.zeros((batch_size, L, H), device=self.device, dtype=DTYPE)
            y_hat = torch.zeros((batch_size, C), device=self.device, dtype=DTYPE)
        elif mode == "noisy_zeros":
            signs = (
                torch.randint(0, 2, (batch_size, L, H), device=self.device, dtype=DTYPE)
                * 2
                - 1
            )
            neurons = torch.where(
                torch.rand(H, device=self.device, dtype=DTYPE) < self.init_noise,
                signs,
                torch.zeros_like(signs, device=self.device, dtype=DTYPE),
            )
            y_hat = torch.zeros((batch_size, C), device=self.device, dtype=DTYPE)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        y_hat = F.pad(y_hat, (0, H - C, 0, 0), mode="constant", value=0).unsqueeze(
            1
        )  # (B, C) -> (B, 1, H)
        y_padded = F.pad(
            2 * y - 1,
            (0, H - C, 0, 0),
            mode="constant",
            value=0,
        ).unsqueeze(1)  # (B, C) -> (B, 1, H)
        state = torch.cat(
            [
                x_padded,
                neurons,
                y_hat,
                y_padded,
            ],
            dim=1,
        )  # NOTE: repeat copies the data
        return state

    def local_field(
        self,
        state: torch.Tensor,
        ignore_right,
    ):
        """
        :param state: shape (B, L+3, H)
        :return: shape (B, L+1, H)
        """
        state_unfolded = (
            state.unfold(1, 3, 1).transpose(-2, -1).flatten(2)
        )  # Shape: (B, L+1, 3*H)
        return torch.einsum(
            "lni,bli->bln",
            self.couplings * self.ignore_right_mask[ignore_right],
            state_unfolded,
        )

    def symmetrize_W(self):
        if self.symmetric_W == "buggy":
            self.couplings[-2, :, 2 * self.H : 2 * self.H + self.C] = (
                self.W_forth.T
                * self.root_H
                * self.lambda_right[-2]
                / self.root_C
                / self.lambda_left[-1]
                / 100
            )
        elif self.symmetric_W:
            norm_old = self.W_back.norm(dim=1)
            self.couplings[-2, :, 2 * self.H : 2 * self.H + self.C] = (
                self.W_forth / self.W_forth.norm(dim=0)[None, :]
            ).T * norm_old[:, None]
        else:
            pass

    def perceptron_rule(
        self,
        state: torch.Tensor,
    ):
        fields = self.local_field(state, ignore_right=1)  # shape (B, L+1, H)
        neurons = state[:, 1:-1, :]  # shape (B, L+1, H)
        S_unfolded = state.unfold(1, 3, 1).transpose(-2, -1)  # shape (B, L+1, 3, H)
        is_unstable = (fields * neurons) <= self.threshold[None, :, None]
        delta = (
            self.lr
            * torch.einsum("bli,blcj->licj", neurons * is_unstable, S_unfolded).flatten(
                2
            )
            / math.sqrt(state.shape[0])
        )
        self.couplings = self.couplings * (1 - self.weight_decay) + delta

        # # W_back <- W_forth (with appropriate scaling)
        self.symmetrize_W()
        return is_unstable

    def relax(
        self,
        state: torch.Tensor,
        max_steps: int,
        ignore_right: int,
    ):
        # final_state = relax_kernel(
        #     state, self.couplings, self.ignore_right_mask[ignore_right], max_steps
        # )
        # unsat = self.fraction_unsat(state)
        # return final_state, max_steps, unsat
        sweeps = 0
        while sweeps < max_steps:
            sweeps += 1
            fields = self.local_field(state, ignore_right=ignore_right)
            torch.sign(fields, out=state[:, 1:-1, :])
        unsat = self.fraction_unsat(state, ignore_right=ignore_right)
        return state, sweeps, unsat

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        max_sweeps: int,
    ):
        state = self.initialize_state(x, y, self.init_mode)
        final_state, num_sweeps, unsat = self.relax(state, max_sweeps, ignore_right=0)
        made_update = self.perceptron_rule(final_state)
        return {
            "sweeps": num_sweeps,
            "hidden_updates": made_update[:, :-1, :],
            "readout_updates": made_update[:, -1, : self.C],
            "hidden_unsat": unsat[:, :-1, :],
            "readout_unsat": unsat[:, -1, : self.C],
            "update_states": final_state[:, 1:-2, :],
        }

    def inference(
        self,
        x: torch.Tensor,
        max_sweeps: int,
    ):
        state = self.initialize_state(
            x,
            torch.zeros((x.shape[0], self.C), device=self.device, dtype=DTYPE),
            self.init_mode,
        )
        final_state, num_sweeps, unsat = self.relax(state, max_sweeps, ignore_right=1)
        logits = final_state[:, -3] @ self.couplings[-1, : self.C, : self.H].T
        states, readout = final_state[:, 1:-2], final_state[:, -2]
        return logits, states, readout

    def set_wback(self, new):
        self.couplings[-2, :, 2 * self.H : 2 * self.H + self.C] = new

    def wforth2wback(self, wforth):
        return (
            wforth.T
            * self.root_H
            * self.lambda_right[-2]
            / self.root_C
            / self.lambda_left[-1]
        )

    @property
    def W_back(self):
        return self.couplings[-2, :, 2 * self.H : 2 * self.H + self.C]

    @property
    def W_forth(self):
        return self.couplings[-1, : self.C, : self.H]

    @property
    def internal_couplings(self):
        return self.couplings[:-1, :, self.H : 2 * self.H]

    @property
    def left_couplings(self):
        return self.couplings[1:-1, :, : self.H]

    @property
    def right_couplings(self):
        return self.couplings[0:-2, :, 2 * self.H : 3 * self.H]

    @property
    def input_couplings(self):
        return self.couplings[0, :, : self.N]

    @property
    def output_couplings(self):
        return self.couplings[-1, : self.C, 2 * self.H : 2 * self.H + self.C]

    @staticmethod
    def split_state(state):
        x = state[:, 0, :]
        S = state[:, 1:-2, :]
        y_hat = state[:, -2, :]
        y = state[:, -1, :]
        return x, S, y_hat, y

    def field_breakdown(self, state, x, y):
        internal = torch.einsum(
            "lni,bli->bln", self.couplings[:, :, self.H : 2 * self.H], state[:, 1:-1, :]
        )
        left = torch.einsum(
            "lni,bli->bln", self.couplings[:, :, : self.H], state[:, 0:-2, :]
        )
        right = torch.einsum(
            "lni,bli->bln",
            self.couplings[:, :, 2 * self.H : 3 * self.H],
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
        is_unsat = (fields * state[:, 1:-1, :]) < 0
        return is_unsat
