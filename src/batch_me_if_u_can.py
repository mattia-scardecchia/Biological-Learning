import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

from src.utils import DTYPE

IGNORE_RIGHT = 1  # for inference mode

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
    H1,
    H2,
    device,
    generator,
    J_D_1,
    J_D_2,
    ferromagnetic: bool = False,
    zero_out_cylinder_contribution: bool = False,
    sparsity: float = 1.0,
    symmetric: bool = False,
):
    """
    :param zero_out_cylinder_contribution: if True, the coupligs in the left rectangle
    of size (H, N) (drawing from neurons in the cylinder) are set to 0.
    """
    if ferromagnetic:
        J = torch.zeros((H2, H2), device=device, dtype=DTYPE)
    else:
        J = torch.randn(H2, H2, device=device, generator=generator, dtype=DTYPE)
        if sparsity < 1.0:
            mask = torch.rand(H2, H2, device=device, generator=generator) < sparsity
            J[~mask] = 0.0
        J /= torch.sqrt(torch.tensor(H2 * sparsity, device=device, dtype=DTYPE))
    if zero_out_cylinder_contribution:
        J[:H2, :H1] = (
            0  # NOTE: we do this here cause we keep cylinder ferromagnetic couplings!
        )
    for i in range(H1):
        J[i, i] = J_D_1
    for i in range(H1, H2):
        J[i, i] = J_D_2
    if symmetric:
        J = (J + J.T) / 2
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
        lambda_internal: float | list[float],
        lambda_fc: float | list[float],
        lr: torch.Tensor,
        threshold: torch.Tensor,
        weight_decay: torch.Tensor,
        init_mode: str,
        init_noise: float,
        fc_left: bool,
        fc_right: bool,
        fc_input: bool,
        symmetric_W: bool,
        symmetric_J_init: bool,
        double_dynamics: bool,
        double_update: bool,
        use_local_ce: bool,
        beta_ce: float,
        p_update: float,
        lambda_cylinder: float,  # ignored
        lambda_wback_skip: float | list[float],
        lambda_wforth_skip: float | list[float],
        lr_wforth_skip: float | list[float],
        weight_decay_wforth_skip: float | list[float],
        lambda_input_skip: float | list[float],
        lambda_input_output_skip: float,
        lr_input_skip: float | list[float],
        weight_decay_input_skip: float | list[float],
        lr_input_output_skip: float,
        weight_decay_input_output_skip: float,
        symmetrize_fc: bool,
        symmetric_threshold_internal_couplings: bool,
        symmetrize_internal_couplings: bool,
        zero_fc_init: bool,
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
        assert not ("noisy" in init_mode and init_noise == 0)
        assert not (double_update and not double_dynamics)
        assert N <= H
        if isinstance(lambda_internal, float):
            lambda_internal = [lambda_internal] * num_layers
        if isinstance(lambda_fc, float):
            lambda_fc = (
                [lambda_fc] * num_layers
            )  # 0-th element is for when fc_input is True (otherwise ignored)
        if isinstance(lambda_wforth_skip, float):
            lambda_wforth_skip = [lambda_wforth_skip] * (num_layers - 1)
        if isinstance(lambda_wback_skip, float):
            lambda_wback_skip = [lambda_wback_skip] * (num_layers - 1)
        if isinstance(lr_wforth_skip, float):
            lr_wforth_skip = [lr_wforth_skip] * (num_layers - 1)
        if isinstance(weight_decay_wforth_skip, float):
            weight_decay_wforth_skip = [weight_decay_wforth_skip] * (num_layers - 1)
        if isinstance(lr_input_skip, float):
            lr_input_skip = [lr_input_skip] * num_layers
        if isinstance(weight_decay_input_skip, float):
            weight_decay_input_skip = [weight_decay_input_skip] * num_layers
        if isinstance(lambda_input_skip, float):
            lambda_input_skip = [lambda_input_skip] * num_layers
        if isinstance(J_D, float):
            J_D = [J_D] * num_layers
        assert len(lambda_input_skip) == num_layers

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
        self.symmetric_J_init = symmetric_J_init
        self.init_mode = init_mode
        self.init_noise = init_noise
        self.double_dynamics = double_dynamics
        self.double_update = double_update
        self.use_local_ce = use_local_ce
        self.beta_ce = beta_ce
        self.p_update = p_update
        self.lambda_wback_skip = torch.tensor(lambda_wback_skip, device=device)
        self.lambda_wforth_skip = torch.tensor(lambda_wforth_skip, device=device)
        self.lr_wforth_skip = torch.tensor(lr_wforth_skip, device=device)
        self.weight_decay_wforth_skip = torch.tensor(
            weight_decay_wforth_skip, device=device
        )
        self.lambda_input_skip = torch.tensor(lambda_input_skip, device=device)
        self.lambda_input_output_skip = (
            torch.tensor(lambda_input_output_skip, device=device) * self.L
        )
        self.lr_input_skip = torch.tensor(lr_input_skip, device=device)
        self.weight_decay_input_skip = torch.tensor(
            weight_decay_input_skip, device=device
        )
        self.lr_input_output_skip = torch.tensor(lr_input_output_skip, device=device)
        self.weight_decay_input_output_skip = torch.tensor(
            weight_decay_input_output_skip, device=device
        )
        self.zero_out_cylinder_contribution = False  # TODO: remove this
        self.learn_free_ferromagnetic = (
            False  # TODO: check and enable this (or remove it)
        )
        self.zero_fc_init = zero_fc_init
        self.symmetrize_fc = symmetrize_fc
        self.symmetric_threshold_internal_couplings = (
            symmetric_threshold_internal_couplings
        )
        self.symmetrize_internal_couplings = symmetrize_internal_couplings

        self.root_H = torch.sqrt(torch.tensor(H, device=device))
        self.root_N = torch.sqrt(torch.tensor(N, device=device))
        self.root_C = torch.sqrt(torch.tensor(C, device=device))
        self.lr = lr.to(device)
        self.weight_decay = weight_decay.to(device)
        self.threshold = threshold.to(device)

        self.device = device
        self.generator = torch.Generator(device=self.device)
        self.cpu_generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)
            self.cpu_generator.manual_seed(seed)

        self.couplings = self.initialize_couplings(
            fc_left=fc_left, fc_right=fc_right
        )  # L+1, H, 3H
        self.symmetrize_couplings()
        self.prepare_tensors(
            lr,
            weight_decay,
            threshold,
            self.lr_input_skip,
            self.lr_input_output_skip,
            self.weight_decay_input_skip,
            self.weight_decay_input_output_skip,
        )

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
            f"init_noise={init_noise},\n"
            f"fc_left={fc_left},\n"
            f"fc_right={fc_right},\n"
            f"fc_input={fc_input},\n"
            f"symmetric_W={symmetric_W},\n"
            f"symmetric_J_init={symmetric_J_init},\n"
            f"double_dynamics={double_dynamics},\n"
            f"double_update={double_update},\n"
            f"use_local_ce={use_local_ce},\n"
            f"beta_ce={beta_ce},\n"
            f"p_update={p_update},\n"
            f"device={device},\n"
            f"seed={seed},\n"
        )

    def prepare_tensors(
        self,
        lr,
        weight_decay,
        threshold,
        lr_input_skip,
        lr_input_output_skip,
        weight_decay_input_skip,
        weight_decay_input_output_skip,
    ):
        self.is_learnable = self.build_is_learnable_mask(self.fc_left, self.fc_right)
        self.lr_tensor = self.build_lr_tensor(lr)
        self.weight_decay_tensor = self.build_weight_decay_tensor(weight_decay)
        self.threshold_tensor = threshold.to(self.device)
        self.ignore_right_mask = self.build_ignore_right_mask()  # 0: no; 1: yes; 2: yes, only label; 3: yes, only Wback feedback; 4: yes, label and Wback feedback.

        self.lr_input_skip_tensor = (
            torch.ones_like(self.input_skip, device=self.device)
            * lr_input_skip[:, None, None]
            * self.lambda_input_skip[:, None, None]
            / self.root_N
        )
        self.weight_decay_input_skip_tensor = (
            self.lr_input_skip_tensor * weight_decay_input_skip[:, None, None]
        )
        self.lr_input_output_skip_tensor = (
            torch.ones_like(self.input_output_skip, device=self.device)
            * lr_input_output_skip
            * self.lambda_input_output_skip
            / self.root_N
        )
        self.weight_decay_input_output_skip_tensor = (
            self.lr_input_output_skip_tensor * weight_decay_input_output_skip
        )

    def initialize_couplings(self, fc_left: bool, fc_right: bool):
        couplings_buffer = []
        self.H1 = 0
        # fc_left = fc_right = 0  # hack to set ferromagnetic to True everywhere

        # First Layer
        if self.fc_input:
            J_x = (
                sample_couplings(
                    self.N,
                    self.H,
                    self.device,
                    self.generator,
                    self.lambda_left[0] / self.lambda_fc[0],
                    0,
                    (not fc_left or self.zero_fc_init),
                )
                * self.lambda_fc[0]
            )
            # J_x = J_x * self.root_H / self.root_N  # only N terms are summed over
            J_x[:, self.N : self.H] = 0
        else:
            J_x = (
                torch.eye(self.H, device=self.device, dtype=DTYPE) * self.lambda_left[0]
            )
            for i in range(self.N, self.H):
                J_x[i, i] = 0
        couplings_buffer.append(J_x)
        couplings_buffer.append(
            sample_couplings(
                self.H1,
                self.H,
                self.device,
                self.generator,
                self.J_D[0],
                self.J_D[0],
                False,
                symmetric=self.symmetric_J_init,
            )
            * self.lambda_internal[0]
        )
        if self.L > 1:  # il L == 1, right couplings will be set later (W_back)
            couplings_buffer.append(
                sample_couplings(
                    self.H1,
                    self.H,
                    self.device,
                    self.generator,
                    0,
                    self.lambda_right[0] / self.lambda_fc[0],
                    (not fc_right or self.zero_fc_init),
                    self.zero_out_cylinder_contribution,
                )
                * self.lambda_fc[0]
            )

        # Middle Layers
        for idx in range(1, self.L - 1):
            couplings_buffer.append(
                sample_couplings(
                    self.H1,
                    self.H,
                    self.device,
                    self.generator,
                    0,
                    self.lambda_left[idx] / self.lambda_fc[idx],
                    (not fc_left or self.zero_fc_init),
                    self.zero_out_cylinder_contribution,
                )
                * self.lambda_fc[idx]
            )
            couplings_buffer.append(
                sample_couplings(
                    self.H1,
                    self.H,
                    self.device,
                    self.generator,
                    self.J_D[idx],
                    self.J_D[idx],
                    False,
                    symmetric=self.symmetric_J_init,
                )
                * self.lambda_internal[idx]
            )
            couplings_buffer.append(
                sample_couplings(
                    self.H1,
                    self.H,
                    self.device,
                    self.generator,
                    0,
                    self.lambda_right[idx] / self.lambda_fc[idx],
                    (not fc_right or self.zero_fc_init),
                    self.zero_out_cylinder_contribution,
                )
                * self.lambda_fc[idx]
            )

        # Last Layer
        if self.L > 1:  # il L == 1, left couplings have been set before
            couplings_buffer.append(
                sample_couplings(
                    self.H1,
                    self.H,
                    self.device,
                    self.generator,
                    0,
                    self.lambda_left[self.L - 1] / self.lambda_fc[self.L - 1],
                    (not fc_left or self.zero_fc_init),
                    self.zero_out_cylinder_contribution,
                )
                * self.lambda_fc[self.L - 1]
            )
            couplings_buffer.append(
                sample_couplings(
                    self.H1,
                    self.H,
                    self.device,
                    self.generator,
                    self.J_D[self.L - 1],
                    self.J_D[self.L - 1],
                    False,
                    symmetric=self.symmetric_J_init,
                )
                * self.lambda_internal[self.L - 1]
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
        # W_initial = sample_readout_weights(self.H, self.C, self.device, self.generator)
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

        # define skip connections, their learning rates and weight decay tensors
        self.Wback_skip = (
            W_back.clone().repeat(self.L - 1, 1, 1)
            / self.lambda_right[-2]
            * self.lambda_wback_skip[:, None, None]
        )
        self.Wforth_skip = (
            W_forth.clone().repeat(self.L - 1, 1, 1)
            / self.lambda_left[-1]
            * self.lambda_wforth_skip[:, None, None]
        )
        self.lr_Wback_skip = torch.zeros_like(self.Wback_skip)
        self.lr_Wforth_skip_tensor = (
            torch.ones_like(self.Wforth_skip, device=self.device)
            * self.lr_wforth_skip[:, None, None]
            * self.lambda_wforth_skip[:, None, None]
            / self.root_H
        )
        self.weight_decay_Wforth_skip_tensor = (
            self.lr_Wforth_skip_tensor * self.weight_decay_wforth_skip[:, None, None]
        )

        assert not self.fc_input
        assert self.lambda_left[0] == 0
        self.input_skip = (
            torch.randn(
                self.L,
                self.H,
                self.N,
                device=self.device,
                generator=self.generator,
                dtype=DTYPE,
            )
            / self.root_N
            * self.lambda_input_skip[:, None, None]
        )
        self.input_output_skip = (
            torch.randn(
                self.C,
                self.N,
                device=self.device,
                generator=self.generator,
                dtype=DTYPE,
            )
            / self.root_N
        ) * self.lambda_input_output_skip

        # p_flip = 0.0
        # for l in range(self.L - 1):
        #     for i in range(self.H):
        #         flip = torch.rand(1, generator=self.cpu_generator) < p_flip
        #         if flip:
        #             couplings[l, i, 2 * self.H + i] *= -1  # lambda_right
        #             couplings[l + 1, i, i] *= -1  # lambda_left

        return couplings.to(self.device)

    def build_is_learnable_mask(self, fc_left: bool, fc_right: bool):
        H, N, L, C = self.H, self.N, self.L, self.C

        mask = torch.ones_like(self.couplings)
        mask[-1, :, H:] = 0
        mask[-1, C:H, :H] = 0
        mask[-2, :, 2 * H + C :] = 0
        mask[0, :, :H] = 0
        if self.fc_input:
            mask[0, :, :N] = 1

        for idx in range(L):
            mask[idx, :, H : 2 * H].fill_diagonal_(0)
            if idx > 0:
                if fc_left:
                    if self.learn_free_ferromagnetic:
                        mask[idx, :N, :N].fill_diagonal_(0)
                    else:
                        mask[idx, :H, :H].fill_diagonal_(0)
                    if self.zero_out_cylinder_contribution:
                        mask[idx, :H, :N] = (
                            0  # NOTE: we are also zeroing out the diagonal up to N here (but it's 0 already)!
                        )
                else:
                    mask[idx, :H, :H] = 0
                    if self.learn_free_ferromagnetic:
                        mask[idx, N:H, N:H].fill_diagonal_(1)
            if idx < L - 1:
                if fc_right:
                    if self.learn_free_ferromagnetic:
                        mask[idx, :N, 2 * H : 2 * H + N].fill_diagonal_(0)
                    else:
                        mask[idx, :H, 2 * H : 3 * H].fill_diagonal_(0)
                    if self.zero_out_cylinder_contribution:
                        mask[idx, :H, 2 * H : 2 * H + N] = (
                            0  # NOTE: we are also zeroing out the diagonal up to N here (but it's 0 already)!
                        )
                else:
                    mask[idx, :H, 2 * H : 3 * H] = 0
                    if self.learn_free_ferromagnetic:
                        mask[idx, N:H, 2 * H + N : 3 * H].fill_diagonal_(1)

                # mask[idx, :N, 2 * H : 2 * H + N].fill_diagonal_(0)
                # if not fc_right:
                #     mask[idx, :, 2 * H : 3 * H] = 0
                # if self.zero_out_cylinder_contribution:
                #     mask[idx, :H, 2 * H : 2 * H + N] = 0

        return mask.to(self.device).to(torch.bool)

    def build_lr_tensor(self, lr):
        # TODO: if we decide to turn this on, we need to ensure that the lr tensor
        # has the appropriate magnitude for the free neurons ferromagnetic couplings
        # (which currently it hasn't)
        assert not self.learn_free_ferromagnetic
        # TODO: here should consider zero_out_cylinder_contribution for normalization
        assert not self.zero_out_cylinder_contribution

        H, L, C = self.H, self.L, self.C
        lr_tensor = torch.zeros_like(self.couplings)  # (L+1, H, 3H)
        for idx in range(L):  # NOTE: wback will be overwritten later
            lr_tensor[idx, :, :] = lr[idx] / math.sqrt(H)
            lr_tensor[idx, :H, H : 2 * H] *= self.lambda_internal[idx]
            lr_tensor[idx, :H, 2 * H : 3 * H] *= self.lambda_fc[
                idx
            ]  # NOTE: this multiplies the diagonal as well
            if idx > 0 or (idx == 0 and self.fc_input):
                lr_tensor[idx, :H, :H] *= self.lambda_fc[
                    idx
                ]  # NOTE: this multiplies the diagonal as well
        lr_tensor[L - 1, :H, 2 * H : 2 * H + C] = (
            lr[-2] * self.lambda_right[-2] / math.sqrt(C)
        )  # NOTE: overwrite W_back
        lr_tensor[L, :C, :H] = lr[-1] * self.lambda_left[-1] / math.sqrt(H)  # W_forth
        lr_tensor[self.is_learnable == 0] = (
            0  # this is because above we accidentally set to nonzero some fictitious couplings (e.g. the complement of Wback)
        )
        return lr_tensor.to(self.device)

    def build_weight_decay_tensor(self, weight_decay):
        H, L, C = self.H, self.L, self.C
        weight_decay_tensor = torch.zeros_like(self.couplings, device=self.device)
        for idx in range(L):
            weight_decay_tensor[idx, :, :] = weight_decay[idx]
        weight_decay_tensor[L - 1, :, 2 * H : 2 * H + C] = weight_decay[-2]
        weight_decay_tensor[L, :C, :H] = weight_decay[-1]
        weight_decay_tensor *= self.lr_tensor
        return weight_decay_tensor

    def build_ignore_right_mask(self):
        H, L = self.H, self.L
        mask = torch.ones_like(self.couplings).unsqueeze(0).repeat(5, 1, 1, 1)
        mask[1, :, :, 2 * H : 3 * H] = 0  # ignore all right fields
        mask[2, -1, :, 2 * H : 3 * H] = 0  # ignore only label's ferromagnetic field
        mask[3, -2, :, 2 * H : 3 * H] = 0  # ignore only W_back's feedback
        mask[4, -2:, :, 2 * H : 3 * H] = (
            0  # ignore W_back's feedback and label's ferromagnetic field
        )
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
            # y_hat = y.clone()
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
        # rescale importance of wrong class prototypes in wback field, while keeping total field roughly the same
        # state[:, -2, : self.C][state[:, -2, : self.C] == -1] = -1 / self.root_C
        # state[:, -2, : self.C] = state[:, -2, : self.C] * self.root_C / 2
        state[:, -2, : self.C] = torch.where(
            state[:, -2, : self.C] == -1,
            -0.5,
            state[:, -2, : self.C] * (self.root_C / 2),
        )

        state_unfolded = (
            state.unfold(1, 3, 1).transpose(-2, -1).flatten(2)
        )  # Shape: (B, L+1, 3*H)
        fields = torch.einsum(
            "lni,bli->bln",
            self.couplings * self.ignore_right_mask[ignore_right],
            state_unfolded,
        )
        if ignore_right in [0, 2]:
            fields[:, :-2, :] += torch.einsum(
                "lhc,bc->blh", self.Wback_skip, state[:, -2, : self.C]
            )
        fields[:, -1, : self.C] += torch.einsum(
            "lch,blh->bc", self.Wforth_skip, state[:, 1:-3, :]
        )
        fields[:, :-1, :] += torch.einsum(
            "lhn,bn->blh", self.input_skip, state[:, 0, : self.N]
        )
        fields[:, -1, : self.C] += torch.einsum(
            "cn,bn->bc", self.input_output_skip, state[:, 0, : self.N]
        )

        # assert self.L == 2
        # beta = 0.3
        # if ignore_right not in [1]:
        #     fields[:, 0, :] += torch.einsum(
        #         "hh,bh->bh",
        #         self.couplings[0, :, 2 * self.H : 3 * self.H],
        #         (F.tanh(self.last_fields[:, 1, :] * beta) - state[:, 2, :]),
        #     )
        # fields[:, 1, :] += torch.einsum(
        #     "hh,bh->bh",
        #     self.couplings[1, :, : self.H],
        #     (F.tanh(self.last_fields[:, 0, :] * beta) - state[:, 1, :]),
        # )
        # state1_masked = torch.where(
        #     self.mask1[None, :].expand(state.shape[0], -1),
        #     state[:, 1, :],
        #     0,
        # )
        # state2_masked = torch.where(
        #     self.mask2[None, :].expand(state.shape[0], -1),
        #     state[:, 2, :],
        #     0,
        # )
        # if ignore_right not in [1]:
        #     fields[:, 0, :] += torch.einsum(
        #         "hh,bh->bh",
        #         self.couplings[0, :, 2 * self.H : 3 * self.H],
        #         (state2_masked - state[:, 2, :]),
        #     )
        # fields[:, 1, :] += torch.einsum(
        #     "hh,bh->bh",
        #     self.couplings[1, :, : self.H],
        #     (state1_masked - state[:, 1, :]),
        # )

        # state[:, -2, : self.C] = torch.where(
        #     state[:, -2, : self.C] == -0.5,
        #     -1.0,
        #     state[:, -2, : self.C] / (self.root_C / 2),
        # )
        # self.last_fields = fields.clone()
        return fields

    def symmetrize_couplings(self):
        if self.symmetric_W == "buggy":
            self.couplings[-2, :, 2 * self.H : 2 * self.H + self.C] = (
                self.W_forth.T
                * self.root_H
                * self.lambda_right[-2]
                / self.root_C
                / self.lambda_left[-1]
                / 100
            )
            self.Wback_skip = (
                self.Wforth_skip.transpose(1, 2)
                * self.root_H
                * self.lambda_wback_skip[:, None, None]
                / self.root_C
                / self.lambda_wforth_skip[:, None, None]
                / 100
            )
        elif self.symmetric_W == "normalize":
            norm_old = self.W_back.norm(dim=0).mean()  # (C,)
            self.couplings[-2, :, 2 * self.H : 2 * self.H + self.C] = (
                self.W_forth / self.W_forth.norm(dim=1)[:, None]
            ).T * norm_old[None, None]

            if torch.any(self.lambda_wforth_skip == 0):
                assert torch.all(self.lambda_wforth_skip == 0)
                return
            norm_old = self.Wback_skip.norm(dim=1).mean(dim=1)  # (L-1, C)
            self.Wback_skip = (
                self.Wforth_skip / self.Wforth_skip.norm(dim=2)[:, :, None]
            ).transpose(1, 2) * norm_old[:, None, None]
        elif self.symmetric_W:
            self.couplings[-2, :, 2 * self.H : 2 * self.H + self.C] = (
                self.W_forth.T
                * self.root_H
                * self.lambda_right[-2]
                / self.root_C
                / self.lambda_left[-1]
            )
            self.Wback_skip = (
                self.Wforth_skip.transpose(1, 2)
                * self.root_H
                * self.lambda_wback_skip[:, None, None]
                / self.root_C
                / self.lambda_wforth_skip[:, None, None]
            )
        else:
            pass

        if self.symmetrize_fc and self.fc_left:
            for l in range(self.L - 1):
                self.couplings[l, :, 2 * self.H : 3 * self.H] = (
                    self.couplings[l + 1, :, : self.H].T
                    * self.lambda_fc[l]
                    / self.lambda_fc[l + 1]
                )

        if self.symmetrize_internal_couplings:
            self.couplings[:-1, :, self.H : 2 * self.H] = (
                self.couplings[:-1, :, self.H : 2 * self.H]
                + self.couplings[:-1, :, self.H : 2 * self.H].transpose(1, 2)
            ) / 2

    def perceptron_rule(
        self,
        state: torch.Tensor,
        delta_mask: Optional[torch.Tensor] = None,
    ):
        fields = self.local_field(state, ignore_right=IGNORE_RIGHT)  # shape (B, L+1, H)
        neurons = state[:, 1:-1, :]  # shape (B, L+1, H)
        S_unfolded = state.unfold(1, 3, 1).transpose(-2, -1)  # shape (B, L+1, 3, H)
        if self.use_local_ce:
            is_unstable = 1 - torch.sigmoid(
                self.beta_ce * (fields * neurons - self.threshold_tensor[None, :, None])
            )  # omit a beta_ce factor to decouple slope and lr
        else:
            is_unstable = (
                (fields * neurons) <= self.threshold_tensor[None, :, None]
            ).float()
        delta = (
            self.lr_tensor
            * torch.einsum("bli,blcj->licj", neurons * is_unstable, S_unfolded).flatten(
                2
            )
            / math.sqrt(state.shape[0])
        )
        if delta_mask is not None:
            delta = delta * delta_mask

        if self.symmetric_threshold_internal_couplings:
            delta[:-1, :, self.H : 2 * self.H] = torch.where(
                (delta[:-1, :, self.H : 2 * self.H] > 0)
                & (delta[:-1, :, self.H : 2 * self.H].transpose(1, 2) > 0),
                (
                    delta[:-1, :, self.H : 2 * self.H]
                    + delta[:-1, :, self.H : 2 * self.H].transpose(1, 2)
                )
                / 2,
                0,
            )

        self.couplings = self.couplings * (1 - self.weight_decay_tensor) + delta

        delta_skip = (
            self.lr_Wforth_skip_tensor
            * torch.einsum(
                "bc,blh->lch",
                neurons[:, -1, : self.C] * is_unstable[:, -1, : self.C],
                state[:, 1:-3, :],
            )
            / math.sqrt(state.shape[0])
        )
        self.Wforth_skip = (
            self.Wforth_skip * (1 - self.weight_decay_Wforth_skip_tensor) + delta_skip
        )

        delta_input_skip = (
            self.lr_input_skip_tensor
            * torch.einsum(
                "blh,bn->lhn",
                neurons[:, :-1, :] * is_unstable[:, :-1, :],
                state[:, 0, : self.N],
            )
            / math.sqrt(state.shape[0])
        )
        self.input_skip = (
            self.input_skip * (1 - self.weight_decay_input_skip_tensor)
            + delta_input_skip
        )
        delta_input_output_skip = (
            self.lr_input_output_skip_tensor
            * torch.einsum(
                "bc,bn->cn",
                neurons[:, -1, : self.C] * is_unstable[:, -1, : self.C],
                state[:, 0, : self.N],
            )
            / math.sqrt(state.shape[0])
        )
        self.input_output_skip = (
            self.input_output_skip * (1 - self.weight_decay_input_output_skip_tensor)
            + delta_input_output_skip
        )

        self.symmetrize_couplings()  # W_back <- W_forth (with appropriate scaling)
        return is_unstable

    def relax(
        self,
        state: torch.Tensor,
        max_steps: int,
        ignore_right: int,
    ):
        sweeps = 0
        # self.last_fields = torch.zeros_like(state[:, 1:-1, :], device=self.device)
        # self.mask1 = torch.rand_like(state[0, 1, :], dtype=torch.float32) < 0.5
        # self.mask2 = torch.rand_like(state[0, 2, :], dtype=torch.float32) < 0.5
        while sweeps < max_steps:
            ir = ignore_right if sweeps >= self.L else 1
            fields = self.local_field(state, ignore_right=ir)
            update_mask = torch.rand_like(fields) < self.p_update
            state[:, 1:-1, :] = torch.where(
                update_mask,
                torch.sign(fields),
                state[:, 1:-1, :],
            )
            # torch.sign(fields, out=state[:, 1:-1, :])
            sweeps += 1
        unsat = self.fraction_unsat(state, ignore_right=ignore_right)
        # first_is_one = state[:, 1:-1, 0] == 1
        # state[:, 1:-1, :] = torch.where(
        #     first_is_one.unsqueeze(-1),
        #     state[:, 1:-1, :],
        #     -state[:, 1:-1, :],
        # )
        # state[:, 1:-1, 0] = 0
        return state, sweeps, unsat

    def double_relax(self, state, max_sweeps):
        sweeps = 0
        while sweeps < max_sweeps:
            ir = 0 if sweeps >= self.L else 1
            fields = self.local_field(state, ignore_right=ir)
            update_mask = torch.rand_like(fields) < self.p_update
            state[:, 1:-1, :] = torch.where(
                update_mask,
                torch.sign(fields),
                state[:, 1:-1, :],
            )
            # state[:, 1:-1, :] = torch.sign(fields)
            sweeps += 1
        fields = self.local_field(state, ignore_right=0)
        first_unsat = state[:, 1:-1, :] != torch.sign(fields)
        first_fixed_point = state.clone()
        while sweeps < 2 * max_sweeps:
            fields = self.local_field(state, ignore_right=3)
            update_mask = torch.rand_like(fields) < self.p_update
            state[:, 1:-1, :] = torch.where(
                update_mask,
                torch.sign(fields),
                state[:, 1:-1, :],
            )
            # state[:, 1:-1, :] = torch.sign(fields)
            sweeps += 1
        fields = self.local_field(state, ignore_right=3)
        second_unsat = state[:, 1:-1, :] != torch.sign(fields)
        return first_fixed_point, state, sweeps, first_unsat, second_unsat

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        max_sweeps: int,
    ):
        state = self.initialize_state(x, y, self.init_mode)
        if self.double_dynamics:
            # Dynamics with annealing
            first_fixed_point, final_state, num_sweeps, _, unsat = self.double_relax(
                state, max_sweeps
            )
        else:
            # Simple dynamics
            final_state, num_sweeps, unsat = self.relax(
                state, max_sweeps, ignore_right=0
            )
        if self.double_update:
            # Double update (J on fixed point with external field, W on the one without it)
            made_update = self.make_double_update(first_fixed_point, final_state)

            # # Double update for the case without double dynamics
            # # Note: this does not quite work.
            # J_mask = torch.ones_like(self.couplings, device=self.device)
            # J_mask[-1, :, :] = 0  # readout row
            # J_mask[-2, :, 2 * self.H :] = 0  # Wback square
            # made_update = self.perceptron_rule(
            #     final_state,
            #     delta_mask=J_mask,
            # )
            # final_state_inference, _, _ = self.relax(
            #     state, max_sweeps, ignore_right=4
            # )
            # W_mask = torch.zeros_like(self.couplings, device=self.device)
            # W_mask[-1, :, :] = 1  # readout row
            # W_mask[-2, :, 2 * self.H :] = 1  # Wback square
            # self.perceptron_rule(
            #     final_state_inference,
            #     delta_mask=W_mask,
            # )
        else:
            # Simple update
            made_update = self.perceptron_rule(final_state)

        return {
            "sweeps": num_sweeps,
            "hidden_updates": made_update[:, :-1, :],
            "readout_updates": made_update[:, -1, : self.C],
            "hidden_unsat": unsat[:, :-1, :],
            "readout_unsat": unsat[:, -1, : self.C],
            "update_states": final_state[:, 1:-2, :],
        }

    def make_double_update(self, J_fixed_point, W_fixed_point):
        J_mask = torch.ones_like(self.couplings, device=self.device)
        J_mask[-1, :, :] = 0  # readout row
        J_mask[-2, :, 2 * self.H :] = 0  # Wback square
        made_update = self.perceptron_rule(
            J_fixed_point,
            delta_mask=J_mask,
        )
        W_mask = torch.zeros_like(self.couplings, device=self.device)
        W_mask[-1, :, :] = 1  # readout row
        W_mask[-2, :, 2 * self.H :] = 1  # Wback square
        self.perceptron_rule(
            W_fixed_point,
            delta_mask=W_mask,
        )
        return made_update

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
        final_state, num_sweeps, unsat = self.relax(
            state, max_sweeps, ignore_right=IGNORE_RIGHT
        )
        logits = self.local_field(final_state, ignore_right=IGNORE_RIGHT)[
            :, -1, : self.C
        ]
        # logits = final_state[:, -3] @ self.couplings[-1, : self.C, : self.H].T
        # logits += torch.einsum("lch,blh->bc", self.Wforth_skip, final_state[:, 1:-3, :])
        # logits += torch.einsum(
        #     "cn,bn->bc", self.input_output_skip, state[:, 0, : self.N]
        # )
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
        right[:, :-2, :] += torch.einsum(
            "lhc,bc->blh", self.Wback_skip, state[:, -2, : self.C]
        )
        left[:, -1, : self.C] += torch.einsum(
            "lch,blh->bc", self.Wforth_skip, state[:, 1:-3, :]
        )
        left[:, -1, : self.C] += torch.einsum(
            "cn,bn->bc", self.input_output_skip, state[:, 0, : self.N]
        )
        left[:, :-1, :] += torch.einsum(
            "lhn,bn->blh", self.input_skip, state[:, 0, : self.N]
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
