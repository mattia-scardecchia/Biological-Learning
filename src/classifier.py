import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F


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


class Classifier:
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
        self.L = num_layers
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

        self.internal_couplings = self.initialize_couplings()  # num_layers, N, N
        self.diagonal_mask = torch.stack(
            [torch.eye(N, device=device, dtype=torch.bool)] * num_layers
        )
        self.W_forth = self.initialize_readout_weights()  # C, N
        self.W_back = self.W_forth.clone()  # C, N
        self.W_forth /= math.sqrt(N)
        self.W_back /= math.sqrt(C)

        self.lr = lr
        self.threshold = threshold
        self.weight_decay = weight_decay

        logging.info(f"Initialized {self} on device: {self.device}")
        logging.info(
            f"Parameters:\n"
            f"N={N},\n"
            f"C={C},\n"
            f"num_layers={num_layers},\n"
            f"J_D={J_D},\n"
            f"lambda_left={lambda_left},\n"
            f"lambda_right={lambda_right},\n"
            f"lr={lr},\n"
            f"threshold={threshold},\n"
            f"weight_decay={weight_decay}\n"
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
        for _ in range(self.L):
            J = torch.randn(
                self.N, self.N, device=self.device, generator=self.generator
            )
            J /= math.sqrt(self.N)
            J.fill_diagonal_(self.J_D.item())
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

    def initialize_state(self, x: torch.Tensor, y: Optional[torch.Tensor]):
        """
        Initializes the state of the neurons within each layer, and
        in the readout layer
        :param y: only for compatibility with the handler.
        """
        batch_size = x.shape[0]
        states = [x.clone() for _ in range(self.L)]
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
        return torch.matmul(states[layer_idx], self.internal_couplings[layer_idx].T)

    def internal_field(self, states: torch.Tensor):
        """
        For each neuron in the network, excluding those in the readout layer,
        computes the internal field.
        :param states: tensor of shape (num_layers, batch_size, N).
        """
        return torch.matmul(states, self.internal_couplings.transpose(1, 2))

    def left_field_layer(
        self, states: torch.Tensor, layer_idx: int, x: Optional[torch.Tensor] = None
    ):
        match layer_idx:
            case 0:
                if x is None:
                    return 0
                return x * self.lambda_left[0]
            case self.L:
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
        if layer_idx == self.L:
            if y is None:
                return 0
            return (2 * y - 1) * self.lambda_right[-1]
        elif layer_idx == self.L - 1:
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
        readout_right = (
            (2 * y - 1) * self.lambda_right[-1] if y is not None else torch.tensor(0)
        )
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
            if layer_idx < self.L
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
        state: tuple[torch.Tensor, torch.Tensor],
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        max_steps: int = 100,
        ignore_right: bool = False,
    ):
        states, readout = state
        steps = 0
        while steps < max_steps:
            steps += 1
            hidden_field, readout_field = self.local_field(
                states, readout, x, y, ignore_right=ignore_right
            )
            new_states = self.sign(hidden_field)
            new_readout = self.sign(readout_field)
            states, readout = new_states, new_readout
        hidden_unsat, readout_unsat = self.fraction_unsat_neurons(states, readout, x, y)
        return (states, readout), steps, (hidden_unsat.permute(1, 0, 2), readout_unsat)

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
        is_unstable_hidden = (
            (hidden_field * states) <= threshold[:-1, None, None]
        ).float()
        delta_J = (
            lr[:-2, None, None]
            * torch.matmul((is_unstable_hidden * states).transpose(1, 2), states)
            / math.sqrt(self.N)
            / math.sqrt(x.shape[0])
        )
        self.internal_couplings = (
            self.internal_couplings
            * (
                1
                - weight_decay[:-2, None, None]
                * lr[:-2, None, None]
                / math.sqrt(self.N)
            )
            + delta_J
        )
        self.internal_couplings[self.diagonal_mask] = self.J_D

        # update W_back
        delta_W_back = (
            (lr[-2] * torch.matmul((is_unstable_hidden[-1] * states[-1]).T, readout).T)
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

        return is_unstable_hidden.permute(1, 0, 2), is_unstable_readout

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        max_steps: int,
    ):
        states, readout = self.initialize_state(x, y)
        (final_states, final_readout), num_sweeps, (hidden_unsat, readout_unsat) = (
            self.relax((states, readout), x, y, max_steps)
        )
        hidden_updates, readout_updates = self.perceptron_rule_update(
            final_states,
            final_readout,
            x,
            y,
            self.lr,
            self.threshold,
            self.weight_decay,
        )
        return {
            "sweeps": num_sweeps,
            "hidden_updates": hidden_updates,  # B, L, N
            "readout_updates": readout_updates,  # B, C
            "hidden_unsat": hidden_unsat,  # B, L, N
            "readout_unsat": readout_unsat,  # B, C
        }

    def inference(self, x: torch.Tensor, max_steps: int):
        initial_states, initial_readout = self.initialize_state(x, y=None)
        (states, readout), _, _ = self.relax(
            (initial_states, initial_readout),
            x,
            y=None,
            max_steps=max_steps,
            ignore_right=True,
        )
        # logits = torch.matmul(states[-1], self.W_forth.T)
        logits = self.left_field_layer(states, self.L, x)
        return logits, states.permute(1, 0, 2), readout

    def fraction_unsat_neurons(
        self,
        states,
        readout,
        x: torch.Tensor,
        y: torch.Tensor,
        ignore_right: bool = False,
    ):
        hidden_field, readout_field = self.local_field(
            states, readout, x, y, ignore_right
        )
        is_unsat_hidden = (hidden_field * states) <= 0
        is_unsat_readout = (readout_field * readout) <= 0
        return is_unsat_hidden, is_unsat_readout

    @property
    def left_couplings(self):
        return torch.stack(
            [torch.eye(self.N) * self.lambda_left[idx] for idx in range(1, self.L)]
        )

    @property
    def right_couplings(self):
        return torch.stack(
            [torch.eye(self.N) * self.lambda_right[idx] for idx in range(self.L - 1)]
        )

    @property
    def input_couplings(self):
        return torch.eye(self.N) * self.lambda_left[0]

    @property
    def output_couplings(self):
        return torch.eye(self.C) * self.lambda_right[-1]

    def field_breakdown(self, state, x, y):
        states, readout = state
        hidden_internal = self.internal_field(states)  # L, B, N
        hidden_right, readout_right = self.right_field(states, readout, y)
        hidden_left, readout_left = self.left_field(states, x)
        readout_left = F.pad(
            readout_left, (0, self.N - self.C), mode="constant", value=0
        )
        readout_right = F.pad(
            readout_right, (0, self.N - self.C), mode="constant", value=0
        )
        hidden_total = hidden_internal + hidden_left + hidden_right
        readout_total = readout_left + readout_right
        return {
            "internal": hidden_internal.permute(
                1, 0, 2
            ),  # NOTE: only internal excludes readout layer
            "left": torch.cat([hidden_left, readout_left.unsqueeze(0)], dim=0).permute(
                1, 0, 2
            ),
            "right": torch.cat(
                [hidden_right, readout_right.unsqueeze(0)], dim=0
            ).permute(1, 0, 2),
            "total": torch.cat(
                [hidden_total, readout_total.unsqueeze(0)], dim=0
            ).permute(1, 0, 2),
        }
