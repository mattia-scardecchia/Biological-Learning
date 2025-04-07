import logging
import math
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.batch_me_if_u_can import BatchMeIfUCan
from src.classifier import Classifier


class Handler:
    def __init__(
        self,
        classifier: BatchMeIfUCan | Classifier,
        skip_representations: bool = False,
        skip_couplings: bool = False,
    ):
        self.classifier = classifier
        self.skip_representations = skip_representations
        self.skip_couplings = skip_couplings

    def evaluate(
        self,
        x: torch.Tensor,  # B, N
        y: torch.Tensor,
        max_steps: int,
    ):
        logits, states, readout = self.classifier.inference(x, max_steps)
        predictions = torch.argmax(logits, dim=1)
        ground_truth = torch.argmax(y, dim=1)
        accuracy = (predictions == ground_truth).float().mean().item()
        accuracy_by_class = {}
        for cls in range(self.classifier.C):
            cls_mask = ground_truth == cls
            accuracy_by_class[cls] = (
                (predictions[cls_mask] == cls).float().mean().item()
            )
        similarity_to_input = torch.einsum("bln,bn->l", states, x) / (
            self.classifier.N * x.shape[0]
        )
        return {
            "overall_accuracy": accuracy,
            "accuracy_by_class": accuracy_by_class,
            "fixed_points": states,  # B, L, N
            "logits": logits,  # B, C
            "similarity_to_input": (similarity_to_input + 1) / 2,  # L,
        }

    def train_epoch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        max_steps: int,
        batch_size: int,
    ):
        """
        Trains the network for one epoch over the training set.
        Shuffles the dataset and processes mini-batches.
        :param inputs: tensor of shape (num_samples, N).
        :param targets: tensor of shape (num_samples, C).
        :param max_steps: maximum relaxation sweeps.
        :return: tuple (list of sweeps per batch, list of update counts per batch)
        """
        metrics = defaultdict(list)
        num_samples = inputs.shape[0]
        idxs_perm = torch.randperm(num_samples, generator=self.classifier.cpu_generator)
        for i in range(0, num_samples - batch_size + 1, batch_size):
            batch_idxs = idxs_perm[i : i + batch_size]
            x = inputs[batch_idxs]
            y = targets[batch_idxs]
            out = self.classifier.train_step(x, y, max_steps)
            metrics["sweeps"].append(out["sweeps"])
            metrics["hidden_updates"].append(out["hidden_updates"])
            metrics["readout_updates"].append(out["readout_updates"])
            metrics["hidden_unsat"].append(out["hidden_unsat"])
            metrics["readout_unsat"].append(out["readout_unsat"])
        for key in ["hidden_updates", "hidden_unsat"]:
            metrics[key] = torch.stack(metrics[key]).mean(
                dim=(0, 1, 3), dtype=torch.float32
            )
        for key in ["readout_updates", "readout_unsat"]:
            metrics[key] = torch.stack(metrics[key]).mean(
                dim=(0, 1, 2), dtype=torch.float32
            )
        return metrics

    def flush_logs(self):
        self.logs = {
            "train_acc_history": [],
            "eval_acc_history": [],
            "train_representations": [],
            "eval_representations": [],
            "W_forth": [],
            "W_back": [],
            "internal_couplings": [],
            "left_couplings": [],
            "right_couplings": [],
        }

    def log(self, metrics, type):
        # Accuracy
        self.logs[f"{type}_acc_history"].append(metrics["overall_accuracy"])

        # Representations
        if not self.skip_representations:
            eval_batch_size = len(metrics["fixed_points"])
            idxs = np.linspace(
                0,
                eval_batch_size,
                min(self.classifier.C * 30, eval_batch_size),
                endpoint=False,
            ).astype(int)  # NOTE: indexing is relative to the eval batch... hacky
            self.logs[f"{type}_representations"].append(
                metrics["fixed_points"][idxs, :, :].clone()
            )

        # Couplings
        if not self.skip_couplings:
            if type == "eval":
                self.logs["W_forth"].append(self.classifier.W_forth.clone())
                self.logs["W_back"].append(self.classifier.W_back.clone())
                self.logs["internal_couplings"].append(
                    self.classifier.internal_couplings.clone()
                )
                self.logs["left_couplings"].append(
                    self.classifier.left_couplings.clone()
                )
                self.logs["right_couplings"].append(
                    self.classifier.right_couplings.clone()
                )

    @torch.inference_mode()
    def train_loop(
        self,
        num_epochs: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        max_steps: int,
        batch_size: int,
        eval_interval: int,
        eval_inputs: torch.Tensor,
        eval_targets: torch.Tensor,
    ):
        """
        Trains the network for multiple epochs.
        Logs training accuracy and (optionally) validation accuracy.
        :param num_epochs: number of epochs.
        :param inputs: training inputs, shape (num_samples, N).
        :param targets: training targets, shape (num_samples, C).
        :param max_steps: maximum relaxation sweeps.
        :param batch_size: mini-batch size.
        :param eval_interval: evaluation interval in epochs.
        :param eval_inputs: validation inputs.
        :param eval_targets: validation targets.
        :return: tuple (train accuracy history, eval accuracy history)
        """
        if eval_interval is None:
            eval_interval = num_epochs + 1  # never evaluate
        self.flush_logs()

        for epoch in range(num_epochs):
            train_metrics = self.evaluate(inputs, targets, max_steps)
            self.log(train_metrics, type="train")

            if (epoch + 1) % eval_interval == 0:
                eval_metrics = self.evaluate(eval_inputs, eval_targets, max_steps)
                self.log(eval_metrics, type="eval")

            out = self.train_epoch(inputs, targets, max_steps, batch_size)

            message = (
                f"Epoch {epoch + 1}/{num_epochs}:\n"
                f"Full Sweeps: {np.mean(out['sweeps']):.1f}\n"
                "Unsat after Relaxation:  "
                f"{', '.join([format(x, '.2f') for x in (out['hidden_unsat'].tolist() + [float(out['readout_unsat'])])])}\n"
                "Perceptron Rule Updates: "
                f"{', '.join([format(x, '.2f') for x in (out['hidden_updates'].tolist() + [float(out['readout_updates'])])])}\n"
                "Similarity of Representations to Inputs: \n"
                f"   Train patterns:       {', '.join([format(x, '.2f') for x in train_metrics['similarity_to_input'].tolist()])}\n"
            )
            if (epoch + 1) % eval_interval == 0:
                message += f"   Eval patterns:        {', '.join([format(x, '.2f') for x in eval_metrics['similarity_to_input'].tolist()])}\n"
            message += f"\nTrain Acc: {train_metrics['overall_accuracy'] * 100:.1f}%\n"
            if (epoch + 1) % eval_interval == 0:
                message += f"Eval Acc:  {eval_metrics['overall_accuracy'] * 100:.1f}%\n"

            logging.info(message)  # NOTE: we log before training epoch

        if not self.skip_representations:
            for type in ["train", "eval"]:
                repr_tensor = torch.stack(
                    self.logs[f"{type}_representations"], dim=0
                ).permute(1, 0, 2, 3)  # B, T, L, N
                repr_dict = {
                    idx: repr_tensor[idx, :, :, :].cpu().numpy()
                    for idx in range(repr_tensor.shape[0])
                }
                self.logs[f"{type}_representations"] = repr_dict

        if not self.skip_couplings:
            for key in [
                "W_forth",  # T, C, N
                "W_back",  # T, N, C
                "internal_couplings",  # T, L, N, N
                "left_couplings",  # T, L, N, N
                "right_couplings",  # T, L, N, N
            ]:
                self.logs[key] = torch.stack(self.logs[key], dim=0).cpu().numpy()

        return self.logs

    def fields_histogram(self, x, y, max_steps=0, ignore_right=0, plot_total=False):
        state = self.classifier.initialize_state(x, y)
        if max_steps:
            kwargs = {"x": x, "y": y} if isinstance(self.classifier, Classifier) else {}
            state, _, _ = self.classifier.relax(
                state,
                max_steps=max_steps,
                ignore_right=ignore_right,
                **kwargs,
            )
        field_breakdown = self.classifier.field_breakdown(state, x, y)
        nrows = 2
        ncols = math.ceil((self.classifier.L + 1) / 2)
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(ncols * 5, nrows * 5), sharex=True
        )
        colors = ["blue", "green", "red", "grey"]

        # Hidden layers
        for idx, ax in enumerate(axs.flatten()[: self.classifier.L]):
            internal = field_breakdown["internal"][:, idx].cpu()
            left = field_breakdown["left"][:, idx].cpu()
            right = field_breakdown["right"][:, idx].cpu()
            total = field_breakdown["total"][:, idx].cpu()
            if not plot_total:
                ax.hist(
                    internal.flatten(),
                    bins=20,
                    density=False,
                    alpha=0.5,
                    label="internal",
                    color=colors[0],
                )
                ax.hist(
                    left.flatten(),
                    bins=20,
                    density=False,
                    alpha=0.5,
                    label="left",
                    color=colors[1],
                )
                ax.hist(
                    right.flatten(),
                    bins=20,
                    density=False,
                    alpha=0.5,
                    label="right",
                    color=colors[2],
                )
            else:
                ax.hist(
                    total.flatten(),
                    bins=20,
                    density=False,
                    alpha=0.5,
                    label="total",
                    color=colors[3],
                )
            ax.set_title(f"Layer {idx}")
            ax.grid()
            ax.legend()

        # Readout layer
        readout_left = field_breakdown["left"][:, -1, : self.classifier.C].cpu()
        readout_right = field_breakdown["right"][:, -1, : self.classifier.C].cpu()
        total = field_breakdown["total"][:, -1, : self.classifier.C].cpu()
        ax = axs.flatten()[self.classifier.L]
        if not plot_total:
            ax.hist(
                readout_left.flatten(),
                bins=20,
                density=False,
                alpha=0.5,
                label="left",
                color=colors[1],
            )
            ax.hist(
                readout_right.flatten(),
                bins=20,
                density=False,
                alpha=0.5,
                label="right",
                color=colors[2],
            )
        else:
            ax.hist(
                total.flatten(),
                bins=20,
                density=False,
                alpha=0.5,
                label="total",
                color=colors[3],
            )
        ax.set_title("Readout")
        ax.grid()
        ax.legend()

        return fig, axs
