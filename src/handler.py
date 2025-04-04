import logging
import math
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.batch_me_if_u_can import BatchMeIfUCan
from src.classifier import Classifier


class Handler:
    def __init__(self, classifier: BatchMeIfUCan | Classifier):
        self.classifier = classifier

    def evaluate(
        self,
        x: torch.Tensor,
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
        fixed_points = {idx: states[:, idx] for idx in range(self.classifier.L)}
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
        for i in range(0, num_samples, batch_size):
            batch_idxs = idxs_perm[i : i + batch_size]
            x = inputs[batch_idxs]
            y = targets[batch_idxs]
            out = self.classifier.train_step(x, y, max_steps)
            metrics["sweeps"].append(out["sweeps"])
        return metrics

    def flush_logs(self):
        self.logs = {
            "train_acc_history": [],
            "train_representations": {},
            "eval_acc_history": [],
            "eval_representations": {},
        }

    def log(self, metrics, type):
        self.logs[f"{type}_acc_history"].append(metrics["overall_accuracy"])
        if self.logs[f"{type}_representations"] == {}:
            num_inputs = len(metrics["fixed_points"][0])
            input_idxs = np.random.choice(
                range(num_inputs),
                min(self.classifier.C * 30, num_inputs),
                replace=False,
            )
            input_idxs.sort()
            for i in input_idxs:
                self.logs[f"{type}_representations"][int(i)] = []
        else:
            for i, lst in self.logs[f"{type}_representations"].items():
                lst.append(
                    [
                        metrics["fixed_points"][idx][i]
                        for idx in range(self.classifier.L)
                    ]
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
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}:\n"
                f"Avg number of full sweeps: {np.mean(out['sweeps']):.1f}\n"
                f"Train Acc: {train_metrics['overall_accuracy'] * 100:.1f}%\n"
            )  # NOTE: we log accuracy of the model before training epoch
            if (epoch + 1) % eval_interval == 0:
                logging.info(
                    f"Eval Acc: {eval_metrics['overall_accuracy'] * 100:.1f}%\n"
                )

        self.logs["eval_representations"] = {
            i: np.array([[t.cpu() for t in sublist] for sublist in reps])
            for i, reps in self.logs["eval_representations"].items()
        }
        self.logs["train_representations"] = {
            i: np.array([[t.cpu() for t in sublist] for sublist in reps])
            for i, reps in self.logs["train_representations"].items()
        }
        return self.logs

    def fields_histogram(self, x, y, max_steps=0, ignore_right=0, plot_total=False):
        state = self.classifier.initialize_state(x, y)
        if max_steps:
            kwargs = {"x": x, "y": y} if isinstance(self.classifier, Classifier) else {}
            state, _ = self.classifier.relax(
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
