import logging
from collections import defaultdict

import numpy as np
import torch

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
        fixed_points[self.classifier.L] = readout
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
            sweeps = self.classifier.train_step(x, y, max_steps)
            metrics["sweeps"].append(sweeps)
        return metrics

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
        train_acc_history = []
        eval_acc_history = []
        representations = defaultdict(list)  # input, time, layer
        for epoch in range(num_epochs):
            convergence_metrics = self.train_epoch(
                inputs, targets, max_steps, batch_size
            )
            train_metrics = self.evaluate(inputs, targets, max_steps)
            avg_sweeps = (
                torch.tensor(convergence_metrics["sweeps"]).float().mean().item()
            )
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}:\n"
                f"Train Acc: {train_metrics['overall_accuracy']:.3f}\n"
                f"Avg number of full sweeps: {avg_sweeps:.3f}\n"
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
                            for idx in range(self.classifier.L)
                        ]
                    )
        representations = {
            i: np.array([[t.cpu() for t in sublist] for sublist in reps])
            for i, reps in representations.items()
        }
        return train_acc_history, eval_acc_history, representations
