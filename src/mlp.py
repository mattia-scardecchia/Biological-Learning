import logging
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class MLPClassifier(pl.LightningModule):
    """
    A simple MLP classifier implemented with PyTorch Lightning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        optimizer: str = "adam",
        scheduler: Optional[str] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the MLP classifier.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            optimizer: Optimizer type ("adam", "sgd", "adamw")
            scheduler: Learning rate scheduler ("step", "cosine", "plateau", None)
            scheduler_params: Parameters for the scheduler
        """
        super().__init__()
        self.save_hyperparameters()

        # Build network layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            nn.LayerNorm(prev_dim)
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Metrics for tracking
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)

    def _shared_step(self, batch, batch_idx):
        """Common step for training, validation and testing"""
        inputs, targets = batch
        # Convert one-hot encoded targets to class indices
        if targets.shape[1] > 1:  # Check if targets are one-hot encoded
            targets = torch.argmax(targets, dim=1)

        # Forward pass
        logits = self(inputs.float())
        loss = F.cross_entropy(logits, targets)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        """Training step"""
        loss, preds, targets = self._shared_step(batch, batch_idx)

        # Log metrics
        self.train_acc(preds, targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss, preds, targets = self._shared_step(batch, batch_idx)

        # Log metrics
        self.val_acc(preds, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step"""
        loss, preds, targets = self._shared_step(batch, batch_idx)

        # Log metrics
        self.test_acc(preds, targets)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Select optimizer
        if self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")

        # Return optimizer if no scheduler is specified
        if self.hparams.scheduler is None:
            return optimizer

        # Configure scheduler
        scheduler_params = self.hparams.scheduler_params or {}

        if self.hparams.scheduler.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_params.get("step_size", 10),
                gamma=scheduler_params.get("gamma", 0.1),
            )
        elif self.hparams.scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_params.get("T_max", 10),
                eta_min=scheduler_params.get("eta_min", 1e-6),
            )
        elif self.hparams.scheduler.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_params.get("factor", 0.1),
                patience=scheduler_params.get("patience", 10),
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            raise ValueError(f"Unsupported scheduler: {self.hparams.scheduler}")

        return [optimizer], [scheduler]


class SimpleDataModule(pl.LightningDataModule):
    def __init__(
        self, train_inputs, train_targets, eval_inputs, eval_targets, batch_size
    ):
        super().__init__()
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.eval_inputs = eval_inputs
        self.eval_targets = eval_targets
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(self.train_inputs, self.train_targets)
        self.val_dataset = TensorDataset(self.eval_inputs, self.eval_targets)
        self.input_dim = self.train_inputs.shape[1]
        self.num_classes = self.train_targets.shape[1]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return self.val_dataloader()


def get_callbacks(config):
    """
    Create callbacks for the training process.

    Args:
        config: A configuration object containing callback parameters

    Returns:
        List of callbacks
    """
    callbacks = []

    # Model checkpoint callback
    if hasattr(config, "checkpoint_callback") and config.checkpoint_callback.enabled:
        checkpoint_callback = ModelCheckpoint(
            monitor=config.checkpoint_callback.monitor,
            mode=config.checkpoint_callback.mode,
            save_top_k=config.checkpoint_callback.save_top_k,
            filename=config.checkpoint_callback.filename,
            dirpath=config.checkpoint_callback.dirpath,
            save_last=config.checkpoint_callback.save_last,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    # Early stopping callback
    if hasattr(config, "early_stopping") and config.early_stopping.enabled:
        early_stopping = EarlyStopping(
            monitor=config.early_stopping.monitor,
            min_delta=config.early_stopping.min_delta,
            patience=config.early_stopping.patience,
            verbose=True,
            mode=config.early_stopping.mode,
        )
        callbacks.append(early_stopping)

    return callbacks
