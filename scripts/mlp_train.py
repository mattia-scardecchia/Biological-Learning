import json
import logging
import os

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.train import get_data
from src.mlp import (
    MLPClassifier,
    SimpleDataModule,
    get_callbacks,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="mlp", version_base="1.3")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    pl.seed_everything(cfg.seed)

    train_inputs, train_targets, eval_inputs, eval_targets, C = get_data(cfg)
    data_module = SimpleDataModule(
        train_inputs,
        train_targets,
        eval_inputs,
        eval_targets,
        batch_size=cfg.dataloader.batch_size,
        num_workers=10,
    )
    data_module.setup()

    # Model
    model = MLPClassifier(
        input_dim=data_module.input_dim,
        hidden_dims=cfg.model.hidden_dims,
        num_classes=data_module.num_classes,
        dropout_rate=cfg.model.dropout_rate,
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        optimizer=cfg.optimizer.name,
        scheduler=cfg.scheduler.name if cfg.scheduler.enabled else None,
        scheduler_params=cfg.scheduler.params if cfg.scheduler.enabled else None,
    )

    # Get callbacks
    callbacks = get_callbacks(cfg)

    # Initialize logger
    logger_dir = os.path.join(cfg.logging.dir, cfg.experiment_name)
    tb_logger = TensorBoardLogger(save_dir=logger_dir, name="")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        deterministic=True,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
    )

    # Train model
    logger.info("Starting training...")
    trainer.fit(model, data_module)

    # Test model
    logger.info("Evaluating on eval set...")
    eval_results = trainer.test(model, data_module)
    logger.info(f"Eval results: {eval_results}")

    # Save test results alongside model
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    logger.info(f"Training completed. Model and logs saved to {trainer.log_dir}")


if __name__ == "__main__":
    main()
