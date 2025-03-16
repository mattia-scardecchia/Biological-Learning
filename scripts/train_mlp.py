import json
import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from src.mlp import DataModule, MLPClassifier, get_callbacks

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="mlp", version_base="1.3")
def main(config: DictConfig):
    pl.seed_everything(config.seed)

    # Data
    data_module = DataModule(
        dataset_config=config.dataset,
        batch_size=config.dataloader.batch_size,
        val_split=config.dataloader.val_split,
        test_split=config.dataloader.test_split,
    )
    data_module.setup()

    # Model
    model = MLPClassifier(
        input_dim=data_module.input_dim,
        hidden_dims=config.model.hidden_dims,
        num_classes=data_module.num_classes,
        dropout_rate=config.model.dropout_rate,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        optimizer=config.optimizer.name,
        scheduler=config.scheduler.name if config.scheduler.enabled else None,
        scheduler_params=config.scheduler.params if config.scheduler.enabled else None,
    )

    # Get callbacks
    callbacks = get_callbacks(config)

    # Initialize logger
    logger_dir = os.path.join(config.logging.dir, config.experiment_name)
    tb_logger = TensorBoardLogger(save_dir=logger_dir, name="")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=config.logging.log_every_n_steps,
        deterministic=True,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
    )

    # Train model
    logger.info("Starting training...")
    trainer.fit(model, data_module)

    # Test model
    logger.info("Evaluating on test set...")
    test_results = trainer.test(model, data_module)
    logger.info(f"Test results: {test_results}")

    # Save test results alongside model
    with open(os.path.join(logger_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    logger.info(f"Training completed. Model and logs saved to {trainer.log_dir}")


if __name__ == "__main__":
    main()
