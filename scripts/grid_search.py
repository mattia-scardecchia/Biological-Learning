import itertools
import logging
import os

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

from src.classifier import Classifier
from src.data import get_balanced_dataset

HYPERPARAM_GRID = {
    "lr": [0.01, 0.003, 0.001, 0.0003, 0.0001],
    "threshold": [6.0, 5.0, 4.0, 3.0, 2.0],
    "lambda_left": [4.0, 3.0, 2.0, 1.0],
    "lambda_x": [7.0, 6.0, 5.0, 4.0],
}


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    train_data_dir = os.path.join(cfg.data.save_dir, "train")
    test_data_dir = os.path.join(cfg.data.save_dir, "test")

    rng = np.random.default_rng(cfg.seed)
    inputs, targets, _, _ = get_balanced_dataset(
        cfg.N,
        cfg.data.P,
        cfg.data.C,
        cfg.data.p,
        train_data_dir,
        rng,
        shuffle=True,
        load_if_available=True,
        dump=True,
    )
    test_inputs, test_targets, _, _ = get_balanced_dataset(
        cfg.N,
        cfg.data.P,
        cfg.data.C,
        cfg.data.p,
        test_data_dir,
        rng,
        shuffle=True,
        load_if_available=True,
        dump=True,
    )

    results = []
    i = 0
    for values in itertools.product(*HYPERPARAM_GRID.values()):
        i += 1
        logging.info(f"Starting iteration {i}")
        hyperparams = dict(zip(HYPERPARAM_GRID.keys(), values))
        hyperparams["lambda_right"] = hyperparams["lambda_left"]
        hyperparams["lambda_y"] = hyperparams["lambda_x"]

        model = Classifier(
            cfg.num_layers,
            cfg.N,
            cfg.data.C,
            hyperparams["lambda_left"],
            hyperparams["lambda_right"],
            hyperparams["lambda_x"],
            hyperparams["lambda_y"],
            cfg.J_D,
            rng,
            sparse_readout=cfg.sparse_readout,
        )

        acc_history = model.train_loop(
            cfg.num_epochs,
            inputs,
            targets,
            cfg.max_steps,
            hyperparams["lr"],
            hyperparams["threshold"],
            rng,
        )

        max_train_acc = max(acc_history) if acc_history else 0.0
        metrics = model.evaluate(test_inputs, test_targets, cfg.max_steps, rng)
        test_accuracy = metrics["overall_accuracy"]
        logging.info(f"Test accuracy: {test_accuracy:.2f}\n")
        results.append(
            {**hyperparams, "max_train_acc": max_train_acc, "test_acc": test_accuracy}
        )
        logging.info(
            f"Summary.\nParams: {hyperparams}\nTrain Acc: {max_train_acc:.2f}, Test Acc: {test_accuracy:.2f}\n"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "grid_search_results.csv"), index=False)
    print("Hyperparameter tuning completed. Results saved.")


if __name__ == "__main__":
    main()
