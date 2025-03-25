import itertools
import logging
import os

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

from src.classifier.classifier import Classifier
from src.classifier.sparse_couplings_classifier import SparseCouplingsClassifier
from src.data import get_balanced_dataset

HYPERPARAM_GRID = {
    "lr": [0.007, 0.005, 0.003, 0.001],
    "threshold": [1.0, 1.5, 2.0, 2.5],
    "lambda_left": [1.0, 2.0, 3.0],
    "lambda_x": [4.0, 5.0, 6.0],
    "J_D": [0.2, 0.4],
}


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    train_data_dir = os.path.join(cfg.data.save_dir, "train")
    test_data_dir = os.path.join(cfg.data.save_dir, "test")
    rng = np.random.default_rng(cfg.seed)

    # ================== Data ==================
    inputs, targets, metadata, prototypes = get_balanced_dataset(
        cfg.N,
        cfg.data.P,
        cfg.data.C,
        cfg.data.p,
        train_data_dir,
        None,
        rng,
        shuffle=True,
        load_if_available=True,
        dump=True,
    )
    eval_inputs, eval_targets, _, _ = get_balanced_dataset(
        cfg.N,
        cfg.data.P,
        cfg.data.C,
        cfg.data.p,
        test_data_dir,
        prototypes,
        rng,
        shuffle=True,
        load_if_available=True,
        dump=True,
    )

    results_file = os.path.join(output_dir, "grid_search_results.csv")
    header_written = False
    i = 0
    for values in itertools.product(*HYPERPARAM_GRID.values()):
        i += 1
        logging.info(f"Starting iteration {i}")
        hyperparams = dict(zip(HYPERPARAM_GRID.keys(), values))
        hyperparams["lambda_right"] = hyperparams["lambda_left"]
        hyperparams["lambda_y"] = hyperparams["lambda_x"]

        # ================== Model ==================
        model_kwargs = {
            "num_layers": cfg.num_layers,
            "N": cfg.N,
            "C": cfg.data.C,
            "lambda_left": hyperparams["lambda_left"],
            "lambda_right": hyperparams["lambda_right"],
            "lambda_x": hyperparams["lambda_x"],
            "lambda_y": hyperparams["lambda_y"],
            "J_D": hyperparams["J_D"],
            "rng": rng,
            "sparse_readout": cfg.sparse_readout,
        }
        if cfg.sparse_couplings:
            model_kwargs["sparsity_level"] = cfg.sparsity_level
            classifier_cls = SparseCouplingsClassifier
        else:
            classifier_cls = Classifier
        model = classifier_cls(**model_kwargs)

        # ================== Training ==================
        train_acc_history, eval_acc_history = model.train_loop(
            cfg.num_epochs,
            inputs,
            targets,
            cfg.max_steps,
            hyperparams["lr"],
            hyperparams["threshold"],
            cfg.eval_interval,
            eval_inputs,
            eval_targets,
            rng,
        )

        max_train_acc, final_train_acc = (
            np.max(train_acc_history),
            train_acc_history[-1],
        )
        max_eval_acc, final_eval_acc = np.max(eval_acc_history), eval_acc_history[-1]
        result_row = {
            **hyperparams,
            "max_train_acc": max_train_acc,
            "final_train_acc": final_train_acc,
            "max_eval_acc": max_eval_acc,
            "final_eval_acc": final_eval_acc,
        }

        # ================== Log results ==================
        df_row = pd.DataFrame([result_row])
        if not header_written:
            df_row.to_csv(results_file, index=False, mode="w")
            header_written = True
        else:
            df_row.to_csv(results_file, index=False, mode="a", header=False)
        logging.info(
            f"Summary.\nParams: {hyperparams}\nFinal Train Acc: {final_train_acc:.2f}, Max Eval Acc: {max_eval_acc:.2f}\n"
        )

    logging.info(f"Hyperparameter tuning completed. Results saved in {results_file}")


if __name__ == "__main__":
    main()
