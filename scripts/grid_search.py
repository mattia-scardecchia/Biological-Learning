import itertools
import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig

from src.classifier import BatchMeIfYouCan
from src.data import prepare_mnist
from src.utils import load_synthetic_dataset

# HYPERPARAM_GRID = {
#     "lr_J": [0.05, 0.075, 0.1],
#     "lr_W": [0.01, 0.02, 0.03],
#     "threshold": [1.25, 1.5, 1.75],
#     "weight_decay_J": [0.0, 0.001, 0.01],
#     "lambda_left": [1.75, 2.0, 2.25],
#     "lambda_x": [4.0, 5.0],
#     "J_D": [0.2, 0.3, 0.4],
#     "num_epochs": [10],
#     "weight_decay_W": [0.0],
# }
HYPERPARAM_GRID = {
    "lr_J": [0.1, 0.05],
    "lr_W": [0.01, 0.005],
    "threshold": [2.5, 3.0, 3.5],
}


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    rng = np.random.default_rng(cfg.seed)

    # ================== Data ==================
    if cfg.data.dataset == "synthetic":
        train_data_dir = os.path.join(cfg.data.synthetic.save_dir, "train")
        test_data_dir = os.path.join(cfg.data.synthetic.save_dir, "test")
        train_data_dir = os.path.join(cfg.data.synthetic.save_dir, "train")
        test_data_dir = os.path.join(cfg.data.synthetic.save_dir, "test")
        (
            train_inputs,
            train_targets,
            eval_inputs,
            eval_targets,
            train_metadata,
            train_class_prototypes,
            eval_metadata,
            eval_class_prototypes,
        ) = load_synthetic_dataset(
            cfg.N,
            cfg.data.P,
            cfg.data.synthetic.C,
            cfg.data.synthetic.p,
            cfg.data.P_eval,
            rng,
            train_data_dir,
            test_data_dir,
            cfg.device,
        )
        C = cfg.data.synthetic.C
    elif cfg.data.dataset == "mnist":
        train_inputs, train_targets, eval_inputs, eval_targets, projection_matrix = (
            prepare_mnist(
                cfg.data.P * 10,
                cfg.data.P_eval * 10,
                cfg.N,
                cfg.data.mnist.binarize,
                cfg.seed,
            )
        )
        C = 10
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")

    # ================== Begin Grid Search ==================

    results_file = os.path.join(output_dir, "grid_search_results.csv")
    header_written = False
    i = 0
    for values in itertools.product(*HYPERPARAM_GRID.values()):
        i += 1
        logging.info(f"Starting iteration {i}")
        hyperparams = dict(zip(HYPERPARAM_GRID.keys(), values))
        # lambda_left = (
        #     [hyperparams["lambda_x"]]
        #     + [hyperparams["lambda_left"]] * (cfg.num_layers - 1)
        #     + [1.0]
        # )
        # lambda_right = (
        #     [hyperparams["lambda_left"]] * (cfg.num_layers - 1)
        #     + [1.0]
        #     + [hyperparams["lambda_x"]]
        # )
        # weight_decay = [hyperparams["weight_decay_J"]] * cfg.num_layers + [
        #     hyperparams["weight_decay_W"]
        # ] * 2
        lambda_left = cfg.lambda_left
        lambda_right = cfg.lambda_right
        weight_decay = cfg.weight_decay
        lr = [hyperparams["lr_J"]] * cfg.num_layers + [hyperparams["lr_W"]] * 2
        threshold = [hyperparams["threshold"]] * (cfg.num_layers + 1)

        # ================== Model Training ==================

        model_kwargs = {
            "num_layers": cfg.num_layers,
            "N": cfg.N,
            "C": C,
            "lambda_left": lambda_left,
            "lambda_right": lambda_right,
            "J_D": cfg.J_D,
            "device": cfg.device,
            "seed": cfg.seed,
        }
        model = BatchMeIfYouCan(**model_kwargs)
        train_acc_history, eval_acc_history, eval_representations = model.train_loop(
            cfg.num_epochs,
            train_inputs,
            train_targets,
            cfg.max_steps,
            torch.tensor(lr),
            torch.tensor(threshold),
            torch.tensor(weight_decay),
            cfg.batch_size,
            cfg.eval_interval,
            eval_inputs,
            eval_targets,
        )

        # ================== Log results ==================

        max_train_acc, final_train_acc = (
            np.max(train_acc_history),
            train_acc_history[-1],
        )
        max_eval_acc, final_eval_acc = np.max(eval_acc_history), eval_acc_history[-1]
        result_row = {
            key: val
            for key, val in hyperparams.items()
            if len(HYPERPARAM_GRID[key]) > 1
        }
        result_row.update(
            {
                "max_train_acc": max_train_acc,
                "final_train_acc": final_train_acc,
                "max_eval_acc": max_eval_acc,
                "final_eval_acc": final_eval_acc,
            }
        )
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
