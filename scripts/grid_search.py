import itertools
import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig

from scripts.train import get_data, parse_config
from src.batch_me_if_u_can import BatchMeIfUCan
from src.classifier import Classifier
from src.handler import Handler

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
    lr, weight_decay, threshold, lambda_left, lambda_right = parse_config(cfg)

    train_inputs, train_targets, eval_inputs, eval_targets, C = get_data(cfg)
    train_inputs = train_inputs.to(cfg.device)
    train_targets = train_targets.to(cfg.device)
    eval_inputs = eval_inputs.to(cfg.device)
    eval_targets = eval_targets.to(cfg.device)

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
        #     + [hyperparams["lambda_l"]] * (cfg.num_layers - 1)
        #     + [1.0]
        # )
        # lambda_right = (
        #     [hyperparams["lambda_r"]] * (cfg.num_layers - 1)
        #     + [1.0]
        #     + [hyperparams["lambda_y"]]
        # )
        # weight_decay = [hyperparams["weight_decay_J"]] * cfg.num_layers + [
        #     hyperparams["weight_decay_W"]
        # ] * 2
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
            "lr": torch.tensor(lr),
            "threshold": torch.tensor(threshold),
            "weight_decay": torch.tensor(weight_decay),
            "init_mode": cfg.init_mode,
            "symmetric_W": cfg.symmetric_W,
        }
        model_cls = BatchMeIfUCan if cfg.fc else Classifier
        model = model_cls(**model_kwargs)
        handler = Handler(
            model,
            cfg.init_mode,
            True,
            True,
            output_dir,
        )

        logs = handler.train_loop(
            cfg.num_epochs,
            train_inputs,
            train_targets,
            cfg.max_steps,
            cfg.batch_size,
            eval_interval=cfg.eval_interval,
            eval_inputs=eval_inputs,
            eval_targets=eval_targets,
        )

        # ================== Log results ==================

        max_train_acc, final_train_acc = (
            np.max(logs["train_acc_history"]),
            logs["train_acc_history"][-1],
        )
        max_eval_acc, final_eval_acc = (
            np.max(logs["eval_acc_history"]),
            logs["eval_acc_history"][-1],
        )
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
