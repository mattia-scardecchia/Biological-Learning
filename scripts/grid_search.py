import itertools
import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt

from scripts.train import (
    get_data,
    parse_config,
    plot_fields_breakdown,
    plot_representation_similarity,
)
from src.batch_me_if_u_can import BatchMeIfUCan
from src.handler import Handler
from src.utils import (
    plot_accuracy_history,
    plot_couplings_distro_evolution,
    plot_couplings_histograms,
)


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

    # Convention: check if there is a key hp-name_values in cfg.
    # If not, use the single value of hp-name in cfg.
    # In any case, always pass the params to the model drawing from HYPERPARAM_GRID.
    HYPERPARAM_GRID = {
        "H": cfg.get("H_values", [cfg.H]),
        "J_D": cfg.get("J_D_values", [cfg.J_D]),
        "lambda_wback": cfg.get("lambda_wback_values", [cfg.lambda_wback]),
    }

    results_file = os.path.join(output_dir, "grid_search_results.csv")
    header_written = False
    i = 0
    for values in itertools.product(*HYPERPARAM_GRID.values()):
        i += 1
        logging.info(f"Starting iteration {i}")
        hyperparams = dict(zip(HYPERPARAM_GRID.keys(), values))

        # lr = [hyperparams["lr_J"]] * cfg.num_layers + [
        #     hyperparams["lr_wback"],
        #     hyperparams["lr_wforth"],
        # ]
        # threshold = [hyperparams["threshold_hidden"]] * cfg.num_layers + [
        #     hyperparams["threshold_readout"]
        # ]
        # weight_decay = [hyperparams["weight_decay_J"]] * cfg.num_layers + [
        #     hyperparams["weight_decay_wback"],
        #     hyperparams["weight_decay_wforth"],
        # ]
        # symmetric_W = hyperparams["symmetric_W"]

        J_D = hyperparams["J_D"]
        lambda_right[-2] = hyperparams["lambda_wback"]
        max_steps = hyperparams["max_steps"]
        H = hyperparams["H"]

        # ================== Model Training ==================

        run_repr = "_".join(
            [
                f"{key}_{val}"
                for key, val in hyperparams.items()
                if len(HYPERPARAM_GRID[key]) > 1
            ]
        )
        plots_dir = os.path.join(
            output_dir,
            f"plots_{run_repr}",
        )

        model_kwargs = {
            "num_layers": cfg.num_layers,
            "N": cfg.N,
            "C": C,
            "lambda_left": lambda_left,
            "lambda_right": lambda_right,
            "lambda_internal": cfg.lambda_internal,
            "J_D": J_D,
            "device": cfg.device,
            "seed": cfg.seed,
            "lr": torch.tensor(lr),
            "threshold": torch.tensor(threshold),
            "weight_decay": torch.tensor(weight_decay),
            "init_mode": cfg.init_mode,
            "init_noise": cfg.init_noise,
            "symmetric_W": cfg.symmetric_W,
            "double_dynamics": cfg.double_dynamics,
            "double_update": cfg.double_update,
            "use_local_ce": cfg.use_local_ce,
            "beta_ce": cfg.beta_ce,
            "fc_left": cfg.fc_left,
            "fc_right": cfg.fc_right,
            "fc_input": cfg.fc_input,
            "lambda_fc": cfg.lambda_fc,
            "H": H,
        }
        model_cls = BatchMeIfUCan
        model = model_cls(**model_kwargs)
        handler = Handler(
            model,
            cfg.init_mode,
            cfg.skip_representations,
            cfg.skip_couplings,
            output_dir,
            cfg.begin_curriculum,
            cfg.p_curriculum,
        )

        # Fields init
        fields_plots_dir = os.path.join(plots_dir, "fields")
        os.makedirs(fields_plots_dir, exist_ok=True)
        if not cfg.skip_fields:
            init_plots_dir = os.path.join(fields_plots_dir, "init")
            os.makedirs(init_plots_dir, exist_ok=True)
            idxs = np.random.randint(0, len(train_inputs), 100)
            x = train_inputs[idxs]
            y = train_targets[idxs]
            plot_fields_breakdown(
                handler,
                cfg,
                init_plots_dir,
                "Field Breakdown at Initialization",
                x,
                y,
            )

        logs = handler.train_loop(
            cfg.num_epochs,
            train_inputs,
            train_targets,
            max_steps,
            cfg.batch_size,
            eval_interval=cfg.eval_interval,
            eval_inputs=eval_inputs,
            eval_targets=eval_targets,
        )

        # Fields final
        if not cfg.skip_fields:
            final_plots_dir = os.path.join(fields_plots_dir, "final")
            os.makedirs(final_plots_dir, exist_ok=True)
            plot_fields_breakdown(
                handler,
                cfg,
                final_plots_dir,
                "Field Breakdown at the End of Training",
                train_inputs,
                train_targets,
            )

        # Accuracy history
        eval_epochs = np.arange(1, cfg.num_epochs + 1, cfg.eval_interval)
        fig = plot_accuracy_history(
            logs["train_acc_history"], logs["eval_acc_history"], eval_epochs
        )
        plt.savefig(os.path.join(plots_dir, "accuracy_history.png"))
        plt.close(fig)

        # Representations
        if not cfg.skip_representations:
            representations_root_dir = os.path.join(plots_dir, "representations")
            os.makedirs(representations_root_dir, exist_ok=True)
            plot_representation_similarity(logs, representations_root_dir, cfg)

        # Couplings
        if not cfg.skip_couplings:
            couplings_root_dir = os.path.join(plots_dir, "couplings")
            os.makedirs(couplings_root_dir, exist_ok=True)
            figs = plot_couplings_histograms(logs, [0, cfg.num_epochs - 1])
            for key, fig in figs.items():
                fig.savefig(os.path.join(couplings_root_dir, f"{key}.png"))
                plt.close(fig)
            figs = plot_couplings_distro_evolution(logs)
            for key, fig in figs.items():
                fig.savefig(os.path.join(couplings_root_dir, f"{key}_evolution.png"))
                plt.close(fig)

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
