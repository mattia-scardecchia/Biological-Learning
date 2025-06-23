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


@hydra.main(config_path="../configs", config_name="baseline_1layer", version_base="1.3")
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
        "lambda_input_skip": cfg.get(
            "lambda_input_skip_values", [cfg.lambda_input_skip]
        ),
        "max_steps_train": cfg.get("max_steps_train_values", [cfg.max_steps_train]),
        "max_steps_eval": cfg.get("max_steps_eval_values", [cfg.max_steps_eval]),
        "threshold_hidden": cfg.get("threshold_hidden_values", [cfg.threshold[0]]),
        "threshold_readout": cfg.get("threshold_readout_values", [cfg.threshold[-1]]),
        "lr_J": cfg.get("lr_J_values", [cfg.lr_J]),
        "weight_decay_J": cfg.get("weight_decay_J_values", [cfg.weight_decay_J]),
        "beta_ce": cfg.get("beta_ce_values", [cfg.beta_ce]),
        "double_dynamics": cfg.get("double_dynamics_values", [cfg.double_dynamics]),
        "symmetric_J_init": cfg.get("symmetric_J_init_values", [cfg.symmetric_J_init]),
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
        lambda_input_skip = hyperparams["lambda_input_skip"]
        max_steps_train = hyperparams["max_steps_train"]
        max_steps_eval = hyperparams["max_steps_eval"]
        H = hyperparams["H"]
        beta_ce = hyperparams["beta_ce"]
        double_dynamics = hyperparams["double_dynamics"]
        symmetric_J_init = hyperparams["symmetric_J_init"]

        lambda_right[-2] = hyperparams["lambda_wback"]
        for i in range(0, cfg.num_layers):
            threshold[i] = hyperparams["threshold_hidden"]
            lr[i] = hyperparams["lr_J"]
            weight_decay[i] = hyperparams["weight_decay_J"]
        threshold[-1] = hyperparams["threshold_readout"]

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
            "symmetric_J_init": symmetric_J_init,
            "double_dynamics": double_dynamics,
            "double_update": cfg.double_update,
            "use_local_ce": cfg.use_local_ce,
            "beta_ce": beta_ce,
            "p_update": cfg.p_update,
            "fc_left": cfg.fc_left,
            "fc_right": cfg.fc_right,
            "fc_input": cfg.fc_input,
            "lambda_fc": cfg.lambda_fc,
            "lambda_cylinder": cfg.lambda_cylinder,
            "lambda_wback_skip": cfg.lambda_wback_skip,
            "lambda_wforth_skip": cfg.lambda_wforth_skip,
            "lr_wforth_skip": cfg.lr_wforth_skip,
            "weight_decay_wforth_skip": cfg.weight_decay_wforth_skip,
            "lambda_input_skip": lambda_input_skip,
            "lambda_input_output_skip": cfg.lambda_input_output_skip,
            "lr_input_skip": cfg.lr_input_skip,
            "weight_decay_input_skip": cfg.weight_decay_input_skip,
            "lr_input_output_skip": cfg.lr_input_output_skip,
            "weight_decay_input_output_skip": cfg.weight_decay_input_output_skip,
            "zero_fc_init": cfg.zero_fc_init,
            "symmetrize_fc": cfg.symmetrize_fc,
            "symmetrize_internal_couplings": cfg.symmetrize_internal_couplings,
            "symmetric_threshold_internal_couplings": cfg.symmetric_threshold_internal_couplings,
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
            max_steps_train,
            max_steps_eval,
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
