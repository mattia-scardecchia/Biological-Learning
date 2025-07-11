import cProfile
import logging
import os
import pstats
import time

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt

from scripts.train import dump_stats, get_data, parse_config, plot_fields_breakdown
from src.batch_me_if_u_can import BatchMeIfUCan
from src.handler import Handler
from src.utils import (
    plot_accuracy_by_class_barplot,
    plot_accuracy_history,
    plot_couplings_distro_evolution,
    plot_couplings_histograms,
    plot_representation_similarity_among_inputs,
    plot_representations_similarity_among_layers,
)


def plot_representation_similarity(logs, save_dir, cfg, num_epochs):
    for representations, dirname in zip(
        [
            logs["update_representations"],
            logs["eval_representations"],
            logs["train_representations"],
        ],
        ["update", "eval", "train"],
    ):
        plot_dir = os.path.join(save_dir, dirname)
        os.makedirs(plot_dir, exist_ok=True)
        for epoch in np.linspace(
            0, num_epochs, min(5, num_epochs), endpoint=False
        ).astype(int):
            fig = plot_representation_similarity_among_inputs(
                representations, epoch, layer_skip=1
            )
            plt.savefig(os.path.join(plot_dir, f"epoch_{epoch}.png"))
            plt.close(fig)
        for input_idx in np.random.choice(
            list(representations.keys()), 3, replace=False
        ):
            fig = plot_representations_similarity_among_layers(
                representations, input_idx, 5
            )
            plt.savefig(os.path.join(plot_dir, f"input_{input_idx}.png"))
            plt.close(fig)
        fig = plot_representations_similarity_among_layers(
            representations, None, 5, True
        )
        plt.savefig(os.path.join(plot_dir, "avg_over_inputs.png"))
        plt.close(fig)


@hydra.main(
    config_path="../configs", config_name="baseline_multi_layer", version_base="1.3"
)
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    lr, weight_decay, threshold, lambda_left, lambda_right = parse_config(cfg)
    lr_cfg, weight_decay_cfg, threshold_cfg = (
        torch.tensor(lr),
        torch.tensor(weight_decay),
        torch.tensor(threshold),
    )

    train_inputs, train_targets, eval_inputs, eval_targets, C = get_data(cfg)
    train_inputs = train_inputs.to(cfg.device)
    train_targets = train_targets.to(cfg.device)
    eval_inputs = eval_inputs.to(cfg.device)
    eval_targets = eval_targets.to(cfg.device)

    # ================== Model Initialization ==================
    model_kwargs = {
        "num_layers": cfg.num_layers,
        "N": cfg.N,
        "C": C,
        "lambda_left": lambda_left,
        "lambda_right": lambda_right,
        "lambda_internal": cfg.lambda_internal,
        "J_D": cfg.J_D,
        "device": cfg.device,
        "seed": cfg.seed,
        "lr": torch.tensor(lr),
        "threshold": torch.tensor(threshold),
        "weight_decay": torch.tensor(weight_decay),
        "init_mode": cfg.init_mode,
        "init_noise": cfg.init_noise,
        "symmetric_W": cfg.symmetric_W,
        "symmetric_J_init": cfg.symmetric_J_init,
        "double_dynamics": cfg.double_dynamics,
        "double_update": cfg.double_update,
        "use_local_ce": cfg.use_local_ce,
        "beta_ce": cfg.beta_ce,
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
        "lambda_input_skip": cfg.lambda_input_skip,
        "lambda_input_output_skip": cfg.lambda_input_output_skip,
        "lr_input_skip": cfg.lr_input_skip,
        "weight_decay_input_skip": cfg.weight_decay_input_skip,
        "lr_input_output_skip": cfg.lr_input_output_skip,
        "weight_decay_input_output_skip": cfg.weight_decay_input_output_skip,
        "zero_fc_init": cfg.zero_fc_init,
        "symmetrize_fc": cfg.symmetrize_fc,
        "symmetrize_internal_couplings": cfg.symmetrize_internal_couplings,
        "symmetric_threshold_internal_couplings": cfg.symmetric_threshold_internal_couplings,
        "symmetric_update_internal_couplings": cfg.symmetric_update_internal_couplings,
        "bias_std": cfg.bias_std,
        "inference_ignore_right": cfg.inference_ignore_right,
        "H": cfg.H,
    }
    model_cls = BatchMeIfUCan
    model = model_cls(**model_kwargs)
    handler = Handler(
        model,
        cfg.init_mode,
        cfg.skip_representations,
        cfg.skip_couplings,
        output_dir,
    )
    lr_input_skip = model.lr_input_skip.clone()
    lr_input_output_skip = model.lr_input_output_skip.clone()
    weight_decay_input_skip = model.weight_decay_input_skip.clone()
    weight_decay_input_output_skip = model.weight_decay_input_output_skip.clone()

    fields_plots_dir = os.path.join(output_dir, "fields")
    os.makedirs(fields_plots_dir, exist_ok=True)
    couplings_root_dir = os.path.join(output_dir, "couplings")
    os.makedirs(couplings_root_dir, exist_ok=True)
    idxs = np.random.randint(0, len(train_inputs), 1000)
    x = train_inputs[idxs]
    y = train_targets[idxs]

    # ================== Training ==================
    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.time()

    # === Phase 1: Train the readout only, with no readout feedback ===
    lr = lr_cfg.clone()
    lr[:-2] = 0.0
    weight_decay = weight_decay_cfg.clone()
    threshold = threshold_cfg.clone()
    model.prepare_tensors(
        lr,
        weight_decay,
        threshold,
        torch.zeros_like(lr_input_skip),
        lr_input_output_skip,
        weight_decay_input_skip,
        weight_decay_input_output_skip,
    )  # re-create lr tensor
    model.set_wback(torch.zeros_like(model.W_back))  # no wback feedback
    model.symmetric_W = (
        False  # otherwise, wback might be updated and start providing feedback
    )
    handler.begin_curriculum = 1.0
    handler.p_curriculum = 0.5

    # Fields before phase 1
    if not cfg.skip_fields:
        plots_dir = os.path.join(fields_plots_dir, "phase-1")
        os.makedirs(plots_dir, exist_ok=True)
        plot_fields_breakdown(
            handler,
            cfg,
            plots_dir,
            "Field Breakdown before Phase 1",
            x,
            y,
        )

    # Train
    if cfg.num_epochs_warmup > 0:
        logs_1 = handler.train_loop(
            cfg.num_epochs_warmup,
            train_inputs,
            train_targets,
            cfg.max_steps_train,
            max_steps_eval=cfg.max_steps_eval,
            batch_size=cfg.batch_size,
            eval_interval=cfg.eval_interval,
            eval_inputs=eval_inputs,
            eval_targets=eval_targets,
        )

        # Couplings evolution during phase 1
        if not cfg.skip_couplings:
            plots_dir = os.path.join(couplings_root_dir, "phase-1")
            os.makedirs(plots_dir, exist_ok=True)
            figs = plot_couplings_histograms(logs_1, [0, cfg.num_epochs_warmup - 1])
            for key, fig in figs.items():
                fig.savefig(os.path.join(plots_dir, f"{key}.png"))
                plt.close(fig)
            figs = plot_couplings_distro_evolution(logs_1)
            for key, fig in figs.items():
                fig.savefig(os.path.join(plots_dir, f"{key}_evolution.png"))
                plt.close(fig)

        # Fields after phase 1
        if not cfg.skip_fields:
            plots_dir = os.path.join(fields_plots_dir, "phase-1-end")
            os.makedirs(plots_dir, exist_ok=True)
            plot_fields_breakdown(
                handler,
                cfg,
                plots_dir,
                "Field Breakdown after Phase 1",
                x,
                y,
            )

    # copy wforth into wback, and re-set symmetric_W option
    model.set_wback(
        model.wforth2wback(model.W_forth)
    )  # because we had set it to 0. as if we were initializing wback and wforth anew, make them symmetric
    model.symmetric_W = cfg.symmetric_W
    model.symmetrize_couplings()  # for plots with buggy option

    # === Phase 2: Train the couplings only, with feedback from the readout ===
    lr = lr_cfg.clone()
    lr[-2:] = 0.0
    weight_decay = weight_decay_cfg.clone()
    threshold = threshold_cfg.clone()
    model.prepare_tensors(
        lr,
        weight_decay,
        threshold,
        lr_input_skip,
        lr_input_output_skip,
        weight_decay_input_skip,
        weight_decay_input_output_skip,
    )  # re-create lr tensor
    handler.begin_curriculum = cfg.begin_curriculum
    handler.p_curriculum = cfg.p_curriculum

    # Fields before phase 2
    if not cfg.skip_fields:
        plots_dir = os.path.join(fields_plots_dir, "phase-2")
        os.makedirs(plots_dir, exist_ok=True)
        plot_fields_breakdown(
            handler,
            cfg,
            plots_dir,
            "Field Breakdown before Phase 2",
            x,
            y,
        )

    # Train
    if cfg.num_epochs_couplings > 0:
        logs_2 = handler.train_loop(
            cfg.num_epochs_couplings,
            train_inputs,
            train_targets,
            cfg.max_steps_train,
            cfg.max_steps_eval,
            cfg.batch_size,
            eval_interval=cfg.eval_interval,
            eval_inputs=eval_inputs,
            eval_targets=eval_targets,
        )

        # Couplings evolution during phase2
        if not cfg.skip_couplings:
            plots_dir = os.path.join(couplings_root_dir, "phase-2")
            os.makedirs(plots_dir, exist_ok=True)
            figs = plot_couplings_histograms(logs_2, [0, cfg.num_epochs_couplings - 1])
            for key, fig in figs.items():
                fig.savefig(os.path.join(plots_dir, f"{key}.png"))
                plt.close(fig)
            figs = plot_couplings_distro_evolution(logs_2)
            for key, fig in figs.items():
                fig.savefig(os.path.join(plots_dir, f"{key}_evolution.png"))
                plt.close(fig)

        # Fields after phase 2
        if not cfg.skip_fields:
            plots_dir = os.path.join(fields_plots_dir, "phase-2-end")
            os.makedirs(plots_dir, exist_ok=True)
            plot_fields_breakdown(
                handler,
                cfg,
                plots_dir,
                "Field Breakdown after Phase 2",
                x,
                y,
            )

    # === Phase 2bis: train the full network as usual ===
    lr = lr_cfg.clone()
    weight_decay = weight_decay_cfg.clone()
    threshold = threshold_cfg.clone()
    model.prepare_tensors(
        lr,
        weight_decay,
        threshold,
        lr_input_skip,
        lr_input_output_skip,
        weight_decay_input_skip,
        weight_decay_input_output_skip,
    )  # re-create lr tensor
    handler.begin_curriculum = cfg.begin_curriculum
    handler.p_curriculum = cfg.p_curriculum

    # Fields before phase 2bis
    if not cfg.skip_fields:
        plots_dir = os.path.join(fields_plots_dir, "phase-2bis")
        os.makedirs(plots_dir, exist_ok=True)
        plot_fields_breakdown(
            handler,
            cfg,
            plots_dir,
            "Field Breakdown before Phase 2bis",
            x,
            y,
        )

    # Train
    if cfg.num_epochs_full > 0:
        logs_2 = handler.train_loop(
            cfg.num_epochs_full,
            train_inputs,
            train_targets,
            cfg.max_steps_train,
            cfg.max_steps_eval,
            cfg.batch_size,
            eval_interval=cfg.eval_interval,
            eval_inputs=eval_inputs,
            eval_targets=eval_targets,
        )  # assume either 2 or 2bis happens (overwrite logs_2)

        # Couplings evolution during phase 2bis
        if not cfg.skip_couplings:
            plots_dir = os.path.join(couplings_root_dir, "phase-2bis")
            os.makedirs(plots_dir, exist_ok=True)
            figs = plot_couplings_histograms(logs_2, [0, cfg.num_epochs_couplings - 1])
            for key, fig in figs.items():
                fig.savefig(os.path.join(plots_dir, f"{key}.png"))
                plt.close(fig)
            figs = plot_couplings_distro_evolution(logs_2)
            for key, fig in figs.items():
                fig.savefig(os.path.join(plots_dir, f"{key}_evolution.png"))
                plt.close(fig)

        # Fields after phase 2
        if not cfg.skip_fields:
            plots_dir = os.path.join(fields_plots_dir, "phase-2bis-end")
            os.makedirs(plots_dir, exist_ok=True)
            plot_fields_breakdown(
                handler,
                cfg,
                plots_dir,
                "Field Breakdown after Phase 2bis",
                x,
                y,
            )

    # === Phase 3: tune the readout weights, with no feedback from the readout ===
    lr = lr_cfg.clone()
    lr[:-2] = 0.0
    weight_decay = weight_decay_cfg.clone()
    threshold = threshold_cfg.clone()
    model.prepare_tensors(
        lr,
        weight_decay,
        threshold,
        torch.zeros_like(lr_input_skip),
        lr_input_output_skip,
        weight_decay_input_skip,
        weight_decay_input_output_skip,
    )  # re-create lr tensor
    model.set_wback(torch.zeros_like(model.W_back))
    model.symmetric_W = False
    handler.begin_curriculum = cfg.begin_curriculum_tuning
    handler.p_curriculum = cfg.p_curriculum_tuning

    # # reset wforth
    # wforth_fresh = sample_readout_weights(
    #     model.H, model.C, model.device, model.generator
    # )
    # model.couplings[-1, : model.C, : model.H] = wforth_fresh.T

    # Final fields before phase 3
    plots_dir = os.path.join(fields_plots_dir, "phase-3")
    os.makedirs(plots_dir, exist_ok=True)
    plot_fields_breakdown(
        handler,
        cfg,
        plots_dir,
        "Field Breakdown before Phase 3",
        train_inputs,
        train_targets,
    )

    # Train
    if cfg.num_epochs_tuning > 0:
        logs_3 = handler.train_loop(
            cfg.num_epochs_tuning,
            train_inputs,
            train_targets,
            cfg.max_steps_train,
            cfg.max_steps_eval,
            cfg.batch_size,
            eval_interval=cfg.eval_interval,
            eval_inputs=eval_inputs,
            eval_targets=eval_targets,
        )

        # Couplings evolution during phase 3
        if not cfg.skip_couplings:
            plots_dir = os.path.join(couplings_root_dir, "phase-3")
            os.makedirs(plots_dir, exist_ok=True)
            figs = plot_couplings_histograms(logs_3, [0, cfg.num_epochs_tuning - 1])
            for key, fig in figs.items():
                fig.savefig(os.path.join(plots_dir, f"{key}.png"))
                plt.close(fig)
            figs = plot_couplings_distro_evolution(logs_3)
            for key, fig in figs.items():
                fig.savefig(os.path.join(plots_dir, f"{key}_evolution.png"))
                plt.close(fig)

        # Fields after phase 3
        if not cfg.skip_fields:
            plots_dir = os.path.join(fields_plots_dir, "phase-3-end")
            os.makedirs(plots_dir, exist_ok=True)
            plot_fields_breakdown(
                handler,
                cfg,
                plots_dir,
                "Field Breakdown after Phase 3",
                x,
                y,
            )

    t1 = time.time()
    logging.info(f"Training took {t1 - t0:.2f} seconds")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.dump_stats(f"profile-{cfg.device}.stats")

    # ================== Evaluation and Plotting ==================

    # Evaluate final model and plot Accuracy
    if cfg.num_epochs_tuning > 0:
        eval_metrics = handler.evaluate(eval_inputs, eval_targets, cfg.max_steps_eval)
        logging.info(f"Final Eval Accuracy: {eval_metrics['overall_accuracy']:.2f}")
        t2 = time.time()
        logging.info(f"Evaluation took {t2 - t1:.2f} seconds")
        fig = plot_accuracy_by_class_barplot(eval_metrics["accuracy_by_class"])
        plt.savefig(os.path.join(output_dir, "eval_accuracy_by_class.png"))
        plt.close(fig)
        eval_epochs = np.arange(1, cfg.num_epochs_tuning + 1, cfg.eval_interval)
        fig = plot_accuracy_history(
            logs_3["train_acc_history"], logs_3["eval_acc_history"], eval_epochs
        )
        plt.savefig(os.path.join(output_dir, "accuracy_history.png"))
        plt.close(fig)

        logging.info(
            "Best train accuracy: {:.2f}".format(np.max(logs_3["train_acc_history"]))
        )
        logging.info(
            "Best eval accuracy: {:.2f}".format(np.max(logs_3["eval_acc_history"]))
        )

    # Representations
    if (
        not cfg.skip_representations
        and (cfg.num_epochs_couplings + cfg.num_epochs_full) > 0
    ):
        representations_root_dir = os.path.join(output_dir, "representations")
        os.makedirs(representations_root_dir, exist_ok=True)
        assert cfg.num_epochs_couplings == 0 or cfg.num_epochs_full == 0
        num_epochs = max(cfg.num_epochs_couplings, cfg.num_epochs_full)
        plot_representation_similarity(
            logs_2, representations_root_dir, cfg, num_epochs
        )

    dump_stats(output_dir, logs_2)


if __name__ == "__main__":
    # with torch.mps.profiler.profile(mode="interval", wait_until_completed=False):
    main()
