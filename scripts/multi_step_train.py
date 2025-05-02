import cProfile
import logging
import os
import pstats
import time

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt

from src.batch_me_if_u_can import BatchMeIfUCan
from src.data import (
    load_synthetic_dataset,
    prepare_cifar,
    prepare_hm_data,
    prepare_mnist,
)
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


def plot_fields_breakdown(handler: Handler, cfg, save_dir, title, x, y):
    for max_steps in [0, cfg.max_steps]:
        for ignore_right in [0, 1]:
            for plot_total in [False, True]:
                fig, axs = handler.fields_histogram(
                    x, y, max_steps, ignore_right, plot_total
                )
                fig.suptitle(
                    title
                    + f". Relaxation: max_steps={max_steps}, ignore_right={ignore_right}"
                )
                fig.tight_layout()
                plt.savefig(
                    os.path.join(
                        save_dir,
                        f"{'field_breakdown' if not plot_total else 'total_field'}_{max_steps}_{ignore_right}.png",
                    )
                )
                plt.close(fig)


def get_data(cfg):
    if cfg.data.dataset == "synthetic":
        train_data_dir = os.path.join(cfg.data.synthetic.save_dir, "train")
        test_data_dir = os.path.join(cfg.data.synthetic.save_dir, "test")
        C = cfg.data.synthetic.C
        rng = np.random.default_rng(cfg.seed)
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
            C,
            cfg.data.synthetic.p,
            cfg.data.P_eval,
            rng,
            train_data_dir,
            test_data_dir,
            cfg.device,
        )
    elif cfg.data.dataset == "mnist":
        C = 10
        train_inputs, train_targets, eval_inputs, eval_targets, projection_matrix = (
            prepare_mnist(
                cfg.data.P * C,
                cfg.data.P_eval * C,
                cfg.N,
                cfg.data.mnist.binarize,
                cfg.seed,
                shuffle=True,
            )
        )
    elif cfg.data.dataset == "cifar":
        C = 10 if cfg.data.cifar.cifar10 else 100
        (
            train_inputs,
            train_targets,
            eval_inputs,
            eval_targets,
            projection_matrix,
            median,
        ) = prepare_cifar(
            cfg.data.P * C,
            cfg.data.P_eval * C,
            cfg.N,
            cfg.data.cifar.binarize,
            cfg.seed,
            cifar10=cfg.data.cifar.cifar10,
            shuffle=True,
        )
    elif cfg.data.dataset == "hm":
        C = cfg.data.hm.C
        (
            train_inputs,
            train_targets,
            eval_inputs,
            eval_targets,
            teacher_linear,
            teacher_mlp,
        ) = prepare_hm_data(
            cfg.data.hm.D,
            cfg.data.hm.C,
            cfg.data.P,
            cfg.data.P_eval,
            cfg.N,
            cfg.data.hm.L,
            cfg.data.hm.width,
            nn.ReLU(),
            cfg.seed,
            cfg.data.hm.binarize,
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")
    return train_inputs, train_targets, eval_inputs, eval_targets, C


def parse_config(cfg):
    try:
        lr = cfg.lr
    except Exception:
        lr = [cfg.lr_J] * cfg.num_layers + [cfg.lr_W] * 2
    try:
        weight_decay = cfg.weight_decay
    except Exception:
        weight_decay = [cfg.weight_decay_J] * cfg.num_layers + [cfg.weight_decay_W] * 2
    try:
        threshold = cfg.threshold
    except Exception:
        threshold = [cfg.threshold_hidden] * cfg.num_layers + [cfg.threshold_readout]
    try:
        lambda_left = cfg.lambda_left
    except Exception:
        lambda_left = [cfg.lambda_x] + [cfg.lambda_l] * (cfg.num_layers - 1) + [1.0]
    try:
        lambda_right = cfg.lambda_right
    except Exception:
        lambda_right = (
            [cfg.lambda_r] * (cfg.num_layers - 1) + [cfg.lambda_wback] + [cfg.lambda_y]
        )
    return lr, weight_decay, threshold, lambda_left, lambda_right


@hydra.main(
    config_path="../configs", config_name="multi_step_train", version_base="1.3"
)
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    lr, weight_decay, threshold, lambda_left, lambda_right = parse_config(cfg)

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
        "symmetric_W": cfg.symmetric_W,
    }
    assert cfg.fc_left or cfg.fc_right
    model_kwargs["fc_left"] = cfg.fc_left
    model_kwargs["fc_right"] = cfg.fc_right
    model_kwargs["lambda_fc"] = cfg.lambda_fc
    model_kwargs["H"] = cfg.H
    model_cls = BatchMeIfUCan

    model = model_cls(**model_kwargs)
    handler = Handler(
        model,
        cfg.init_mode,
        cfg.skip_representations,
        cfg.skip_couplings,
        output_dir,
    )

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
    lr = torch.tensor(cfg.lr)
    lr[:-2] = 0.0
    weight_decay = torch.tensor(cfg.weight_decay)
    threshold = torch.tensor(cfg.threshold)
    model.prepare_tensors(lr, weight_decay, threshold)
    model.set_wback(torch.zeros_like(model.W_back))

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
            cfg.max_steps,
            cfg.batch_size,
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

    # copy wforth into wback
    model.set_wback(model.wforth2wback(model.W_forth))

    # === Phase 2: Train the couplings only, with feedback from the readout ===
    lr = torch.tensor(cfg.lr)
    lr[-2:] = 0.0
    weight_decay = torch.tensor(cfg.weight_decay)
    threshold = torch.tensor(cfg.threshold)
    model.prepare_tensors(lr, weight_decay, threshold)

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
            cfg.max_steps,
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
    lr = torch.tensor(cfg.lr)
    weight_decay = torch.tensor(cfg.weight_decay)
    threshold = torch.tensor(cfg.threshold)
    model.prepare_tensors(lr, weight_decay, threshold)

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
            cfg.max_steps,
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
    lr = torch.tensor(cfg.lr)
    lr[:-2] = 0.0
    weight_decay = torch.tensor(cfg.weight_decay)
    threshold = torch.tensor(cfg.threshold)
    model.prepare_tensors(lr, weight_decay, threshold)
    model.set_wback(torch.zeros_like(model.W_back))

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
            cfg.max_steps,
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
        eval_metrics = handler.evaluate(eval_inputs, eval_targets, cfg.max_steps)
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
        plot_representation_similarity(
            logs_2, representations_root_dir, cfg, cfg.num_epochs_couplings
        )


if __name__ == "__main__":
    # with torch.mps.profiler.profile(mode="interval", wait_until_completed=False):
    main()
