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
from src.classifier import Classifier
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


def plot_representation_similarity(logs, save_dir, cfg):
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
            0, cfg.num_epochs, min(5, cfg.num_epochs), endpoint=False
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


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
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
    if cfg.fc_left or cfg.fc_right:
        model_kwargs["fc_left"] = cfg.fc_left
        model_kwargs["fc_right"] = cfg.fc_right
        model_kwargs["lambda_fc"] = cfg.lambda_fc
        model_kwargs["H"] = cfg.H
        model_cls = BatchMeIfUCan
    else:
        model_cls = Classifier
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

    fields_plots_dir = os.path.join(output_dir, "fields")
    os.makedirs(fields_plots_dir, exist_ok=True)
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

    # ================== Training ==================
    profiler = cProfile.Profile()
    profiler.enable()

    # torch.mps.profiler.start()
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    t0 = time.time()
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
    t1 = time.time()
    logging.info(f"Training took {t1 - t0:.2f} seconds")
    # logging.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # torch.mps.profiler.stop()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.dump_stats(f"profile-{cfg.device}.stats")

    # ================== Evaluation and Plotting ==================
    # Field Breakdown
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

    # Evaluate final model and plot Accuracy
    eval_metrics = handler.evaluate(eval_inputs, eval_targets, cfg.max_steps)
    logging.info(f"Final Eval Accuracy: {eval_metrics['overall_accuracy']:.2f}")
    t2 = time.time()
    logging.info(f"Evaluation took {t2 - t1:.2f} seconds")
    fig = plot_accuracy_by_class_barplot(eval_metrics["accuracy_by_class"])
    plt.savefig(os.path.join(output_dir, "eval_accuracy_by_class.png"))
    plt.close(fig)
    eval_epochs = np.arange(1, cfg.num_epochs + 1, cfg.eval_interval)
    fig = plot_accuracy_history(
        logs["train_acc_history"], logs["eval_acc_history"], eval_epochs
    )
    plt.savefig(os.path.join(output_dir, "accuracy_history.png"))
    plt.close(fig)

    if not cfg.skip_representations:
        # Representations
        representations_root_dir = os.path.join(output_dir, "representations")
        os.makedirs(representations_root_dir, exist_ok=True)
        plot_representation_similarity(logs, representations_root_dir, cfg)

    # Couplings
    if not cfg.skip_couplings:
        couplings_root_dir = os.path.join(output_dir, "couplings")
        os.makedirs(couplings_root_dir, exist_ok=True)
        figs = plot_couplings_histograms(logs, [0, cfg.num_epochs - 1])
        for key, fig in figs.items():
            fig.savefig(os.path.join(couplings_root_dir, f"{key}.png"))
            plt.close(fig)
        figs = plot_couplings_distro_evolution(logs)
        for key, fig in figs.items():
            fig.savefig(os.path.join(couplings_root_dir, f"{key}_evolution.png"))
            plt.close(fig)

    logging.info(
        "Best train accuracy: {:.2f}".format(np.max(logs["train_acc_history"]))
    )
    logging.info("Best eval accuracy: {:.2f}".format(np.max(logs["eval_acc_history"])))


if __name__ == "__main__":
    # with torch.mps.profiler.profile(mode="interval", wait_until_completed=False):
    main()
