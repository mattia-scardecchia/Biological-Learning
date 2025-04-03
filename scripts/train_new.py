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

from src.batch_me_if_u_can import BatchMeIfUCan as Classifier
from src.data import prepare_cifar, prepare_mnist
from src.handler import Handler
from src.utils import (
    load_synthetic_dataset,
    plot_accuracy_by_class_barplot,
    plot_accuracy_history,
    plot_representation_similarity_among_inputs,
    plot_representations_similarity_among_layers,
)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir

    # ================== Data ==================
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
        train_inputs, train_targets, eval_inputs, eval_targets, median = prepare_cifar(
            cfg.data.P * C,
            cfg.data.P_eval * C,
            cfg.N,
            cfg.data.cifar.binarize,
            cfg.seed,
            cifar10=cfg.data.cifar.cifar10,
            shuffle=True,
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")

    train_inputs = train_inputs.to(cfg.device)
    train_targets = train_targets.to(cfg.device)
    eval_inputs = eval_inputs.to(cfg.device)
    eval_targets = eval_targets.to(cfg.device)

    # ================== Model Initialization ==================
    model_kwargs = {
        "num_layers": cfg.num_layers,
        "N": cfg.N,
        "C": C,
        "lambda_left": cfg.lambda_left,
        "lambda_right": cfg.lambda_right,
        "J_D": cfg.J_D,
        "device": cfg.device,
        "seed": cfg.seed,
        "lr": torch.tensor(cfg.lr),
        "threshold": torch.tensor(cfg.threshold),
        "weight_decay": torch.tensor(cfg.weight_decay),
    }
    model = Classifier(**model_kwargs)
    handler = Handler(model)

    # ================== Training ==================
    profiler = cProfile.Profile()
    profiler.enable()

    t0 = time.time()
    train_acc_history, eval_acc_history, eval_representations = handler.train_loop(
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

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.dump_stats(f"profile-{cfg.device}.stats")

    # ================== Evaluation and Plotting ==================
    if not cfg.skip_final_eval:
        eval_metrics = handler.evaluate(eval_inputs, eval_targets, cfg.max_steps)
        logging.info(f"Final Eval Accuracy: {eval_metrics['overall_accuracy']:.2f}")
        t2 = time.time()
        logging.info(f"Evaluation took {t2 - t1:.2f} seconds")

        fig = plot_accuracy_by_class_barplot(eval_metrics["accuracy_by_class"])
        plt.savefig(os.path.join(output_dir, "eval_accuracy_by_class.png"))
        plt.close(fig)

        eval_epochs = np.arange(1, cfg.num_epochs + 1, cfg.eval_interval)
        fig = plot_accuracy_history(train_acc_history, eval_acc_history, eval_epochs)
        plt.savefig(os.path.join(output_dir, "accuracy_history.png"))
        plt.close(fig)

        representations_plots_dir = os.path.join(output_dir, "representations")
        os.makedirs(representations_plots_dir, exist_ok=True)
        for epoch in np.linspace(0, cfg.num_epochs - 1, 3).astype(int):
            fig = plot_representation_similarity_among_inputs(
                eval_representations, epoch, layer_skip=1
            )
            plt.savefig(os.path.join(representations_plots_dir, f"epoch_{epoch}.png"))
            plt.close(fig)
        for input_idx in np.random.choice(len(eval_inputs), 3, replace=False):
            fig = plot_representations_similarity_among_layers(
                eval_representations, input_idx, 5
            )
            plt.savefig(
                os.path.join(representations_plots_dir, f"input_{input_idx}.png")
            )
            plt.close(fig)
        fig = plot_representations_similarity_among_layers(
            eval_representations, None, 5, True
        )
        plt.savefig(os.path.join(representations_plots_dir, "avg_over_inputs.png"))
        plt.close(fig)

    logging.info("Best train accuracy: {:.2f}".format(np.max(train_acc_history)))
    logging.info("Best eval accuracy: {:.2f}".format(np.max(eval_acc_history)))


if __name__ == "__main__":
    main()
