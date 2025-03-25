import logging
import os
import time

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt

from src.classifier.classifier import Classifier
from src.classifier.sparse_couplings_classifier import SparseCouplingsClassifier
from src.data import get_balanced_dataset
from src.utils import (
    plot_accuracy_by_class_barplot,
    plot_accuracy_history,
    plot_fixed_points_similarity_heatmap,
)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    train_data_dir = os.path.join(cfg.data.save_dir, "train")
    test_data_dir = os.path.join(cfg.data.save_dir, "test")
    rng = np.random.default_rng(cfg.seed)

    # ================== Data ==================
    train_inputs, train_targets, train_metadata, train_class_prototypes = (
        get_balanced_dataset(
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
    )
    eval_inputs, eval_targets, eval_metadata, eval_class_prototypes = (
        get_balanced_dataset(
            cfg.N,
            cfg.data.P,
            cfg.data.C,
            cfg.data.p,
            test_data_dir,
            train_class_prototypes,
            rng,
            shuffle=False,
            load_if_available=True,
            dump=True,
        )
    )

    # ================== Model ==================
    model_kwargs = {
        "num_layers": cfg.num_layers,
        "N": cfg.N,
        "C": cfg.data.C,
        "lambda_left": cfg.lambda_left,
        "lambda_right": cfg.lambda_right,
        "lambda_x": cfg.lambda_x,
        "lambda_y": cfg.lambda_y,
        "J_D": cfg.J_D,
        "rng": rng,
        "sparse_readout": cfg.sparse_readout,
    }
    if cfg.sparse_couplings:
        model_kwargs["sparsity_level"] = cfg.sparsity_level
        classifier_cls = SparseCouplingsClassifier
    else:
        classifier_cls = Classifier
    model = classifier_cls(**model_kwargs)

    init_plots_dir = os.path.join(output_dir, "init")
    os.makedirs(init_plots_dir)
    fig1, fig2 = model.plot_fields_histograms(x=train_inputs[0], y=train_targets[0])
    fig1.suptitle("Fields Breakdown at Initialization, with external fields")
    fig1.savefig(os.path.join(init_plots_dir, "fields_breakdown.png"))
    plt.close(fig1)
    fig2.suptitle("Total Field at Initialization, with external fields")
    fig2.savefig(os.path.join(init_plots_dir, "total_field.png"))
    plt.close(fig2)
    fig3 = model.plot_couplings_histograms()
    fig3.suptitle("Couplings at Initialization")
    fig3.savefig(os.path.join(init_plots_dir, "couplings.png"))
    plt.close(fig3)

    # ================== Training ==================
    t0 = time.time()
    train_acc_history, eval_acc_history = model.train_loop(
        cfg.num_epochs,
        train_inputs,
        train_targets,
        cfg.max_steps,
        cfg.lr,
        cfg.threshold,
        cfg.eval_interval,
        eval_inputs,
        eval_targets,
        rng,
    )
    t1 = time.time()
    logging.info(f"Training took {t1 - t0:.2f} seconds")

    eval_metrics = model.evaluate(eval_inputs, eval_targets, cfg.max_steps, rng)
    logging.info(f"Final Eval Accuracy: {eval_metrics['overall_accuracy']:.2f}")
    t2 = time.time()
    logging.info(f"Evaluation took {t2 - t1:.2f} seconds")

    # ================== Plotting ==================
    fig = plot_fixed_points_similarity_heatmap(eval_metrics["fixed_points"])
    plt.savefig(os.path.join(output_dir, "eval_representations_similarity.png"))
    plt.close(fig)

    fig = plot_accuracy_by_class_barplot(eval_metrics["accuracy_by_class"])
    plt.savefig(os.path.join(output_dir, "eval_accuracy_by_class.png"))
    plt.close(fig)

    eval_epochs = np.arange(1, cfg.num_epochs + 1, cfg.eval_interval)
    fig = plot_accuracy_history(train_acc_history, eval_acc_history, eval_epochs)
    plt.savefig(os.path.join(output_dir, "accuracy_history.png"))
    plt.close(fig)

    fig = model.plot_couplings_histograms()
    fig.suptitle("Couplings at the end of training")
    plt.savefig(os.path.join(output_dir, "couplings.png"))
    plt.close(fig)

    logging.info("best train accuracy: {:.2f}".format(np.max(train_acc_history)))
    logging.info("best eval accuracy: {:.2f}".format(np.max(eval_acc_history)))


if __name__ == "__main__":
    main()
