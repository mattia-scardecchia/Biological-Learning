import os

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt

from src.classifier import Classifier
from src.data import get_balanced_dataset


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    train_data_dir = os.path.join(cfg.data.save_dir, "train")
    test_data_dir = os.path.join(cfg.data.save_dir, "test")
    rng = np.random.default_rng(cfg.seed)

    # ================== Data ==================
    inputs, targets, metadata, class_prototypes = get_balanced_dataset(
        cfg.N,
        cfg.data.P,
        cfg.data.C,
        cfg.data.p,
        train_data_dir,
        rng,
        shuffle=True,
        load_if_available=True,
        dump=True,
    )
    test_inputs, test_targets, test_metadata, test_class_prototypes = (
        get_balanced_dataset(
            cfg.N,
            cfg.data.P,
            cfg.data.C,
            cfg.data.p,
            test_data_dir,
            rng,
            shuffle=True,
            load_if_available=True,
            dump=True,
        )
    )

    # ================== Model ==================
    model = Classifier(
        cfg.num_layers,
        cfg.N,
        cfg.data.C,
        cfg.lambda_left,
        cfg.lambda_right,
        cfg.lambda_x,
        cfg.lambda_y,
        cfg.J_D,
        rng,
        sparse_readout=cfg.sparse_readout,
    )

    fig1, fig2 = model.plot_fields_histograms(x=inputs[0], y=targets[0])
    fig1.suptitle("Fields Breakdown at Initialization, with external fields")
    fig1.savefig(os.path.join(output_dir, "fields_breakdown.png"))
    plt.close(fig1)
    fig2.suptitle("Total Field at Initialization, with external fields")
    fig2.savefig(os.path.join(output_dir, "total_field.png"))
    plt.close(fig2)

    # ================== Training ==================
    model.train_loop(
        cfg.num_epochs,
        inputs,
        targets,
        cfg.max_steps,
        cfg.lr,
        cfg.threshold,
        rng,
    )


if __name__ == "__main__":
    main()
