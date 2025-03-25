import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig

from src.classifier.classifier import Classifier
from src.classifier.sparse_couplings_classifier import SparseCouplingsClassifier
from src.classifier.torch_classifier import TorchClassifier
from src.data import get_balanced_dataset


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    # Set up directories and random generator
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    train_data_dir = os.path.join(cfg.data.save_dir, "train")
    test_data_dir = os.path.join(cfg.data.save_dir, "test")

    # -------------------- Data Loading --------------------
    # Load data for the numpy-based model
    train_inputs_np, train_targets_np, _, train_prototypes = get_balanced_dataset(
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
    eval_inputs_np, eval_targets_np, _, _ = get_balanced_dataset(
        cfg.N,
        cfg.data.P,
        cfg.data.C,
        cfg.data.p,
        test_data_dir,
        train_prototypes,
        rng,
        shuffle=False,
        load_if_available=True,
        dump=True,
    )

    # For the torch model, convert arrays to tensors
    train_inputs_torch = torch.tensor(train_inputs_np, dtype=torch.float32)
    train_targets_torch = torch.tensor(train_targets_np, dtype=torch.float32)
    eval_inputs_torch = torch.tensor(eval_inputs_np, dtype=torch.float32)
    eval_targets_torch = torch.tensor(eval_targets_np, dtype=torch.float32)

    # -------------------- Model Initialization --------------------
    # Initialize numpy-based model; choose sparse version if desired.
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
        model_np = SparseCouplingsClassifier(**model_kwargs)
    else:
        model_np = Classifier(**model_kwargs)

    # Initialize Torch model
    model_torch = TorchClassifier(
        num_layers=cfg.num_layers,
        N=cfg.N,
        C=cfg.data.C,
        lambda_left=cfg.lambda_left,
        lambda_right=cfg.lambda_right,
        lambda_x=cfg.lambda_x,
        lambda_y=cfg.lambda_y,
        J_D=cfg.J_D,
        device=cfg.device,  # e.g., "cpu" or "cuda"
        seed=cfg.seed,
    )

    # -------------------- Training --------------------
    model_np.train_step(train_inputs_np[0], train_targets_np[0], 100, 0.001, 1.5)
    model_torch.train_step(
        train_inputs_torch[0], train_targets_torch[0], 100, 0.001, 1.5
    )


if __name__ == "__main__":
    main()
