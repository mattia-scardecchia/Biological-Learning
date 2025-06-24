import torch
import torch
import numpy as np
import pickle
import os
import logging
from itertools import combinations
from typing import Dict
from omegaconf import DictConfig
import hydra
from src.data import prepare_mnist
from tabulate import tabulate

logger = logging.getLogger(__name__)


def load_model(epoch: int, model_path: str):
    logger.info(f"Loading model from epoch {epoch} from {model_path}")
    model_file = os.path.join(model_path, f"model_epoch_{epoch}.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} does not exist.")
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    return model


def relaxation_trajectory(classifier, x, y, steps, state=None):
    states = []
    unsats = []
    max_steps = max(steps)
    if state is None:
        state = classifier.initialize_state(x, y, "zeros")
    for step in range(max_steps):
        state, _, unsat = classifier.relax(
            state,
            max_steps=1,
            ignore_right=0,
        )
        if step + 1 in steps:
            # Store the state and unsat status
            states.append(state.clone())
            unsats.append(unsat.clone())
    states = torch.stack(states, dim=0)  # T, B, L, N
    states = states.permute(1, 0, 2, 3)  # B, T, L, N
    unsats = torch.stack(unsats, dim=0)  # T, B, L, N
    unsats = unsats.permute(1, 0, 2, 3)  # B, T, L, N
    return states, unsats


def compute_overlap_evolution(states, steps) -> Dict[str, torch.Tensor]:
    # data, time, state
    overlaps_stats = {}
    for time1, time2 in combinations(range(len(steps)), 2):
        state_1 = states[:, time1, :]
        state_2 = states[:, time2, :]
        overlaps = (state_1 * state_2).sum(dim=-1) / state_1.shape[-1]
        overlaps_stats[f"{steps[time1]}-{steps[time2]}"] = overlaps
    logger.info(f"Computed overlaps for {list(overlaps_stats.keys())}")
    return overlaps_stats


def plot_overlap_from_key(overlaps_stats, key):
    xy = [
        (float(k.split("-")[1]), overlaps_stats[k])
        for k in overlaps_stats.keys()
        if k.startswith(key)
    ]
    print(len(xy), "overlaps found for key", key)
    x = [item[0] for item in xy]
    y = [item[1].mean().item() for item in xy]
    y_err = [item[1].std().item() for item in xy]
    return x, y, y_err


def table_overlap_evolution(overlaps_stats, keys):
    rows = []
    for key in keys:
        x, y, y_err = plot_overlap_from_key(overlaps_stats, key)
        for xi, yi, yerri in zip(x, y, y_err):
            rows.append({"key": key, "y": yi, "y_err": yerri})
    print(tabulate(rows, headers="keys", tablefmt="github"))


@hydra.main(config_path="../configs", config_name="compute_overlap")
def main(cfg: DictConfig):
    P = cfg["P"]
    C = cfg["C"]
    P_eval = cfg["P_eval"]
    N = cfg["N"]
    binarize = cfg["binarize"]
    seed = cfg["seed"]
    device = cfg["device"]
    model_path = cfg["model_path"]
    models = os.listdir(model_path)
    logger.info(f"Found models: {models}")
    train_inputs, train_targets, eval_inputs, eval_targets, projection_matrix = (
        prepare_mnist(
            P * C,
            P_eval * C,
            N,
            binarize,
            seed,
            shuffle=True,
        )
    )
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    eval_inputs = eval_inputs.to(device)
    eval_targets = eval_targets.to(device)
    classifier = load_model(cfg["epoch"], model_path=cfg["model_path"])
    # computing overlaps
    states, unsats = relaxation_trajectory(
        classifier,
        train_inputs,
        train_targets,
        steps=cfg["steps"],
    )
    states = states[:, :, 1, :]
    overlap_stats = compute_overlap_evolution(states, cfg["steps"])
    logger.info("Computed train overlaps")
    table_overlap_evolution(overlap_stats, list(overlap_stats.keys()))
    # computing eval overlaps
    states, unsats = relaxation_trajectory(
        classifier,
        eval_inputs,
        eval_targets,
        steps=cfg["steps"],
    )
    states = states[:, :, 1, :]
    overlap_stats = compute_overlap_evolution(states, cfg["steps"])
    logger.info("Computed eval overlaps")
    table_overlap_evolution(overlap_stats, list(overlap_stats.keys()))


if __name__ == "__main__":
    main()
