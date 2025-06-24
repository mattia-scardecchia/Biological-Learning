import torch
import torch
import numpy as np
import pickle
import os
import logging
from itertools import combinations
from typing import Dict
import argparse
import json

logger = logging.getLogger(__name__)


def load_model(epoch: int, model_path: str):
    logger.info(f"Loading model from epoch {epoch} from {model_path}")
    model_file = os.path.join(model_path, f"model_epoch_{epoch}.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} does not exist.")
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    return model


def compute_overlap_evolution(states, steps) -> Dict[str, torch.Tensor]:
    # data, time, state
    overlaps_stats = {}
    for time1, time2 in combinations(range(len(steps)), 2):
        state_1 = states[:, time1, :]
        state_2 = states[:, time2, :]
        overlaps = (state_1 * state_2).sum(dim=-1) / state_1.shape[-1]
        overlaps_mean = overlaps.mean(dim=0).item()
        overlaps_error = overlaps.std(dim=0).item() / (overlaps.shape[0] ** 0.5)
        overlaps_std = overlaps.std(dim=0).item()
        overlaps_stats[f"{steps[time1]}-{steps[time2]}"] = {
            "mean": overlaps_mean,
            "error": overlaps_error,
            "std": overlaps_std,
        }
    logger.info(f"Computed overlaps for {list(overlaps_stats.keys())}")
    return overlaps_stats


def compute_convergence_evolution(statistics) -> Dict[str, torch.Tensor]:
    average_overlap = statistics["overlaps"].mean(dim=-1)
    std_overlap = statistics["overlaps"].std(dim=-1)
    min_overlap = statistics["overlaps"].min(dim=-1).values
    return {
        "average_overlap": average_overlap,
        "std_overlap": std_overlap,
        "min_overlap": min_overlap,
    }


def compute_statistics(statistics):
    overlap_stats = compute_overlap_evolution(statistics["states"], statistics["steps"])
    convergence_stats = compute_convergence_evolution(statistics)
    return {
        "overlaps": overlap_stats,
        "convergence": convergence_stats,
        "steps": statistics["steps"],
        "epoch": statistics["epoch"],
    }


def main(**args):
    experiment_path = args["model_path"]
    files = os.listdir(experiment_path)
    train_files = [f for f in files if "train" in f and f.endswith(".pkl")]
    eval_files = [f for f in files if "eval" in f and f.endswith(".pkl")]
    train_stats, eval_stats = [], []
    for train_file in train_files:
        logger.info(f"Processing training file: {train_file}")
        with open(os.path.join(experiment_path, train_file), "rb") as f:
            statistics = pickle.load(f)
        statistics = compute_statistics(statistics)
        train_stats.append(statistics)
    for eval_file in eval_files:
        logger.info(f"Processing training file: {eval_file}")
        with open(os.path.join(experiment_path, eval_file), "rb") as f:
            statistics = pickle.load(f)
        statistics = compute_statistics(statistics)
        eval_stats.append(statistics)
    logger.info(
        f"Computed statistics for {len(train_stats)} training and {len(eval_stats)} evaluation files."
    )
    with open(os.path.join(experiment_path, "train_statistics.json"), "w") as f:
        json.dump(train_stats, f, indent=2)
    with open(os.path.join(experiment_path, "eval_statistics.json"), "w") as f:
        json.dump(eval_stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute overlap statistics from trained model."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model directory."
    )
    args = parser.parse_args()
    main(model_path=args.model_path)
