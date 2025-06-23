#!/usr/bin/env python3
import csv
import os

# Path to the root directory containing all cfg:*/ folders
ROOT_DIR = "/Users/mat/Desktop/Files/Code/Biological-Learning/multirun/prova/2025-06-23-10-11-44"


def parse_folder_name(folder_name):
    """
    Given a folder name like
    'cfg:J_D=0.0,double_dynamics=False,lambda_input_skip=[2.0],seed=13,symmetric_J_init=False'
    return a dict of { key: value, ... }.
    """
    assert folder_name.startswith("cfg:")
    cfg_str = folder_name[len("cfg:") :]
    pairs = cfg_str.split(",")
    cfg = {}
    for pair in pairs:
        key, val = pair.split("=", 1)
        cfg[key] = val
    return cfg


def parse_log_file(log_path):
    """
    Read train.log and extract:
      - max_train_acc    from 'Best train accuracy: {value}'
      - max_eval_acc     from 'Best eval accuracy: {value}'
      - final_train_acc  from the last 'Train Acc: {value}%'
      - final_eval_acc   from the last 'Eval Acc:  {value}%'
    """
    max_train_acc = None
    max_eval_acc = None
    final_train_acc = None
    final_eval_acc = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if "Best train accuracy:" in line:
                max_train_acc = line.split("Best train accuracy:")[1].strip()
            elif "Best eval accuracy:" in line:
                max_eval_acc = line.split("Best eval accuracy:")[1].strip()
            elif line.startswith("Train Acc:"):
                # e.g. "Train Acc: 82.7%"
                value = line.split("Train Acc:", 1)[1].strip().rstrip("%")
                final_train_acc = value
            elif line.startswith("Eval Acc:"):
                # e.g. "Eval Acc:  77.1%"
                value = line.split("Eval Acc:", 1)[1].strip().rstrip("%")
                final_eval_acc = value

    return {
        "max_train_acc": max_train_acc,
        "max_eval_acc": max_eval_acc,
        "final_train_acc": final_train_acc,
        "final_eval_acc": final_eval_acc,
    }


def main():
    rows = []
    # Walk only top-level entries in ROOT_DIR
    for entry in os.listdir(ROOT_DIR):
        folder_path = os.path.join(ROOT_DIR, entry)
        if not os.path.isdir(folder_path):
            continue
        try:
            config = parse_folder_name(entry)
        except AssertionError:
            # skip folders not starting with 'cfg:'
            continue

        log_path = os.path.join(folder_path, "train.log")
        if not os.path.isfile(log_path):
            print(f"Warning: no train.log in {folder_path}, skipping.")
            continue

        metrics = parse_log_file(log_path)
        row = {**config, **metrics}
        rows.append(row)

    if not rows:
        print("No valid rows found. Exiting.")
        return

    # Determine CSV columns: all config keys sorted, then the metric columns
    config_keys = sorted(
        k
        for k in rows[0].keys()
        if k
        not in {"max_train_acc", "max_eval_acc", "final_train_acc", "final_eval_acc"}
    )
    metric_keys = ["max_train_acc", "max_eval_acc", "final_train_acc", "final_eval_acc"]
    fieldnames = config_keys + metric_keys

    # Write summary CSV
    out_path = os.path.join(ROOT_DIR, "summary.csv")
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()
