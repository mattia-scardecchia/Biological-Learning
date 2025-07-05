import ast
import json
import os

import pandas as pd
import yaml

rows = []
base_dir = "experiments/xor"
ignore_pattern = os.path.join("ours")

for root, dirs, files in os.walk(base_dir):
    rel_path = os.path.relpath(root, base_dir)
    parts = rel_path.strip(os.sep).split(os.sep)
    if not parts or parts[0] not in {"mlp", "perceptron", "ours"}:
        continue
    algorithm = parts[0]

    # ─── handle JSON-based runs for mlp & perceptron ──────────────────────────
    if "eval_results.json" in files and "train_results.json" in files:
        if ignore_pattern in root:
            continue

        run_folder = parts[-1]
        cfg = run_folder[len("cfg:") :] if run_folder.startswith("cfg:") else run_folder
        kvs = cfg.split(",")

        row = {"algorithm": algorithm}
        for kv in kvs:
            if "=" in kv:
                key, val = kv.split("=", 1)
                try:
                    val = ast.literal_eval(val)
                except Exception:
                    pass
                row[key] = val

        # unified H
        if algorithm == "perceptron":
            row["H"] = None
            del row["model.hidden_dims"]
        else:
            assert algorithm == "mlp"
            assert len(row["model.hidden_dims"]) == 1, (
                "MLP is assumed to have a single hidden layer."
            )
            row["H"] = row["model.hidden_dims"][0]
            del row["model.hidden_dims"]

        # train_acc
        with open(os.path.join(root, "train_results.json")) as f:
            tdata = json.load(f)
            tdata = tdata[0] if isinstance(tdata, list) else tdata
            row["train_acc"] = tdata.get("train_acc", tdata.get("test_acc"))

        # test_acc
        with open(os.path.join(root, "eval_results.json")) as f:
            edata = json.load(f)
            edata = edata[0] if isinstance(edata, list) else edata
            row["test_acc"] = edata.get("test_acc")

        rows.append(row)
        continue

    # ─── handle grid_search CSV for ours ─────────────────────────────────────
    if algorithm == "ours" and "grid_search_results.csv" in files:
        csv_path = os.path.join(root, "grid_search_results.csv")
        df_csv = pd.read_csv(csv_path)
        cfg_path = os.path.join(root, ".hydra", "config.yaml")
        with open(cfg_path) as f:
            hydra_cfg = yaml.safe_load(f)
        H_val = hydra_cfg.get("H")
        for _, rec in df_csv.iterrows():
            row = {
                "algorithm": "ours",
                "seed": int(rec["seed"]),
                "bias_std": rec["bias_std"],
                "H": H_val,
                "train_acc": rec["final_train_acc"],
                "test_acc": rec["final_eval_acc"],
            }
            rows.append(row)

# dump to CSV
df = pd.DataFrame(rows)
df.to_csv(os.path.join(base_dir, "results.csv"), index=False)
print("Saved results to 'results.csv'")


# all columns except seed and the two metrics
group_cols = [c for c in df.columns if c not in {"seed", "train_acc", "test_acc"}]

# compute mean and std across seeds
summary = (
    df.groupby(group_cols, dropna=False)
    .agg(
        train_acc_mean=("train_acc", "mean"),
        train_acc_std=("train_acc", "std"),
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
    )
    .reset_index()
)

# write out the summary
summary.to_csv(os.path.join(base_dir, "summary_results.csv"), index=False)
print("Saved per‐config summary to 'summary_results.csv'")
