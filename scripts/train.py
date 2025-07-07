import copy
import cProfile
import json
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
    handle_input_input_overlaps,
    plot_accuracy_by_class_barplot,
    plot_accuracy_history,
    plot_couplings_distro_evolution,
    plot_couplings_histograms,
    plot_representations_similarity_among_layers,
)


def dump_stats(output_dir, logs):
    df = {
        "train_acc_history": logs["train_acc_history"],
        "eval_acc_history": logs["eval_acc_history"],
        "symmetricity_history": logs["symmetricity_history"],
    }
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(df, f, indent=4)


def log_representations(logs, save_dir, cfg):
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

        P = len(representations.keys())
        X = np.stack([representations[p] for p in range(P)], axis=2)
        input_input_overlaps = np.einsum("tlph,tlqh->tlpq", X, X) / X.shape[3]
        handle_input_input_overlaps(
            input_input_overlaps / 2 + 0.5, plot_dir, cfg.num_epochs
        )

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
    max_steps_val = max(cfg.max_steps_train, cfg.max_steps_eval)
    for max_steps in [0, max_steps_val]:
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
        ) = prepare_cifar(
            cfg.data.P * C,
            cfg.data.P_eval * C,
            cfg.N,
            cfg.data.cifar.binarize,
            cfg.seed,
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
    elif cfg.data.dataset == "marc":
        C = 2
        assert cfg.data.P + cfg.data.P_eval <= 20000 / C
        P = cfg.data.P * C
        P_eval = cfg.data.P_eval * C
        logging.warning("Dirty data loading!!!")
        data_dir = "experiments/xor-data"  # requires pulling git lfs
        inputs = np.load(
            os.path.join(data_dir, f"xi_N100_P{200 if P == 200 else 20000}.npy")
        )
        targets = np.load(
            os.path.join(data_dir, f"y_N100_P{200 if P == 200 else 20000}.npy")
        )
        # rng = np.random.default_rng(cfg.seed)
        # random_proj = rng.standard_normal((100, 100))
        # inputs = np.sign(inputs @ random_proj)
        perm = np.random.permutation(len(inputs))
        inputs = inputs[perm]
        targets = targets[perm]
        train_inputs = torch.tensor(inputs[0:P], dtype=torch.float32)
        train_targets = torch.tensor(targets[0:P], dtype=torch.float32)
        eval_inputs = torch.tensor(inputs[P : P + P_eval], dtype=torch.float32)
        eval_targets = torch.tensor(targets[P : P + P_eval], dtype=torch.float32)

        train_order = train_targets.argmax(dim=1).argsort()
        eval_order = eval_targets.argmax(dim=1).argsort()
        train_inputs = train_inputs[train_order]
        train_targets = train_targets[train_order]
        eval_inputs = eval_inputs[eval_order]
        eval_targets = eval_targets[eval_order]
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")
    return train_inputs, train_targets, eval_inputs, eval_targets, C


def parse_config(cfg):
    if "threshold" in cfg and "threshold_hidden" in cfg:
        for i in range(cfg.num_layers):
            assert cfg.threshold[i] == cfg.threshold_hidden
    if "threshold" in cfg and "threshold_readout" in cfg:
        assert cfg.threshold[-1] == cfg.threshold_readout
    if "lr" in cfg and "lr_J" in cfg:
        for i in range(cfg.num_layers):
            assert cfg.lr[i] == cfg.lr_J
    if "lr" in cfg and "lr_W" in cfg:
        assert (
            cfg.lr[-1] == cfg.lr_W
        )  # NOTE: we do not check lr for wback, since it will be set to 0 later
    if "weight_decay" in cfg and "weight_decay_J" in cfg:
        for i in range(cfg.num_layers):
            assert cfg.weight_decay[i] == cfg.weight_decay_J
    if "weight_decay" in cfg and "weight_decay_W" in cfg:
        assert cfg.weight_decay[-1] == cfg.weight_decay_W
    try:
        lr = cfg.lr
    except Exception:
        lr = [cfg.lr_J] * cfg.num_layers + [cfg.lr_W] * 2
    try:
        weight_decay = cfg.weight_decay
    except Exception:
        weight_decay = [cfg.weight_decay_J] * cfg.num_layers + [cfg.weight_decay_W] * 2
    try:
        threshold = copy.deepcopy(cfg.threshold)
    except Exception:
        threshold = [cfg.threshold_hidden] * cfg.num_layers + [cfg.threshold_readout]
    assert isinstance(cfg.J_D, (float, int))
    for i in range(cfg.num_layers):
        threshold[i] = threshold[i] + cfg.J_D
    try:
        lambda_left = cfg.lambda_left
    except Exception:
        lambda_left = (
            [cfg.lambda_x] + [cfg.lambda_l] * (cfg.num_layers - 1) + [cfg.lambda_wforth]
        )
    try:
        lambda_right = cfg.lambda_right
    except Exception:
        lambda_right = (
            [cfg.lambda_r] * (cfg.num_layers - 1) + [cfg.lambda_wback] + [cfg.lambda_y]
        )
    if lr[-2] != 0:
        logging.warning("Detected lr of wback != 0. Setting it to 0")
        lr[-2] = 0
    return lr, weight_decay, threshold, lambda_left, lambda_right


@hydra.main(
    config_path="../configs", config_name="baseline_1layer_smallP", version_base="1.3"
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
        "init_noise": cfg.init_noise,
        "symmetric_W": cfg.symmetric_W,
        "symmetric_J_init": cfg.symmetric_J_init,
        "double_dynamics": cfg.double_dynamics,
        "double_update": cfg.double_update,
        "use_local_ce": cfg.use_local_ce,
        "beta_ce": cfg.beta_ce,
        "p_update": cfg.p_update,
        "fc_left": cfg.fc_left,
        "fc_right": cfg.fc_right,
        "fc_input": cfg.fc_input,
        "lambda_fc": cfg.lambda_fc,
        "lambda_cylinder": cfg.lambda_cylinder,
        "lambda_wback_skip": cfg.lambda_wback_skip,
        "lambda_wforth_skip": cfg.lambda_wforth_skip,
        "lr_wforth_skip": cfg.lr_wforth_skip,
        "weight_decay_wforth_skip": cfg.weight_decay_wforth_skip,
        "lambda_input_skip": cfg.lambda_input_skip,
        "lambda_input_output_skip": cfg.lambda_input_output_skip,
        "lr_input_skip": cfg.lr_input_skip,
        "weight_decay_input_skip": cfg.weight_decay_input_skip,
        "lr_input_output_skip": cfg.lr_input_output_skip,
        "weight_decay_input_output_skip": cfg.weight_decay_input_output_skip,
        "zero_fc_init": cfg.zero_fc_init,
        "symmetrize_fc": cfg.symmetrize_fc,
        "symmetrize_internal_couplings": cfg.symmetrize_internal_couplings,
        "symmetric_threshold_internal_couplings": cfg.symmetric_threshold_internal_couplings,
        "symmetric_update_internal_couplings": cfg.symmetric_update_internal_couplings,
        "bias_std": cfg.bias_std,
        "H": cfg.H,
    }
    model_cls = BatchMeIfUCan
    model = model_cls(**model_kwargs)
    handler = Handler(
        model,
        cfg.init_mode,
        cfg.skip_representations,
        cfg.skip_couplings,
        output_dir,
        cfg.begin_curriculum,
        cfg.p_curriculum,
        skip_overlaps=cfg.skip_overlaps,
    )

    fields_plots_dir = os.path.join(output_dir, "fields")
    os.makedirs(fields_plots_dir, exist_ok=True)
    if not cfg.skip_fields:
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

    # initial checkpoint
    checkpoints_path = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    if cfg.save_model_and_data:
        chkpt = model.make_checkpoint(full=False)
        save_path = os.path.join(checkpoints_path, "init_model.npz")
        np.savez(save_path, **chkpt)
        data_dict = {
            "train_inputs": train_inputs.cpu().numpy(),
            "train_targets": train_targets.cpu().numpy(),
            "eval_inputs": eval_inputs.cpu().numpy(),
            "eval_targets": eval_targets.cpu().numpy(),
        }
        data_path = os.path.join(checkpoints_path, "data.npz")
        np.savez(data_path, **data_dict)

    # TODO: DELETE THIS
    # chkpt_path = "/Users/mat/Desktop/Files/Code/Biological-Learning/outputs/prova/2025-06-30-11-37-27/checkpoints/final_model.npz"
    # loaded = np.load(chkpt_path)
    # d_loaded = {key: loaded[key] for key in loaded}
    # model.load_checkpoint(d_loaded, full=False)
    # data_path = "/Users/mat/Desktop/Files/Code/Biological-Learning/outputs/prova/2025-06-30-11-37-27/checkpoints/data.npz"
    # loaded = np.load(data_path)
    # train_inputs = torch.tensor(loaded["train_inputs"], dtype=torch.float32).to(
    #     cfg.device
    # )
    # train_targets = torch.tensor(loaded["train_targets"], dtype=torch.float32).to(
    #     cfg.device
    # )
    # eval_inputs = torch.tensor(loaded["eval_inputs"], dtype=torch.float32).to(
    #     cfg.device
    # )
    # eval_targets = torch.tensor(loaded["eval_targets"], dtype=torch.float32).to(
    #     cfg.device
    # )

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
        cfg.max_steps_train,
        cfg.max_steps_eval,
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
    # Final checkpoint
    if cfg.save_model_and_data:
        chkpt = model.make_checkpoint(full=False)
        save_path = os.path.join(checkpoints_path, "final_model.npz")
        np.savez(save_path, **chkpt)

    # loaded = np.load("data.npz")
    # d_loaded = {key: loaded[key] for key in loaded}
    # for key in d_loaded:
    #     # check equality of loaded and original data
    #     assert np.array_equal(d_loaded[key], chkpt[key]), (
    #         f"Loaded {key} does not match original data"
    #     )

    # Field Breakdown
    if not cfg.skip_fields:
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
    eval_metrics = handler.evaluate(eval_inputs, eval_targets, cfg.max_steps_eval)
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
        log_representations(logs, representations_root_dir, cfg)

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

    dump_stats(output_dir, logs)


if __name__ == "__main__":
    # with torch.mps.profiler.profile(mode="interval", wait_until_completed=False):
    main()
