import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms


class SignActivation(nn.Module):
    def forward(self, x):
        return torch.sign(x)


def load_balanced_dataset(save_dir: str):
    """Load dataset as saved by dump_balanced_dataset. \\
    :param save_dir: directory where the dataset is saved. \\
    :return: inputs, targets, metadata.
    """
    with open(os.path.join(save_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    with open(os.path.join(save_dir, "inputs.npy"), "rb") as f:
        inputs = np.load(f)
    with open(os.path.join(save_dir, "targets.npy"), "rb") as f:
        targets = np.load(f)
    with open(os.path.join(save_dir, "class_prototypes.npy"), "rb") as f:
        class_prototypes = np.load(f)
    return inputs, targets, metadata, class_prototypes


def dump_balanced_dataset(inputs, targets, metadata, save_dir: str, class_prototypes):
    """Dump dataset as generated by generate_balanced_dataset."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    with open(os.path.join(save_dir, "inputs.npy"), "wb") as f:
        np.save(f, inputs)
    with open(os.path.join(save_dir, "targets.npy"), "wb") as f:
        np.save(f, targets)
    with open(os.path.join(save_dir, "class_prototypes.npy"), "wb") as f:
        np.save(f, class_prototypes)


def generate_balanced_dataset(
    N: int,
    P: int,
    C: int,
    p: float,
    rng: np.random.Generator,
    prototypes: Optional[np.ndarray],
):
    """Generates a balanced dataset with P*C patterns and C classes. Sample C N-dimensional
    binary (+/- 1) class prototypes and flip a fraction p of their bits P times to generate
    the patterns."""
    class_prototypes = (
        prototypes.copy()
        if prototypes is not None
        else rng.choice([-1, 1], size=(C, N))
    )
    inputs = np.zeros((P * C, N), dtype=np.int8)
    for c in range(C):
        inputs[c * P : (c + 1) * P] = class_prototypes[c]
        flips = 2 * rng.binomial(1, 1 - p, size=(P, N)) - 1
        inputs[c * P : (c + 1) * P] *= flips
    targets = np.repeat(np.eye(C), P, axis=0)
    metadata = {"N": N, "P": P, "C": C, "p": p}
    return inputs, targets, metadata, class_prototypes


def get_balanced_dataset(
    N: int,
    P: int,
    C: int,
    p: float,
    save_dir: str,
    prototypes: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    shuffle: bool = True,
    load_if_available: bool = True,
    dump: bool = True,
):
    """Generates a balanced dataset with P*C patterns and C classes. Sample C N-dimensional
    binary (+/- 1) class prototypes and flip a fraction p of their bits P times to generate
    the patterns. \\
    Dump the class prototypes, the generated pairs (pattern, class), and a metadata file which
    contains the dataset parameters. \\
    :param N: input dimensionality. \\
    :param P: number of patterns per class. \\
    :param C: number of classes. \\
    :param p: probability of flipping a bit from class prototypes. \\
    :param prototypes: if not None, use these prototypes instead of generating new ones. \\
    :param save_dir: directory to save the dataset. \\
    :param load_if_available: if True, load the dataset from save_dir if metadata (and prototypes) match. \\
    :param dump: if True, dump the dataset to save_dir.
    """
    if load_if_available and os.path.exists(save_dir):
        inputs, targets, metadata, class_prototypes = load_balanced_dataset(save_dir)
        if (
            metadata["N"] == N
            and metadata["P"] == P
            and metadata["C"] == C
            and metadata["p"] == p
            and (prototypes is None or np.array_equal(class_prototypes, prototypes))
        ):
            return inputs, targets, metadata, class_prototypes

    rng = np.random.default_rng() if rng is None else rng
    inputs, targets, metadata, class_prototypes = generate_balanced_dataset(
        N, P, C, p, rng, prototypes
    )
    if shuffle:
        indices = rng.permutation(P * C)
        inputs = inputs[indices]
        targets = targets[indices]
    if dump:
        dump_balanced_dataset(inputs, targets, metadata, save_dir, class_prototypes)
    return inputs, targets, metadata, class_prototypes


def prepare_mnist(num_samples_train, num_samples_eval, N, binarize, seed, shuffle=True):
    # load MNIST dataset
    torch.manual_seed(seed)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    eval_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Downsample the datasets
    train_perm = (
        torch.randperm(len(train_dataset))
        if shuffle
        else torch.arange(len(train_dataset))
    )
    train_indices = train_perm[:num_samples_train]
    train_images = torch.stack([train_dataset[i][0] for i in train_indices])
    train_labels = torch.tensor([train_dataset[i][1] for i in train_indices])
    eval_perm = (
        torch.randperm(len(eval_dataset))
        if shuffle
        else torch.arange(len(eval_dataset))
    )
    eval_indices = eval_perm[:num_samples_eval]
    eval_images = torch.stack([eval_dataset[i][0] for i in eval_indices])
    eval_labels = torch.tensor([eval_dataset[i][1] for i in eval_indices])

    # Sort samples by class
    sort_idx = torch.argsort(eval_labels)
    eval_labels = eval_labels[sort_idx]
    eval_images = eval_images[sort_idx]
    sort_idx = torch.argsort(train_labels)
    train_labels = train_labels[sort_idx]
    train_images = train_images[sort_idx]

    # Convert labels to one-hot encoding
    train_labels = torch.eye(10)[train_labels]
    eval_labels = torch.eye(10)[eval_labels]

    # Random linear projection and binarization
    projection_matrix = torch.randn(784, N)
    train_images = train_images @ projection_matrix
    eval_images = eval_images @ projection_matrix
    if binarize:
        train_images = torch.sign(train_images)
        eval_images = torch.sign(eval_images)
    else:
        train_images = (train_images - train_images.min()) / (
            train_images.max() - train_images.min()
        )
        eval_images = (eval_images - eval_images.min()) / (
            eval_images.max() - eval_images.min()
        )

    return train_images, train_labels, eval_images, eval_labels, projection_matrix


def prepare_cifar(
    num_samples_train, num_samples_eval, N, binarize, seed, cifar10=True, shuffle=True
):
    torch.manual_seed(seed)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    # Load CIFAR dataset
    if cifar10:
        dataset_cls = datasets.CIFAR10
        num_classes = 10
    else:
        dataset_cls = datasets.CIFAR100
        num_classes = 100
    train_dataset = dataset_cls(
        root="./data", train=True, download=True, transform=transform
    )
    eval_dataset = dataset_cls(
        root="./data", train=False, download=True, transform=transform
    )

    # Downsample datasets
    train_perm = (
        torch.randperm(len(train_dataset))
        if shuffle
        else torch.arange(len(train_dataset))
    )
    train_indices = train_perm[:num_samples_train]
    train_images = torch.stack([train_dataset[i][0] for i in train_indices])
    train_labels = torch.tensor([train_dataset[i][1] for i in train_indices])
    eval_perm = (
        torch.randperm(len(eval_dataset))
        if shuffle
        else torch.arange(len(eval_dataset))
    )
    eval_indices = eval_perm[:num_samples_eval]
    eval_images = torch.stack([eval_dataset[i][0] for i in eval_indices])
    eval_labels = torch.tensor([eval_dataset[i][1] for i in eval_indices])

    # Sort samples by class for better evaluation insights
    sort_idx = torch.argsort(eval_labels)
    eval_labels = eval_labels[sort_idx]
    eval_images = eval_images[sort_idx]
    sort_idx = torch.argsort(train_labels)
    train_labels = train_labels[sort_idx]
    train_images = train_images[sort_idx]

    # Convert labels to one-hot encoding
    train_labels = torch.eye(num_classes)[train_labels]
    eval_labels = torch.eye(num_classes)[eval_labels]

    # Random linear projection, and Binarize/Normalize the data
    projection_matrix = torch.randn(3072, N)
    # projection_matrix = torch.eye(3072, 3072)
    train_images = train_images @ projection_matrix
    eval_images = eval_images @ projection_matrix
    if binarize:
        median = train_images.median().item()
        train_images = torch.sign(train_images - median)
        eval_images = torch.sign(eval_images - median)
    else:
        train_images = (train_images - train_images.min()) / (
            train_images.max() - train_images.min()
        )
        eval_images = (eval_images - eval_images.min()) / (
            eval_images.max() - eval_images.min()
        )
        median = None

    return (
        train_images,
        train_labels,
        eval_images,
        eval_labels,
        projection_matrix,
        median,
    )


@torch.inference_mode()
def prepare_hm_data(
    D,
    C,
    P,
    P_eval,
    M,
    L,
    width,
    hidden_activation,
    seed,
    binarize,
):
    """
    Prepares synthetic data using two teacher networks:
      - teacher_linear: a linear model (D -> C) for label generation.
      - teacher_mlp: an MLP with L hidden layers and final output dimension M.

    Parameters:
        D (int): Latent dimension.
        C (int): Number of classes.
        P (int): Number of patterns per class (total samples = P * C).
        M (int): Output dimension of the teacher MLP.
        L (int): Number of hidden layers in teacher MLP.
        width (int or list[int]): Hidden layer width(s). If int, used for all layers.
        activation (nn.Module or list[nn.Module]): Activation function(s) for hidden layers.
        seed (int): Random seed.

    Returns:
        latent_inputs (torch.Tensor): Latent inputs (P * C, D), sorted by predicted class.
        projected_inputs (torch.Tensor): Teacher MLP outputs, shape (P * C, M).
        labels (torch.Tensor): One-hot labels, shape (P * C, C).
        teacher_linear (nn.Module): The linear teacher network.
        teacher_mlp (nn.Module): The MLP teacher network.
    """
    torch.manual_seed(seed)

    # Instantiate the teacher_linear model.
    teacher_linear = nn.Linear(D, C)

    # Build the teacher_mlp.
    if isinstance(width, int):
        widths = [width] * L
    else:
        assert len(width) == L, "Length of width list must equal L."
        widths = width

    if not isinstance(hidden_activation, list):
        activations = [hidden_activation] * L
    else:
        assert len(hidden_activation) == L, "Length of activation list must equal L."
        activations = hidden_activation

    mlp_layers = []
    in_features = D
    for i in range(L):
        mlp_layers.append(nn.Linear(in_features, widths[i]))
        mlp_layers.append(nn.LayerNorm(widths[i]))
        mlp_layers.append(activations[i])
        in_features = widths[i]
    mlp_layers.append(nn.Linear(in_features, M))
    if binarize:
        mlp_layers.append(SignActivation())
    teacher_mlp = nn.Sequential(*mlp_layers)

    total_samples = (P + P_eval) * C
    latent_inputs = torch.randn(total_samples, D)
    logits = teacher_linear(latent_inputs)
    preds = torch.argmax(logits, dim=1)
    projected_inputs = teacher_mlp(latent_inputs)

    # split the data into training and evaluation sets
    latent_inputs_train = latent_inputs[: P * C]
    logits_train = logits[: P * C]
    preds_train = preds[: P * C]
    projected_inputs_train = projected_inputs[: P * C]
    latent_inputs_eval = latent_inputs[P * C :]
    logits_eval = logits[P * C :]
    preds_eval = preds[P * C :]
    projected_inputs_eval = projected_inputs[P * C :]

    # Sort samples by predicted class.
    sort_idx = torch.argsort(preds_train)
    train_inputs = projected_inputs_train[sort_idx]
    train_preds = preds_train[sort_idx]
    train_labels = torch.eye(C)[train_preds]
    sort_idx = torch.argsort(preds_eval)
    eval_inputs = projected_inputs_eval[sort_idx]
    eval_preds = preds_eval[sort_idx]
    eval_labels = torch.eye(C)[eval_preds]

    return (
        train_inputs,
        train_labels,
        eval_inputs,
        eval_labels,
        teacher_linear,
        teacher_mlp,
    )
