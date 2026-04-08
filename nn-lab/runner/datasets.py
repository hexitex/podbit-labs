"""
Fixed-menu dataset loaders.

Standard datasets from torchvision + synthetic generators.
Datasets are cached in a data/ directory.
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_dataset(spec: dict, batch_size: int = 64) -> tuple:
    """
    Load dataset from spec.

    Returns (train_loader, val_loader, test_loader, info)
    where info = {"num_classes": int, "input_shape": tuple, "is_regression": bool}
    """
    ds = spec.get("training", {}).get("dataset", "mnist")
    ds_name = ds if isinstance(ds, str) else ds.get("name", "mnist")
    ds_params = ds if isinstance(ds, dict) else {}

    if ds_name == "mnist":
        return _load_torchvision("MNIST", batch_size)
    elif ds_name == "fashion_mnist":
        return _load_torchvision("FashionMNIST", batch_size)
    elif ds_name == "cifar10":
        return _load_torchvision("CIFAR10", batch_size)
    elif ds_name == "synthetic_quadratic":
        return _load_synthetic_quadratic(ds_params, batch_size)
    elif ds_name == "synthetic_regression":
        return _load_synthetic_regression(ds_params, batch_size)
    else:
        raise ValueError(f"not supported: dataset '{ds_name}'")


def _load_torchvision(name: str, batch_size: int) -> tuple:
    """Load a torchvision dataset. Downloads on first use."""
    import torchvision
    import torchvision.transforms as T

    _ensure_data_dir()

    if name == "CIFAR10":
        transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        input_shape = (3, 32, 32)
    else:
        transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        input_shape = (1, 28, 28)

    dataset_cls = getattr(torchvision.datasets, name)
    train_full = dataset_cls(DATA_DIR, train=True, download=True, transform=transform)
    test_set = dataset_cls(DATA_DIR, train=False, download=True, transform=transform)

    # Split train into train/val (90/10)
    n_val = len(train_full) // 10
    n_train = len(train_full) - n_val
    train_set, val_set = torch.utils.data.random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),  # deterministic split
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, {
        "num_classes": 10,
        "input_shape": input_shape,
        "is_regression": False,
    }


def _load_synthetic_quadratic(params: dict, batch_size: int) -> tuple:
    """
    Synthetic quadratic loss surface: f(x) = x^T A x
    where A has specified curvature spectrum.
    """
    dim = params.get("dim", 10)
    curvature_spec = params.get("curvature_spec", None)
    n_samples = params.get("n_samples", 2000)

    rng = np.random.RandomState(42)

    # Build curvature matrix
    if curvature_spec and isinstance(curvature_spec, list):
        eigenvalues = np.array(curvature_spec[:dim])
        if len(eigenvalues) < dim:
            eigenvalues = np.concatenate([eigenvalues, np.ones(dim - len(eigenvalues))])
    else:
        # Default: condition number ~100
        eigenvalues = np.logspace(-2, 0, dim)

    Q = np.linalg.qr(rng.randn(dim, dim))[0]
    A = Q @ np.diag(eigenvalues) @ Q.T

    # Generate data: X random, y = x^T A x
    X = rng.randn(n_samples, dim).astype(np.float32)
    y = np.array([x @ A @ x for x in X], dtype=np.float32)

    X_t = torch.from_numpy(X)
    # Targets must be 1-D [batch] — NOT [batch, 1]. LLM-generated models use
    # .squeeze(-1) on the output, producing [batch]. If targets are [batch, 1],
    # MSELoss silently broadcasts [batch] vs [batch, 1] into a [batch, batch]
    # matrix, computing garbage loss and preventing convergence.
    y_t = torch.from_numpy(y)  # already 1-D from the list comprehension

    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)

    train_ds = TensorDataset(X_t[:n_train], y_t[:n_train])
    val_ds = TensorDataset(X_t[n_train:n_train + n_val], y_t[n_train:n_train + n_val])
    test_ds = TensorDataset(X_t[n_train + n_val:], y_t[n_train + n_val:])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        {"num_classes": 1, "input_shape": (dim,), "is_regression": True},
    )


def _load_synthetic_regression(params: dict, batch_size: int) -> tuple:
    """Synthetic linear regression: y = Wx + b + noise."""
    dim = params.get("dim", 10)
    noise = params.get("noise", 0.1)
    n_samples = params.get("n_samples", 2000)

    rng = np.random.RandomState(42)
    W_true = rng.randn(dim).astype(np.float32)  # 1-D weights → 1-D targets
    b_true = float(rng.randn(1))

    X = rng.randn(n_samples, dim).astype(np.float32)
    y = (X @ W_true + b_true + noise * rng.randn(n_samples).astype(np.float32))

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)  # shape [n_samples] — matches model output after .squeeze(-1)

    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)

    train_ds = TensorDataset(X_t[:n_train], y_t[:n_train])
    val_ds = TensorDataset(X_t[n_train:n_train + n_val], y_t[n_train:n_train + n_val])
    test_ds = TensorDataset(X_t[n_train + n_val:], y_t[n_train + n_val:])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        {"num_classes": 1, "input_shape": (dim,), "is_regression": True},
    )
