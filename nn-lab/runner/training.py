"""
Training loop — configures optimizer, scheduler, runs epochs, collects measurements.
"""

import time
import torch
import torch.nn as nn
from runner.measurements import MeasurementCollector
from runner.sparsity import apply_sparsity


def build_optimizer(model: nn.Module, spec: dict) -> torch.optim.Optimizer:
    """Build optimizer from spec."""
    opt_spec = spec.get("training", {}).get("optimizer", "adam")
    if isinstance(opt_spec, str):
        opt_name = opt_spec
        opt_params = {}
    else:
        opt_name = opt_spec.get("name", "adam")
        opt_params = {k: v for k, v in opt_spec.items() if k != "name"}

    lr = opt_params.get("lr", opt_params.get("learning_rate", 1e-3))

    # Check for weight_decay in modifications
    mods = spec.get("architecture", {}).get("modifications", [])
    wd = 0.0
    for mod in (mods or []):
        if isinstance(mod, dict) and mod.get("name") == "weight_decay":
            wd = mod.get("lambda", 1e-4)
        elif mod == "weight_decay":
            wd = 1e-4
    wd = opt_params.get("weight_decay", wd)

    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=opt_params.get("momentum", 0.9),
            weight_decay=wd,
        )
    elif opt_name == "adam":
        betas = (opt_params.get("beta1", 0.9), opt_params.get("beta2", 0.999))
        return torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas,
            weight_decay=wd,
        )
    elif opt_name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(), lr=lr,
            alpha=opt_params.get("alpha", 0.99),
            momentum=opt_params.get("momentum", 0.0),
            weight_decay=wd,
        )
    elif opt_name == "lbfgs":
        return torch.optim.LBFGS(
            model.parameters(), lr=lr,
            max_iter=opt_params.get("max_iter", 20),
        )
    else:
        raise ValueError(f"not supported: optimizer '{opt_name}'")


def build_scheduler(optimizer: torch.optim.Optimizer, spec: dict, epochs: int):
    """Build LR scheduler from spec. Returns None for constant."""
    sched_spec = spec.get("training", {}).get("lr_schedule", "constant")
    if isinstance(sched_spec, str):
        sched_name = sched_spec
        sched_params = {}
    else:
        sched_name = sched_spec.get("name", "constant")
        sched_params = {k: v for k, v in sched_spec.items() if k != "name"}

    if sched_name == "constant":
        return None

    elif sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs,
            eta_min=sched_params.get("eta_min", 0),
        )

    elif sched_name == "cyclical":
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=sched_params.get("min", 1e-4),
            max_lr=sched_params.get("max", 1e-2),
            step_size_up=sched_params.get("period", 10) // 2,
            mode='triangular2',
        )

    elif sched_name == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sched_params.get("milestones", [epochs // 2]),
            gamma=sched_params.get("gamma", 0.1),
        )

    else:
        raise ValueError(f"not supported: lr_schedule '{sched_name}'")


def build_loss_fn(is_regression: bool) -> nn.Module:
    """Build loss function."""
    if is_regression:
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


def train_single_run(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    spec: dict,
    device: torch.device,
    seed: int,
    measurements: list,
    is_regression: bool,
) -> dict:
    """
    Train a single run with a specific seed.

    Returns dict with seed and all requested measurements.
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    model = model.to(device)
    epochs = spec.get("training", {}).get("epochs", 10)
    loss_fn = build_loss_fn(is_regression)
    optimizer = build_optimizer(model, spec)
    scheduler = build_scheduler(optimizer, spec, epochs)

    collector = MeasurementCollector(measurements, device)
    collector.start_time = time.time()

    # Early stopping
    early = spec.get("training", {}).get("early_stopping", None)
    early_metric = early.get("metric", "val_loss") if early else None
    early_target = early.get("target", None) if early else None

    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        if is_lbfgs:
            # LBFGS needs closure
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                def closure():
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    if is_regression:
                        outputs = outputs.squeeze(-1)
                        targets_f = targets.float().squeeze(-1) if targets.dim() > 1 else targets.float()
                        loss = loss_fn(outputs, targets_f)
                    else:
                        loss = loss_fn(outputs, targets)
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                train_loss_sum += loss.item() * inputs.size(0)
                train_count += inputs.size(0)
        else:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if is_regression:
                    outputs = outputs.squeeze(-1)
                    targets = targets.float().squeeze(-1) if targets.dim() > 1 else targets.float()
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * inputs.size(0)
                train_count += inputs.size(0)

        train_loss = train_loss_sum / max(train_count, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                if is_regression:
                    outputs = outputs.squeeze(-1)
                    targets = targets.float().squeeze(-1) if targets.dim() > 1 else targets.float()
                loss = loss_fn(outputs, targets)
                val_loss_sum += loss.item() * inputs.size(0)
                val_count += inputs.size(0)

        val_loss = val_loss_sum / max(val_count, 1)

        # Collect epoch measurements
        collector.on_epoch_end(epoch, train_loss, val_loss, model, optimizer)

        if scheduler is not None:
            scheduler.step()

        # Early stopping check
        if early_target is not None:
            if early_metric == "val_loss" and val_loss <= early_target:
                break
            elif early_metric == "final_accuracy":
                # Quick accuracy check
                from runner.measurements import _compute_accuracy
                acc = _compute_accuracy(model, val_loader, device)
                if acc >= early_target:
                    break

    # Apply sparsity if requested (post-training)
    arch = spec.get("architecture", {})
    if arch.get("sparsity"):
        apply_sparsity(model, arch["sparsity"]["method"], arch["sparsity"]["ratio"])

    # Collect final measurements
    sparsity_threshold = 1e-6
    for m in measurements:
        if isinstance(m, dict) and m.get("name") == "sparsity_by_layer":
            sparsity_threshold = m.get("threshold", 1e-6)

    result = collector.finalize(model, loss_fn, test_loader, optimizer, sparsity_threshold)
    result["seed"] = seed

    return result
