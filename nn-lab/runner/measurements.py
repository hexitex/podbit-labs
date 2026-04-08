"""
Measurement collectors — gather data during and after training.

Each measurement is a function that takes the current state and returns data.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from runner.sparsity import measure_sparsity_by_layer


def collect_gradient_stats(model: nn.Module) -> list[dict]:
    """Collect gradient statistics per layer."""
    stats = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.data
            stats.append({
                "layer": name,
                "mean": g.mean().item(),
                "variance": g.var().item(),
                "max": g.abs().max().item(),
                "norm": g.norm().item(),
            })
    return stats


def collect_weight_norms(model: nn.Module) -> list[dict]:
    """Collect L2 norm of weights per layer."""
    norms = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            norms.append({
                "layer": name,
                "l2_norm": module.weight.data.norm().item(),
            })
    return norms


def collect_effective_lr(optimizer: torch.optim.Optimizer) -> list[dict]:
    """
    For adaptive optimizers (Adam, RMSprop), compute effective learning rate per param group.
    effective_lr ≈ lr / (sqrt(v) + eps) for Adam.
    """
    results = []
    for i, group in enumerate(optimizer.param_groups):
        lr = group['lr']
        # For Adam-type optimizers, approximate effective LR from state
        state_stats = []
        for p in group['params']:
            if p in optimizer.state:
                s = optimizer.state[p]
                if 'exp_avg_sq' in s:
                    v = s['exp_avg_sq']
                    eps = group.get('eps', 1e-8)
                    eff = lr / (v.sqrt() + eps)
                    state_stats.append({
                        "mean_effective_lr": eff.mean().item(),
                        "min_effective_lr": eff.min().item(),
                        "max_effective_lr": eff.max().item(),
                    })

        if state_stats:
            results.append({
                "param_group": i,
                "base_lr": lr,
                "effective_lr_mean": np.mean([s["mean_effective_lr"] for s in state_stats]),
                "effective_lr_min": np.min([s["min_effective_lr"] for s in state_stats]),
                "effective_lr_max": np.max([s["max_effective_lr"] for s in state_stats]),
            })
        else:
            results.append({"param_group": i, "base_lr": lr, "effective_lr_mean": lr})

    return results


def compute_hessian_top_eigenvalue(model: nn.Module, loss_fn, data_loader, device: torch.device,
                                    num_batches: int = 5) -> float:
    """
    Estimate top eigenvalue of the loss Hessian via power iteration.
    Expensive — only use on small models.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)

    # Safety check — Hessian-vector products are O(params) per iteration
    if total_params > 500_000:
        return float('nan')  # too large

    # Power iteration for top eigenvalue
    v = [torch.randn_like(p) for p in params]
    v_norm = torch.sqrt(sum((vi ** 2).sum() for vi in v))
    v = [vi / v_norm for vi in v]

    for _ in range(20):  # iterations
        # Compute Hv via finite differences
        Hv = _hessian_vector_product(model, loss_fn, data_loader, device, params, v, num_batches)

        # eigenvalue estimate = v^T Hv
        eigenvalue = sum((vi * hvi).sum() for vi, hvi in zip(v, Hv)).item()

        # normalize
        v_norm = torch.sqrt(sum((hvi ** 2).sum() for hvi in Hv))
        if v_norm.item() < 1e-12:
            break
        v = [hvi / v_norm for hvi in Hv]

    model.train()
    return eigenvalue


def _hessian_vector_product(model, loss_fn, data_loader, device, params, v, num_batches):
    """Compute Hessian-vector product via backprop."""
    model.zero_grad()

    total_loss = 0.0
    count = 0
    for i, (inputs, targets) in enumerate(data_loader):
        if i >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        if outputs.shape[-1] == 1:
            outputs = outputs.squeeze(-1)
            targets = targets.float().squeeze(-1) if targets.dim() > 1 else targets.float()
        loss = loss_fn(outputs, targets)
        total_loss += loss
        count += 1

    if count == 0:
        return [torch.zeros_like(p) for p in params]

    avg_loss = total_loss / count
    grads = torch.autograd.grad(avg_loss, params, create_graph=True)

    # grad dot v
    grad_v = sum((g * vi).sum() for g, vi in zip(grads, v))

    # Second derivative
    Hv = torch.autograd.grad(grad_v, params, retain_graph=False)
    return [hv.detach() for hv in Hv]


class MeasurementCollector:
    """Collects requested measurements during training."""

    def __init__(self, requested: list, device: torch.device):
        self.requested = set()
        self.intervals = {}
        for m in requested:
            if isinstance(m, str):
                self.requested.add(m)
            else:
                name = m.get("name", m)
                self.requested.add(name)
                if "interval" in m:
                    self.intervals[name] = m["interval"]
                if "threshold" in m:
                    self.intervals[f"{name}_threshold"] = m["threshold"]
                if "checkpoints" in m:
                    self.intervals[f"{name}_checkpoints"] = m["checkpoints"]

        self.device = device
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.gradient_snapshots: list[dict] = []
        self.weight_norm_snapshots: list[dict] = []
        self.effective_lr_snapshots: list[dict] = []
        self.start_time: float = 0.0

    def _should_collect_at(self, measurement: str, epoch: int) -> bool:
        interval = self.intervals.get(measurement, 1)
        return epoch % interval == 0

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float,
                     model: nn.Module, optimizer: torch.optim.Optimizer):
        """Called at the end of each epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if "gradient_stats_by_layer" in self.requested and self._should_collect_at("gradient_stats_by_layer", epoch):
            self.gradient_snapshots.append({
                "epoch": epoch,
                "stats": collect_gradient_stats(model),
            })

        if "weight_norm_by_layer" in self.requested and self._should_collect_at("weight_norm_by_layer", epoch):
            self.weight_norm_snapshots.append({
                "epoch": epoch,
                "norms": collect_weight_norms(model),
            })

        if "effective_lr_by_layer" in self.requested and self._should_collect_at("effective_lr_by_layer", epoch):
            self.effective_lr_snapshots.append({
                "epoch": epoch,
                "lr": collect_effective_lr(optimizer),
            })

    def finalize(self, model: nn.Module, loss_fn, test_loader, optimizer,
                 sparsity_threshold: float = 1e-6) -> dict:
        """Collect final measurements after training."""
        result: dict = {}

        if "loss_curve" in self.requested:
            result["loss_curve"] = {
                "train": self.train_losses,
                "val": self.val_losses,
            }

        if "convergence_epoch" in self.requested:
            threshold = self.intervals.get("convergence_epoch_threshold", None)
            if threshold is not None:
                conv_epoch = None
                for i, vl in enumerate(self.val_losses):
                    if vl < threshold:
                        conv_epoch = i + 1
                        break
                result["convergence_epoch"] = conv_epoch
            else:
                # Default: epoch where val loss stops improving significantly
                result["convergence_epoch"] = _detect_convergence(self.val_losses)

        if "final_accuracy" in self.requested:
            result["final_accuracy"] = _compute_accuracy(model, test_loader, self.device)

        if "sparsity_by_layer" in self.requested:
            result["sparsity_by_layer"] = measure_sparsity_by_layer(model, sparsity_threshold)

        if "gradient_stats_by_layer" in self.requested:
            result["gradient_stats_by_layer"] = self.gradient_snapshots

        if "effective_lr_by_layer" in self.requested:
            result["effective_lr_by_layer"] = self.effective_lr_snapshots

        if "weight_norm_by_layer" in self.requested:
            result["weight_norm_by_layer"] = self.weight_norm_snapshots

        if "hessian_top_eigenvalue" in self.requested:
            checkpoints = self.intervals.get("hessian_top_eigenvalue_checkpoints", ["final"])
            if "final" in checkpoints:
                result["hessian_top_eigenvalue"] = compute_hessian_top_eigenvalue(
                    model, loss_fn, test_loader, self.device
                )

        if "wall_clock_time" in self.requested:
            result["wall_clock_time_ms"] = int((time.time() - self.start_time) * 1000)

        return result


def _detect_convergence(val_losses: list[float], patience: int = 5) -> int | None:
    """Detect convergence epoch — where val loss stops improving for `patience` epochs."""
    if not val_losses:
        return None
    best = float('inf')
    wait = 0
    for i, loss in enumerate(val_losses):
        if loss < best - 1e-6:
            best = loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                return i + 1 - patience
    return None


def _compute_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if outputs.shape[-1] > 1:
                preds = outputs.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
            else:
                # Regression — accuracy not meaningful
                return float('nan')
    model.train()
    return correct / total if total > 0 else 0.0
