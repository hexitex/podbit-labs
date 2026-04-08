"""
Fixed-menu sparsity methods.

Applied after training or during training (depending on method).
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def apply_sparsity(model: nn.Module, method: str, ratio: float) -> None:
    """
    Apply sparsity to the model in-place.

    Args:
        model: The trained model
        method: "unstructured", "structured_2_4", or "magnitude"
        ratio: Target sparsity ratio (0-1)
    """
    if method == "unstructured":
        _apply_unstructured(model, ratio)
    elif method == "magnitude":
        _apply_magnitude(model, ratio)
    elif method == "structured_2_4":
        _apply_structured_2_4(model)
    else:
        raise ValueError(f"not supported: sparsity method '{method}'")


def _apply_unstructured(model: nn.Module, ratio: float) -> None:
    """Global unstructured L1 pruning."""
    params_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            params_to_prune.append((module, 'weight'))

    if params_to_prune:
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=ratio,
        )
        # Make pruning permanent
        for module, param_name in params_to_prune:
            prune.remove(module, param_name)


def _apply_magnitude(model: nn.Module, ratio: float) -> None:
    """Per-layer magnitude pruning."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name='weight', amount=ratio)
            prune.remove(module, 'weight')


def _apply_structured_2_4(model: nn.Module) -> None:
    """
    Structured 2:4 sparsity — 2 out of every 4 weights are zero.
    This is the NVIDIA Ampere pattern. Applied per-layer to weight matrices.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                w = module.weight.data
                original_shape = w.shape
                flat = w.view(-1)

                # Pad to multiple of 4
                pad_len = (4 - flat.numel() % 4) % 4
                if pad_len > 0:
                    flat = torch.cat([flat, torch.zeros(pad_len, device=flat.device)])

                # Reshape to groups of 4 and keep top-2 by magnitude
                groups = flat.view(-1, 4)
                _, indices = groups.abs().topk(2, dim=1)
                mask = torch.zeros_like(groups)
                mask.scatter_(1, indices, 1.0)
                flat = (groups * mask)[:flat.numel() - pad_len if pad_len else flat.numel()]

                module.weight.data = flat[:w.numel()].view(original_shape)


def measure_sparsity_by_layer(model: nn.Module, threshold: float = 1e-6) -> list[dict]:
    """Measure sparsity per layer."""
    results = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            w = module.weight.data
            total = w.numel()
            near_zero = (w.abs() < threshold).sum().item()
            results.append({
                "layer": name,
                "total_params": total,
                "near_zero": near_zero,
                "sparsity": near_zero / total if total > 0 else 0.0,
            })
    return results
