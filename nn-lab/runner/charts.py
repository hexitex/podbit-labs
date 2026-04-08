"""
Chart generation — produces PNG files alongside structured data.

Clean, functional style. No decorative elements.
"""

import os
from typing import Any

# Lazy import matplotlib to avoid startup cost when charts aren't needed
_plt = None
_np = None


def _ensure_mpl():
    global _plt, _np
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        _plt = plt
        _np = np


# =============================================================================
# CHART STYLE
# =============================================================================

def _apply_style():
    """Consistent clean style."""
    _plt.rcParams.update({
        'figure.figsize': (8, 5),
        'figure.dpi': 120,
        'font.size': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


CONDITION_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#ea580c', '#7c3aed',
                    '#0891b2', '#be185d', '#65a30d', '#9333ea', '#0d9488']


# =============================================================================
# INDIVIDUAL CHARTS
# =============================================================================

def chart_loss_curves(conditions: list[dict], artifact_dir: str) -> dict | None:
    """Line chart: loss curves per condition. Mean with std band if multiple runs."""
    _ensure_mpl()
    _apply_style()

    fig, (ax_train, ax_val) = _plt.subplots(1, 2, figsize=(14, 5))

    for idx, cond in enumerate(conditions):
        label = cond["label"]
        color = CONDITION_COLORS[idx % len(CONDITION_COLORS)]
        runs = cond["runs"]

        train_curves = [r["loss_curve"]["train"] for r in runs if "loss_curve" in r]
        val_curves = [r["loss_curve"]["val"] for r in runs if "loss_curve" in r]

        if not train_curves:
            continue

        # Align lengths
        min_len = min(len(c) for c in train_curves)
        train_arr = _np.array([c[:min_len] for c in train_curves])
        val_arr = _np.array([c[:min_len] for c in val_curves]) if val_curves else None
        epochs = _np.arange(1, min_len + 1)

        # Train
        mean_t = train_arr.mean(axis=0)
        ax_train.plot(epochs, mean_t, color=color, label=label)
        if train_arr.shape[0] > 1:
            std_t = train_arr.std(axis=0)
            ax_train.fill_between(epochs, mean_t - std_t, mean_t + std_t, color=color, alpha=0.15)

        # Val
        if val_arr is not None:
            mean_v = val_arr.mean(axis=0)
            ax_val.plot(epochs, mean_v, color=color, label=label)
            if val_arr.shape[0] > 1:
                std_v = val_arr.std(axis=0)
                ax_val.fill_between(epochs, mean_v - std_v, mean_v + std_v, color=color, alpha=0.15)

    ax_train.set_xlabel('Epoch')
    ax_train.set_ylabel('Loss')
    ax_train.set_title('Training Loss')
    ax_train.legend()

    ax_val.set_xlabel('Epoch')
    ax_val.set_ylabel('Loss')
    ax_val.set_title('Validation Loss')
    ax_val.legend()

    _plt.tight_layout()
    path = os.path.join(artifact_dir, "loss_curves.png")
    _plt.savefig(path, bbox_inches='tight')
    _plt.close(fig)

    return {"name": "loss_curves.png", "type": "loss_curve"}


def chart_sparsity_by_layer(conditions: list[dict], artifact_dir: str) -> dict | None:
    """Grouped bar chart: sparsity per layer per condition."""
    _ensure_mpl()
    _apply_style()

    # Collect layer names from first condition's first run
    all_layers = set()
    for cond in conditions:
        for run in cond["runs"]:
            if "sparsity_by_layer" in run:
                for entry in run["sparsity_by_layer"]:
                    all_layers.add(entry["layer"])

    if not all_layers:
        return None

    layers = sorted(all_layers)
    n_conditions = len(conditions)
    x = _np.arange(len(layers))
    width = 0.8 / max(n_conditions, 1)

    fig, ax = _plt.subplots(figsize=(max(8, len(layers) * 1.5), 5))

    for idx, cond in enumerate(conditions):
        sparsities = []
        for layer in layers:
            vals = []
            for run in cond["runs"]:
                if "sparsity_by_layer" in run:
                    for entry in run["sparsity_by_layer"]:
                        if entry["layer"] == layer:
                            vals.append(entry["sparsity"])
            sparsities.append(_np.mean(vals) if vals else 0.0)

        ax.bar(x + idx * width, sparsities, width,
               label=cond["label"], color=CONDITION_COLORS[idx % len(CONDITION_COLORS)])

    ax.set_xlabel('Layer')
    ax.set_ylabel('Sparsity')
    ax.set_title('Sparsity by Layer')
    ax.set_xticks(x + width * (n_conditions - 1) / 2)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.legend()

    _plt.tight_layout()
    path = os.path.join(artifact_dir, "sparsity_by_layer.png")
    _plt.savefig(path, bbox_inches='tight')
    _plt.close(fig)

    return {"name": "sparsity_by_layer.png", "type": "sparsity_comparison"}


def chart_gradient_stats(conditions: list[dict], artifact_dir: str) -> dict | None:
    """Line chart: gradient magnitude over epochs per layer."""
    _ensure_mpl()
    _apply_style()

    has_data = any(
        "gradient_stats_by_layer" in run
        for cond in conditions for run in cond["runs"]
    )
    if not has_data:
        return None

    fig, axes = _plt.subplots(1, 2, figsize=(14, 5))

    for idx, cond in enumerate(conditions):
        color = CONDITION_COLORS[idx % len(CONDITION_COLORS)]
        label = cond["label"]

        # Average across runs
        all_snapshots = [r.get("gradient_stats_by_layer", []) for r in cond["runs"]]
        if not all_snapshots or not all_snapshots[0]:
            continue

        # Plot mean gradient magnitude over epochs (averaged across layers)
        for snap_list in all_snapshots[:1]:  # first run for clarity
            epochs_g = [s["epoch"] for s in snap_list]
            mean_grads = [_np.mean([st["mean"] for st in s["stats"]]) for s in snap_list]
            max_grads = [max(st["max"] for st in s["stats"]) for s in snap_list]

            axes[0].plot(epochs_g, mean_grads, color=color, label=label)
            axes[1].plot(epochs_g, max_grads, color=color, label=label)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean Gradient')
    axes[0].set_title('Mean Gradient Magnitude')
    axes[0].legend()

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Max Gradient')
    axes[1].set_title('Max Gradient Magnitude')
    axes[1].legend()

    _plt.tight_layout()
    path = os.path.join(artifact_dir, "gradient_stats.png")
    _plt.savefig(path, bbox_inches='tight')
    _plt.close(fig)

    return {"name": "gradient_stats.png", "type": "gradient_stats"}


def chart_comparison_boxplot(conditions: list[dict], comparison: dict,
                              artifact_dir: str) -> dict | None:
    """Box/strip plot for comparison metrics across runs."""
    _ensure_mpl()
    _apply_style()

    if not comparison:
        return None

    metrics = [k for k, v in comparison.items() if isinstance(v, dict) and "per_run" in v]
    if not metrics:
        return None

    fig, axes = _plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        data = comparison[metric]
        per_run = data.get("per_run", [])
        if per_run:
            axes[i].boxplot([per_run], labels=[metric.replace('_', '\n')])
            # Overlay individual points
            axes[i].scatter([1] * len(per_run), per_run, alpha=0.5, zorder=3)
        axes[i].set_title(metric.replace('_', ' ').title())
        mean_val = data.get("mean", 0)
        axes[i].axhline(y=mean_val, color='red', linestyle='--', alpha=0.5, label=f'mean={mean_val:.4f}')
        axes[i].legend()

    _plt.tight_layout()
    path = os.path.join(artifact_dir, "comparison_metrics.png")
    _plt.savefig(path, bbox_inches='tight')
    _plt.close(fig)

    return {"name": "comparison_metrics.png", "type": "comparison_boxplot"}


# =============================================================================
# DISPATCHER
# =============================================================================

def generate_charts(conditions: list[dict], comparison: dict | None,
                    measurements: list, artifact_dir: str) -> list[dict]:
    """Generate all applicable charts. Returns list of chart metadata."""
    measurement_names = set()
    for m in measurements:
        measurement_names.add(m if isinstance(m, str) else m.get("name", ""))

    charts = []

    # Only chart if >1 condition or >1 run (single scalar results don't need charts)
    multi = len(conditions) > 1 or (conditions and len(conditions[0].get("runs", [])) > 1)

    if "loss_curve" in measurement_names and multi:
        c = chart_loss_curves(conditions, artifact_dir)
        if c:
            charts.append(c)

    if "sparsity_by_layer" in measurement_names and len(conditions) > 1:
        c = chart_sparsity_by_layer(conditions, artifact_dir)
        if c:
            charts.append(c)

    if "gradient_stats_by_layer" in measurement_names and multi:
        c = chart_gradient_stats(conditions, artifact_dir)
        if c:
            charts.append(c)

    if comparison:
        c = chart_comparison_boxplot(conditions, comparison, artifact_dir)
        if c:
            charts.append(c)

    return charts
