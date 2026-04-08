"""
Experiment orchestrator — the main entry point for the Python runner.

Reads a spec, builds models, trains, collects measurements, generates charts,
and outputs structured results as JSON.
"""

import json
import os
import sys
import time
import copy
import torch

from runner.gpu import detect_device, should_require_gpu
from runner.models import build_model
from runner.datasets import load_dataset
from runner.training import train_single_run
from runner.charts import generate_charts
from runner.comparisons import compute_comparisons


def run_experiment(spec: dict, artifact_dir: str) -> dict:
    """
    Run a complete experiment from a declarative spec.

    Returns structured result dict (output as JSON to stdout by the caller).
    """
    start_time = time.time()

    os.makedirs(artifact_dir, exist_ok=True)

    # --- GPU detection ---
    requested_vendor = spec.get("compute", {}).get("gpu_vendor", None) if isinstance(spec.get("compute"), dict) else None
    requires_gpu = should_require_gpu(spec)

    try:
        device_info = detect_device(requested_vendor)
    except RuntimeError as e:
        return {"error": str(e), "result": None}

    device = device_info["device"]

    # Honest refusal: GPU required but only CPU available
    if requires_gpu and device_info["compute"] == "cpu":
        return {
            "error": "not testable on current hardware: experiment requires GPU but none available. "
                     "Silent CPU fallback is not permitted — timing and numerical behaviour would differ.",
            "result": None,
        }

    # --- Build conditions ---
    conditions_spec = _build_conditions(spec)

    # --- Load dataset (shared across conditions unless overridden) ---
    batch_size = spec.get("training", {}).get("batch_size", 64)
    measurements = spec.get("measurements", [])
    runs_count = spec.get("training", {}).get("runs", 3)
    base_seed = 42

    conditions_results = []

    for cond in conditions_spec:
        cond_spec = cond["spec"]
        label = cond["label"]

        train_loader, val_loader, test_loader, ds_info = load_dataset(cond_spec, batch_size)

        run_results = []
        for run_idx in range(runs_count):
            seed = base_seed + run_idx

            # Build a fresh model for each run
            model = build_model(cond_spec)

            run_data = train_single_run(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                spec=cond_spec,
                device=device,
                seed=seed,
                measurements=measurements,
                is_regression=ds_info["is_regression"],
            )
            run_results.append(run_data)

        conditions_results.append({
            "label": label,
            "runs": run_results,
        })

    # --- Compute comparisons ---
    comparison = {}
    comp_measurements = spec.get("comparison_measurements", [])
    if len(conditions_results) > 1 and comp_measurements:
        comparison = compute_comparisons(conditions_results, comp_measurements)

    # --- Generate charts ---
    charts = generate_charts(conditions_results, comparison, measurements, artifact_dir)

    # --- Build result ---
    elapsed_ms = int((time.time() - start_time) * 1000)

    framework_version = torch.__version__
    meta = {
        "execution_time_ms": elapsed_ms,
        "compute": device_info["compute"],
        "gpu_vendor": device_info.get("gpu_vendor"),
        "gpu_model": device_info.get("gpu_model"),
        "framework_version": f"PyTorch {framework_version}",
        "charts": charts,
    }

    result = {
        "conditions": conditions_results,
        "meta": meta,
    }
    if comparison:
        result["comparison"] = comparison

    # Save structured data as artifact
    result_path = os.path.join(artifact_dir, "result_data.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)

    return {"error": None, "result": result}


def _build_conditions(spec: dict) -> list[dict]:
    """
    Build list of condition specs from the experiment spec.

    For single-condition experiments, returns one condition with label "default".
    For comparison experiments, merges each condition's overrides with the base spec.
    """
    conditions = spec.get("conditions", None)

    if not conditions:
        # Single condition
        return [{"label": "default", "spec": spec}]

    base = {k: v for k, v in spec.items() if k != "conditions" and k != "comparison_measurements"}
    result = []

    for cond in conditions:
        merged = copy.deepcopy(base)
        label = cond.get("label", f"condition_{len(result)}")

        # Merge architecture overrides
        if "architecture" in cond:
            merged["architecture"] = {**merged.get("architecture", {}), **cond["architecture"]}

        # Merge training overrides
        if "training" in cond:
            merged["training"] = {**merged.get("training", {}), **cond["training"]}

        # Merge modifications if specified at condition level
        if "modifications" in cond:
            arch = merged.get("architecture", {})
            arch["modifications"] = cond["modifications"]
            merged["architecture"] = arch

        result.append({"label": label, "spec": merged})

    return result


def _json_default(obj):
    """JSON serialization fallback for numpy/torch types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if obj is None or (isinstance(obj, float) and (obj != obj)):  # NaN
        return None
    return str(obj)
