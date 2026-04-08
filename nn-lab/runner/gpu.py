"""
GPU detection and device selection.

Supports NVIDIA (CUDA), Intel (XPU), and AMD (ROCm) GPUs.
Round-robins across available GPUs so concurrent jobs spread across devices.
Enforces honest refusal when GPU is required but unavailable.
"""

import os
import torch

# File-based counter for round-robin when memory is equal (cold start).
# Lives next to the data/ dir so it persists across jobs.
_COUNTER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", ".gpu_next"
)


def _read_and_advance_counter(n_gpus: int) -> int:
    """Atomically read the next GPU index and advance the counter."""
    os.makedirs(os.path.dirname(_COUNTER_PATH), exist_ok=True)
    idx = 0
    try:
        with open(_COUNTER_PATH, "r") as f:
            idx = int(f.read().strip()) % n_gpus
    except (FileNotFoundError, ValueError):
        idx = 0
    try:
        with open(_COUNTER_PATH, "w") as f:
            f.write(str((idx + 1) % n_gpus))
    except OSError:
        pass  # non-fatal — worst case two jobs hit the same GPU once
    return idx


def _pick_least_loaded(candidates: list[dict]) -> dict:
    """
    Pick the GPU with the least memory currently in use.
    Falls back to round-robin counter if all GPUs report equal usage (cold start).
    """
    if len(candidates) <= 1:
        return candidates[0]

    # Check actual memory usage per CUDA device
    usage = []
    for c in candidates:
        dev = c["device"]
        if dev.type == "cuda":
            mem = torch.cuda.memory_reserved(dev.index)
            usage.append((mem, c))
        elif dev.type == "xpu":
            # XPU doesn't have a standard memory query — use counter
            usage.append((0, c))
        else:
            usage.append((0, c))

    # If all equal (cold start), use file counter for tiebreaking
    mem_values = [u[0] for u in usage]
    if len(set(mem_values)) <= 1:
        idx = _read_and_advance_counter(len(candidates))
        return candidates[idx]

    # Otherwise pick lowest usage
    usage.sort(key=lambda x: x[0])
    return usage[0][1]


def detect_device(requested_vendor: str | None = None) -> dict:
    """
    Detect available compute device, round-robin across GPUs.

    Returns:
        {
            "device": torch.device,
            "compute": "gpu" | "cpu",
            "gpu_vendor": str | None,
            "gpu_model": str | None,
        }

    Raises RuntimeError if GPU is requested but unavailable.
    """
    candidates = []

    # NVIDIA CUDA
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            candidates.append({
                "device": torch.device(f"cuda:{i}"),
                "compute": "gpu",
                "gpu_vendor": "nvidia",
                "gpu_model": props.name,
                "memory_gb": round(props.total_mem / (1024 ** 3), 1),
            })

    # Intel XPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            name = torch.xpu.get_device_name(i) if hasattr(torch.xpu, 'get_device_name') else 'Intel XPU'
            candidates.append({
                "device": torch.device(f"xpu:{i}"),
                "compute": "gpu",
                "gpu_vendor": "intel",
                "gpu_model": name,
                "memory_gb": None,
            })

    # AMD ROCm (presents as CUDA in PyTorch with ROCm backend)
    for c in candidates:
        if c["gpu_vendor"] == "nvidia" and c["gpu_model"] and "gfx" in c["gpu_model"].lower():
            c["gpu_vendor"] = "amd"

    # Filter by requested vendor
    if requested_vendor:
        vendor_lower = requested_vendor.lower()
        filtered = [c for c in candidates if c["gpu_vendor"] == vendor_lower]
        if not filtered:
            available = ", ".join(set(c["gpu_vendor"] for c in candidates)) or "none"
            raise RuntimeError(
                f"not testable on current hardware: requested {requested_vendor} GPU "
                f"but available GPUs: {available}"
            )
        return _pick_least_loaded(filtered)

    # Pick least-loaded GPU
    if candidates:
        return _pick_least_loaded(candidates)

    # CPU fallback
    return {
        "device": torch.device("cpu"),
        "compute": "cpu",
        "gpu_vendor": None,
        "gpu_model": None,
    }


def should_require_gpu(spec: dict) -> bool:
    """Decide if this experiment needs GPU based on model scale and dataset."""
    dataset = spec.get("training", {}).get("dataset", "")
    ds_name = dataset if isinstance(dataset, str) else dataset.get("name", "")

    # Synthetic small-dim tasks on tiny models can run on CPU
    if ds_name.startswith("synthetic_"):
        arch = spec.get("architecture", {})
        dim = arch.get("width", 64)
        depth = arch.get("depth", 2)
        if dim <= 128 and depth <= 4:
            return False

    # Real datasets always prefer GPU
    if ds_name in ("mnist", "cifar10", "fashion_mnist"):
        arch = spec.get("architecture", {})
        if arch.get("depth", 2) <= 2 and arch.get("width", 64) <= 64:
            return False  # tiny model on MNIST is fine on CPU
        return True

    return True
