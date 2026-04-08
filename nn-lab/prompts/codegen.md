You have full access to PyTorch and can write arbitrary training code — custom models, custom loss functions, custom training loops, novel architectures, custom datasets, custom measurements. You are NOT limited to fixed menus.

GPU budget guidance — pick the smallest experiment that can answer the question, then scale up only when the hypothesis genuinely demands it:
- Synthetic datasets (`synthetic_quadratic`, `synthetic_regression`) train in seconds and isolate dynamics — a strong default for claims about optimisers, gradients, loss surfaces, and convergence.
- MNIST / Fashion-MNIST / CIFAR-10 are all available when the hypothesis needs real images, more channels, or harder classification.
- Model size, epoch count, and batch size are yours to choose. A 3-layer MLP on synthetic data answers many structural questions in under a minute; scale up when the hypothesis is about scale or when smaller models can't express the effect.
- Stay within the configured execution timeout. When in doubt about whether something will fit, run a smaller version first and grow it.

Write a complete Python training script to test the hypothesis below.

## Available Runner Utilities

The nn-lab runner package is on the Python path. You can import and use any of these:

```python
from runner.gpu import detect_device, should_require_gpu
# detect_device() -> {"device": torch.device, "compute": "gpu"|"cpu", "gpu_vendor": str|None, "gpu_model": str|None}
# Raises RuntimeError if GPU required but unavailable.

from runner.datasets import load_dataset
# load_dataset(spec, batch_size=64) -> (train_loader, val_loader, test_loader, info)
# info = {"num_classes": int, "input_shape": tuple, "is_regression": bool}
# Datasets: mnist, fashion_mnist, cifar10, synthetic_quadratic(dim, curvature_spec), synthetic_regression(dim, noise)
# spec format: {"training": {"dataset": "mnist"}} or {"training": {"dataset": {"name": "synthetic_quadratic", "dim": 10}}}

from runner.measurements import MeasurementCollector, collect_gradient_stats, collect_weight_norms, collect_effective_lr, compute_hessian_top_eigenvalue
# MeasurementCollector(requested_list, device) — call .on_epoch_end(epoch, train_loss, val_loss, model, optimizer) each epoch, then .finalize(model, loss_fn, test_loader, optimizer) -> dict

from runner.sparsity import apply_sparsity, measure_sparsity_by_layer
# apply_sparsity(model, method, ratio) — methods: unstructured, structured_2_4, magnitude
# measure_sparsity_by_layer(model, threshold=1e-6) -> list[dict]

from runner.charts import generate_charts
# generate_charts(conditions, comparison, measurements, artifact_dir) -> list[dict]

from runner.comparisons import compute_comparisons
# compute_comparisons(conditions, requested) -> dict

from runner.training import build_optimizer, build_scheduler, build_loss_fn, train_single_run
# build_optimizer(model, spec) -> Optimizer (spec = {"optimizer": "adam", "lr": 0.001, ...})
# build_scheduler(optimizer, spec, epochs) -> Scheduler|None
# build_loss_fn(is_regression) -> nn.Module
# train_single_run(model, train_loader, val_loader, test_loader, spec, device, seed, measurements, is_regression) -> dict
```

You can also use these directly: `torch`, `torch.nn`, `numpy`, `time`, `json`, `os`, `math`.

## Dataset & Model Size — pick what fits the hypothesis

**Use the smallest experiment that can answer the question, and grow it when the question demands it.** Real training takes real GPU time — match the scale of the experiment to the scale of the claim.

Available datasets and when each is a natural fit:
- **Synthetic** (`synthetic_quadratic`, `synthetic_regression`) — fastest, fully controllable. Strong default for claims about optimiser dynamics, gradient behaviour, loss-landscape geometry, and convergence properties.
- **MNIST / Fashion-MNIST** — small real classification tasks (28×28 grayscale). Use when you need real data but not colour or high resolution.
- **CIFAR-10** — colour images, multi-channel convolutions. Use when the hypothesis depends on real visual content, channel structure, or harder classification.
- **Custom datasets** — when none of the above fit, build a small dataset directly with PyTorch (noise injection, class imbalance, distribution shift, custom regression targets, etc.).

Model size and training length are yours to choose. A small MLP on synthetic data answers many structural claims; reach for ResNet/Transformer-Tiny when the hypothesis is about depth, residual structure, or attention. Train for as many epochs as the signal needs and the timeout allows.

## Using Runner Utilities vs Custom Code

You can freely mix runner utilities with your own code. Use the utilities when they cover what you need — write custom code when they don't. For example:
- Use `load_dataset` for standard datasets, but create your own `DataLoader` for custom data
- Use `build_optimizer` for standard optimizers, but instantiate custom ones directly
- Write your own model class instead of using the runner's fixed architectures
- Write your own training loop instead of using `train_single_run`

## Required Output Format

Your code MUST set a global variable called `result` with this structure:

```python
result = {
    "conditions": [
        {
            "label": "descriptive_name",
            "runs": [
                {
                    "seed": 42,
                    "loss_curve": {"train": [float, ...], "val": [float, ...]},
                    "final_accuracy": float,  # or final_loss for regression
                    # ... any additional measurements
                },
            ]
        },
        # For comparison experiments, add more conditions
    ],
    "meta": {
        "execution_time_ms": int(elapsed_ms),
        "compute": "gpu" or "cpu",
        "charts": [],  # list of {"name": "filename.png", "type": "description"} if you generate charts
    }
}
```

If the experiment fails due to hardware limitations, set:
```python
result = {"_error": "not testable on current hardware: <reason>"}
```

## Conventions

1. Set a global `result` variable — the executor reads it. (See the structure above.)
2. Seed each run with `torch.manual_seed(seed)` for reproducibility.
3. Use `detect_device()` for GPU/CPU selection.
4. Save any chart PNGs to `artifact_dir`.
5. All numeric values must be JSON-serializable — convert tensors with `.item()`, numpy arrays with `.tolist()`.
6. Wrap the experiment in try/except and set `result = {"_error": str(e)}` on failure.

## Sandbox (real, runtime-enforced)

- Network access is blocked — no HTTP requests, no remote downloads. Datasets must come from `runner.datasets` or be generated in-process.
- File I/O is restricted to `artifact_dir`.
- No subprocess.

Respond with JSON: {"code": "your complete training script here"}
