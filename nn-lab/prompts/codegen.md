CRITICAL CONSTRAINT — READ FIRST:
This lab has LIMITED GPU time. Every experiment MUST be as small and fast as possible.
- DEFAULT to synthetic datasets (synthetic_quadratic, synthetic_regression) — they train in SECONDS
- ONLY use MNIST/FashionMNIST if the claim genuinely requires real image data
- NEVER use CIFAR-10 unless the claim explicitly requires colour images or multi-channel convolutions
- Model size: 2-4 layers, width 32-128. Do NOT build large models unless the claim is about scale
- Epochs: 10-20 maximum. Do NOT train to convergence unless convergence IS the signal
- Batch size: 64-128. Do NOT use small batches that slow training
- If in doubt, use synthetic_quadratic with a 3-layer MLP. This answers most claims about training dynamics in under 30 seconds.

Write a complete Python training script to test the hypothesis below.

You have full access to PyTorch and can write arbitrary training code — custom models, custom loss functions, custom training loops, novel architectures. You are NOT limited to fixed menus.

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

## Dataset & Model Size Strategy

**Use the smallest experiment that can answer the question.** This lab runs real training — large datasets and models waste GPU time without adding signal.

**Decision order:**
1. **Synthetic datasets first** (`synthetic_quadratic`, `synthetic_regression`) — use these when the claim is about optimizer dynamics, gradient behaviour, loss landscape geometry, convergence properties, or any training signal that doesn't depend on real data. These are tiny, fast, and isolate the variable you're testing.
2. **MNIST / Fashion-MNIST** — use when you need a real classification task but the claim isn't about dataset complexity. 28x28 grayscale, trains in seconds.
3. **CIFAR-10** — use ONLY when the claim specifically requires colour images, higher input dimensionality, or multi-channel convolutions. CIFAR-10 is slow — avoid it unless the hypothesis demands it.
4. **Custom tiny datasets** — for claims about data properties (noise, class imbalance, distribution shift), generate a small synthetic dataset directly with PyTorch rather than loading a large one.

**Model size:** Use 2-4 layer networks with width 32-128 unless the claim is specifically about scale. A small MLP on synthetic data answers most structural claims. Don't build a ResNet when an MLP suffices.

**Epochs:** Use the minimum needed to observe the signal (often 10-20). Don't train to convergence unless convergence behaviour IS the signal.

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

## Rules

1. You MUST set a global `result` variable. The executor checks for this.
2. Use `torch.manual_seed(seed)` for reproducibility on each run.
3. Network access is blocked. Do not attempt HTTP requests.
4. Save any chart PNGs to `artifact_dir`.
5. All numeric values must be JSON-serializable — convert tensors with `.item()`, numpy arrays with `.tolist()`.
6. Wrap the experiment in try/except and set `result = {"_error": str(e)}` on failure.
7. **Minimise experiment size.** Default to synthetic datasets + small models. Only use MNIST if you need real data. Only use CIFAR-10 if the claim requires colour/multi-channel input. See "Dataset & Model Size Strategy" above.
8. Use `detect_device()` for GPU/CPU selection.

Respond with JSON: {"code": "your complete training script here"}
