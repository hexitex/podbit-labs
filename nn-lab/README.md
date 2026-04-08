# nn-lab

> **Status: alpha (`0.6.1-alpha.1`).** Part of [podbit-labs](../README.md). Designed to run alongside a [Podbit](https://github.com/hexitex/PodBit) instance.

Declarative neural network training and experiment execution. nn-lab takes an experiment spec, builds (or LLM-generates) a PyTorch script, runs it via [executor.py](../lab-core/executor.py) against the Python runner in [runner/](runner/), collects structured measurements, and produces a verdict on the hypothesis.

## Spec types

Configured in [config.json](config.json) under `capabilities.specTypes`:

- **`nn_training_profile`** — train a single configuration; collect loss curve, convergence epoch, accuracy, gradient stats, weight norms, sparsity distribution
- **`nn_paired_comparison`** — train two or more matched configurations; compute cross-condition metrics (convergence speed ratio, accuracy divergence, gradient divergence)
- **`nn_sparsity_profile`** — apply a sparsity method, measure per-layer sparsity and accuracy impact
- **`nn_optimizer_dynamics`** — train on synthetic loss surfaces; collect effective LR, Hessian top eigenvalue
- **`nn_architecture_scaling`** — vary depth/width; measure accuracy and convergence scaling
- **`nn_codegen`** — fall-back path; LLM generates custom PyTorch (custom models, losses, optimizers, novel measurements)

## Fixed-menu spec extractor contract

The fixed menus in [config.json](config.json) (`capabilities.menus`) **are the API contract** for the Podbit spec extractor. The extractor selects from these menus and the resulting spec is validated by [src/spec-validator.ts](src/spec-validator.ts); validation failure routes the spec to `nn_codegen` instead of the deterministic builder. This is by design — fixed menus give reproducibility for the common case, while `nn_codegen` keeps the long tail open. **Do not treat the menus as a constraint on lab internals; they exist to lock the extractor's surface area.**

Menus include:

- **architectures**: `mlp`, `cnn`, `cnn_depthwise_separable`, `resnet`, `transformer_tiny`
- **datasets**: `mnist`, `fashion_mnist`, `cifar10`, `synthetic_quadratic`, `synthetic_regression`
- **optimizers**: `sgd`, `adam`, `rmsprop`, `lbfgs`
- **lr_schedules**: `constant`, `cosine`, `cyclical`, `step`
- **measurements**: `loss_curve`, `convergence_epoch`, `final_accuracy`, `sparsity_by_layer`, `gradient_stats_by_layer`, `effective_lr_by_layer`, `weight_norm_by_layer`, `hessian_top_eigenvalue`, `wall_clock_time`
- **sparsity_methods**: `unstructured`, `structured_2_4`, `magnitude`
- **modifications**: `bottleneck`, `dropout`, `weight_decay`, `batch_norm`, `layer_norm`

## Python runner

The [runner/](runner/) package is invoked by the generated script with `python -m runner <spec.json>` and returns a JSON `result` to the parent. Lazy imports keep startup cheap and headless-safe.

- [runner/datasets.py](runner/datasets.py) — MNIST, Fashion-MNIST, CIFAR-10, `synthetic_quadratic` (configurable curvature spectrum), `synthetic_regression`; deterministic train/val/test split; caches to `data/`
- [runner/training.py](runner/training.py) — SGD / Adam / RMSprop / LBFGS with optional weight decay; constant / cosine / cyclical / step schedulers; per-epoch metrics; LBFGS closure pattern; early stopping
- [runner/measurements.py](runner/measurements.py) — gradient stats, weight norms, effective learning rates, Hessian top-eigenvalue estimation
- [runner/models.py](runner/models.py) — fixed-menu architectures with optional batch norm, layer norm, dropout, bottleneck, weight decay
- [runner/sparsity.py](runner/sparsity.py) — unstructured, structured 2:4, magnitude pruning; per-layer sparsity measurement
- [runner/comparisons.py](runner/comparisons.py) — cross-condition divergence and ratio metrics
- [runner/charts.py](runner/charts.py) — loss / accuracy / comparison PNGs (lazy matplotlib, Agg backend)
- [runner/experiment.py](runner/experiment.py) — orchestrator: detects GPU/CPU, loads dataset, trains N runs across seeds, aggregates measurements, generates charts

## Podbit integration

LLM calls (used by `nn_codegen` and the verdict evaluator) are routed through Podbit's `/api/llm/call` proxy at `podbit.url` (default `http://localhost:4710`) using the subsystem `lab:nn-lab`. Assign a model to `lab:nn-lab` on Podbit's Models page before submitting jobs.

## Running

```bash
npm install
npm run build        # required: see "Build output" below
npm run dev          # tsx, live reload
# or, after build:
npm start
```

Server defaults to port `4715`. Web UI at `http://localhost:4715/ui` ([public/queue.html](public/queue.html)) shows queued / running / completed jobs with SSE updates, live stage tracking (codegen, executing, evaluating), error traces, verdicts, and cancellation.

### Build output

[package.json](package.json) `main` and `start` point at `dist/index.js`, but `dist/` is gitignored ([.gitignore](../.gitignore) line 12) and is **not** checked in. You must run `npm run build` before `npm start`. `npm run dev` does not need a build (it runs `src/index.ts` via `tsx`). This matches `math-lab` and `critique-lab`, which already have local `dist/` builds.

## Python dependencies

The sandbox interpreter is resolved from `sandbox.pythonPath` in [config.json](config.json) (default `python`). Required:

- `torch` (with CUDA if you want GPU training)
- `torchvision` (lazy-imported by [runner/datasets.py](runner/datasets.py) for MNIST / Fashion-MNIST / CIFAR-10 loaders)
- `numpy`
- `matplotlib` (lazy-imported by [runner/charts.py](runner/charts.py), Agg backend, headless)

## Layout

- [src/index.ts](src/index.ts) — pipeline wiring; routes `nn_codegen` to LLM, fixed-menu specs to the deterministic script builder
- [src/script-builder.ts](src/script-builder.ts) — converts a validated spec into a Python script that calls `runner.run_experiment`
- [src/spec-validator.ts](src/spec-validator.ts) — validates spec fields against the fixed menus
- [src/codegen.ts](src/codegen.ts) — LLM codegen path for `nn_codegen`
- [prompts/codegen.md](prompts/codegen.md) — codegen system prompt (minimal GPU usage, synthetic-by-default, output format)
- [runner/](runner/) — Python training runner (above)
- [public/queue.html](public/queue.html) — queue UI
- `artifacts/`, `data/` — runtime, gitignored

## Configuration

- [config.example.json](config.example.json) — documented template
- [config.json](config.json) — local config; placeholder model blocks; **never commit real keys**
