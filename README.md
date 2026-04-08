# podbit-labs

> **Status: alpha (`0.6.1-alpha.1`, tracking [Podbit](https://github.com/hexitex/PodBit) releases).** APIs, config schemas, database schemas, and prompts may change without notice. Not yet recommended for production use.

Lab subsystems for the [Podbit](https://github.com/hexitex/PodBit) project.

This repository contains:

- `lab-core` shared infrastructure (`@lab/core`) used by every lab
- `math-lab` computational verification of mathematical claims
- `nn-lab` neural network training and experiment execution
- `critique-lab` LLM based critique of knowledge graph nodes

## Installation

`lab-core` must be built first because every lab depends on it via `file:../lab-core`.

```bash
cd lab-core
npm install
npm run build

cd ../math-lab
npm install

cd ../nn-lab
npm install

cd ../critique-lab
npm install
```

Each lab can then be run with `npm run dev` (tsx) or `npm run build && npm start`.

### Python

Python is required by every lab. `lab-core` ships [executor.py](lab-core/executor.py), the sandbox process that runs generated code on behalf of the labs. Each lab's `config.json` points at the interpreter via `sandbox.pythonPath` (default `python`).

Per lab Python requirements:

- `math-lab` needs a substantial scientific Python stack: `numpy`, `scipy` (with `optimize`, `integrate`, `special`, `stats`, `interpolate`, `linalg`, `signal`, `fft`, `sparse`), `sympy`, `mpmath`, `networkx`, and `numba` (JIT compilation for tight loops, Matsubara sums, Monte Carlo). All spec types in [math-lab/config.json](math-lab/config.json) (math, parameter sweep, convergence analysis, curve shape, quantum model, wave system, coupled dynamics, many body model) assume this stack is present
- `nn-lab` needs `torch` (with CUDA if you want GPU training), `torchvision` (lazy imported by [nn-lab/runner/datasets.py](nn-lab/runner/datasets.py) for MNIST, FashionMNIST, CIFAR10 loaders), `numpy`, and `matplotlib` (lazy imported by [nn-lab/runner/charts.py](nn-lab/runner/charts.py) with the Agg backend for headless chart generation). The full runner lives in [nn-lab/runner/](nn-lab/runner/)
- `critique-lab` is LLM only and does not execute generated code, but still resolves `pythonPath` at startup so a working `python` on PATH is expected

See the Podbit repo for the recommended Python environment and version pinning.

## Setup

These labs are not standalone. They are designed to run alongside a Podbit instance, which owns model assignment, semaphore based concurrency, rate limit cooldowns, and budget tracking. Each lab routes its LLM calls through Podbit via the `podbit.url` and `podbit.subsystem` fields in its `config.json`.

For installation, configuration, model registry setup, and how to wire each lab into Podbit, see the main Podbit repository. Start there before attempting to run anything in this repo.

## Configuration

Each lab ships a `config.example.json` with documentation comments and a sanitized `config.json` with placeholder model blocks. Real API keys belong in `.env` files (gitignored) or in Podbit's model registry, never in `config.json`.
