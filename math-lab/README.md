# math-lab

> **Status: alpha (`0.6.1-alpha.1`).** Part of [podbit-labs](../README.md). Designed to run alongside a [Podbit](https://github.com/hexitex/PodBit) instance.

Computational verification of mathematical claims. Math-lab takes an experiment specification (a claim plus setup parameters), asks an LLM to generate Python computation code, runs it in the sandboxed [executor.py](../lab-core/executor.py), and produces a structured verdict (supported / refuted / inconclusive).

## Spec types

Configured in [config.json](config.json) under `capabilities.specTypes`:

- **`math`** — pure numerical computation with numpy / scipy / sympy / mpmath; verify identities, evaluate expressions, compare against known constants
- **`parameter_sweep`** — sweep a parameter across a range and observe behaviour
- **`convergence_analysis`** — test whether a series, sequence, or iterative formula converges
- **`curve_shape`** — monotonicity, extrema, inflection points, bounds for mathematically-defined functions
- **`quantum_model`** — density matrices, Hamiltonians, entanglement, Bell inequalities, decoherence
- **`wave_system`** — transfer matrices, band structures, topological invariants, waveguide modes
- **`coupled_dynamics`** — ODE/SDE integration for optomechanics, nonlinear oscillators, feedback loops
- **`many_body_model`** — BCS / BdG, Green's functions, spectral functions, self-energy

## Podbit integration

Math-lab is not standalone. LLM calls (codegen and evaluation) are routed through Podbit's `/api/llm/call` proxy at `podbit.url` (default `http://localhost:4710`) using the subsystem `lab:math-lab`. Podbit owns model assignment, semaphore concurrency, rate-limit cooldowns, and budget tracking. Assign a model to `lab:math-lab` on Podbit's Models page before submitting jobs.

## Running

```bash
npm install
npm run dev          # tsx, live reload
# or
npm run build && npm start
```

Server defaults to port `4714`. Web UI at `http://localhost:4714/ui` ([public/queue.html](public/queue.html)) shows the queue with SSE updates, error traces, and verdicts.

## Python dependencies

The sandbox interpreter is resolved from `sandbox.pythonPath` in [config.json](config.json) (default `python`). The full scientific stack is required:

- `numpy`
- `scipy` — `optimize`, `integrate`, `special`, `stats`, `interpolate`, `linalg`, `signal`, `fft`, `sparse`
- `sympy`
- `mpmath`
- `networkx`
- `numba` (JIT for tight loops, Matsubara sums, Monte Carlo)

See the [Podbit](https://github.com/hexitex/PodBit) repo for the recommended Python environment.

## Layout

- [src/index.ts](src/index.ts) — pipeline wiring, calls `createLabServer()` from `@lab/core`
- [src/codegen.ts](src/codegen.ts) — LLM-based Python code generation per spec type
- [prompts/](prompts/) — system prompts and code templates
  - `codegen.txt`, `codegen-reference.txt` — codegen system prompt + reference
  - `triage-reference.txt` — verdict triage reference
  - `lab-identity.md` — lab description shown to the model
  - `template.py` — execution wrapper with imports and helpers
- [public/queue.html](public/queue.html) — queue UI
- `artifacts/` — generated code and outputs (TTL'd, gitignored)
- `data/` — SQLite job/result database (gitignored)
- `dist/` — compiled output (gitignored)

## Configuration

- [config.example.json](config.example.json) — documented template
- [config.json](config.json) — local config; placeholder model blocks; **never commit real keys**
- Defaults: 20 min execution timeout, 1 MB max output, network kill switch on, retry limit 2, max 2 concurrent jobs, 7-day artifact TTL
