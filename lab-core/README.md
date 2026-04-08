# lab-core

> **Status: alpha (`0.6.1-alpha.1`).** Internal package for the [podbit-labs](../README.md) repo. Not published to npm — consumed via `file:../lab-core`.

Shared TypeScript infrastructure (`@lab/core`) for every lab in this repo. Each lab implements a small `LabPipeline` interface and lets `lab-core` provide the HTTP server, job queue, sandbox, persistence, LLM client, and verdict plumbing.

## What it provides

`createLabServer()` (see [src/server.ts](src/server.ts)) wires up an Express server that exposes the standard lab contract: job submission, queue/SSE, artifact storage, metrics, auth, and graceful shutdown. Labs only need to supply their `generate` / `execute` / `evaluate` steps.

Key exports from [src/index.ts](src/index.ts):

- `createLabServer(options)` — main factory; takes a `LabPipeline`, `BaseLabConfig`, and capabilities builder
- `LabPipeline`, `ExperimentSpec`, `SandboxResult`, `LabVerdict` — core contract types ([src/types.ts](src/types.ts))
- `executePython()` — spawn the Python sandbox with timeout + JSON capture ([src/sandbox.ts](src/sandbox.ts))
- `evaluate()` — generic verdict evaluator with optional LLM interpretation ([src/evaluator.ts](src/evaluator.ts))
- `callLlm()` / `callLlmJson()` — LLM client supporting Anthropic, OpenAI, and OpenAI-compatible endpoints (also routes through Podbit when configured)
- `JobQueue` — concurrent job execution with retries and cooldowns
- `db.*` — SQLite helpers for jobs, artifacts, results
- `loadConfig()`, `loadModelSlot()`, `setConfig()` — config loaders
- `logger` — pino-based structured logger

Sub-path exports: `@lab/core/sandbox`, `@lab/core/evaluator`.

## executor.py

[executor.py](executor.py) is the Python harness that every lab uses to run generated code. It is spawned by `executePython()` and:

- Runs an arbitrary Python file and captures its `result` variable (or `main()` return) as JSON on stdout
- Suppresses user `print()` so only the structured result reaches the parent
- Patches the `socket` module at the C level to **block all network access** by default (`network_kill` flag)
- Returns `{"result": ..., "error": ..., "execution_time_ms": ...}` so the parent can produce a verdict
- Timeouts are enforced by the parent (`sandbox.executionTimeoutMs` in each lab's `config.json`)

Labs that don't execute code (e.g. [critique-lab](../critique-lab/README.md)) still resolve `pythonPath` at startup but never invoke `executor.py`.

## Building

`lab-core` must be built before any lab can be installed, because each lab depends on it via `file:../lab-core` and consumes the compiled output in `dist/`.

```bash
npm install
npm run build
```

Output goes to `dist/` (gitignored). The compiled entrypoint is `dist/index.js` with sub-path entrypoints `dist/sandbox.js` and `dist/evaluator.js` (see [package.json](package.json) `exports`).

## How labs consume it

```ts
import { createLabServer, LabPipeline } from '@lab/core';

const pipeline: LabPipeline = {
  async generate(spec) { /* LLM codegen */ },
  async execute(spec, code) { /* run via executePython */ },
  async evaluate(spec, sandboxResult) { /* return LabVerdict */ },
};

await createLabServer({ pipeline, config, capabilities });
```

See [math-lab/src/index.ts](../math-lab/src/index.ts), [nn-lab/src/index.ts](../nn-lab/src/index.ts), and [critique-lab/src/index.ts](../critique-lab/src/index.ts) for real implementations.
