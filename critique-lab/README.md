# critique-lab

> **Status: alpha (`0.6.1-alpha.1`).** Part of [podbit-labs](../README.md). Designed to run alongside a [Podbit](https://github.com/hexitex/PodBit) instance.

LLM-based critique of [Podbit](https://github.com/hexitex/PodBit) knowledge graph nodes and lab experiments. Critique-lab is **LLM-only** — it does not generate or execute code, has no sandbox, and does no computation. Every spec is text-in / text-out via the model assigned to `lab:critique-lab` on Podbit.

## Spec types

Configured in [config.json](config.json) under `capabilities.specTypes`:

- **`node_critique`** — assess a knowledge graph node on specificity, novelty, grounding, falsifiability, and clarity (each scored 1–10). Returns a verdict (supported / refuted / inconclusive) and a recommendation (`keep`, `promote`, `rework`, `demote`, `delete`).
- **`experiment_review`** — review a lab experiment that tested a knowledge graph claim. Validates whether the design correctly operationalizes the claim, surfaces confounds, and corrects the verdict if methodology is flawed. Returns a methodology score and an action (`confirm`, `correct`, `retest`).

Verdict builders live at [src/index.ts](src/index.ts) (`buildNodeCritiqueVerdict`, `buildExperimentReviewVerdict`).

## Podbit integration

Critique-lab is not standalone:

- **Node context flows from Podbit.** When a critique request arrives, Podbit supplies the node's content, metadata, parent nodes (sources), child nodes (derivatives), and prior verification attempts.
- **All LLM calls route through Podbit.** Requests are forwarded to `/api/llm/call` at `podbit.url` (default `http://localhost:4710`) using the subsystem `lab:critique-lab`. Podbit owns model assignment, rate limiting, cooldowns, and budget tracking. Assign a model to `lab:critique-lab` on Podbit's Models page before submitting jobs.
- **Verdicts persist locally** in SQLite (`critique-lab.db`) and are returned via the queue UI and REST endpoints.

## Running

```bash
npm install
npm run dev          # tsx, live reload
# or
npm run build && npm start
```

Server defaults to port `4716`. Web UI at `http://localhost:4716/ui` ([public/queue.html](public/queue.html)) shows the queue with SSE updates, error traces, and verdicts.

## Why python is required but no Python stack is

Critique-lab inherits the lab-core config schema, which includes `sandbox.pythonPath`. The path is resolved at startup for compatibility with the shared lab framework, but critique-lab never spawns the sandbox or invokes [executor.py](../lab-core/executor.py) — there is no code execution path. You only need a working `python` on PATH; you do **not** need numpy, torch, or any scientific libraries.

## Layout

- [src/index.ts](src/index.ts) — pipeline wiring
  - `pipeline.generate()` returns the critique prompt (no code generation)
  - `pipeline.execute()` is a no-op that returns a stub `SandboxResult`
  - `pipeline.evaluate()` calls the LLM and returns a structured `LabVerdict`
- [prompts/critique.md](prompts/critique.md) — `node_critique` system prompt (scoring criteria, recommendation taxonomy)
- [prompts/experiment-review.md](prompts/experiment-review.md) — `experiment_review` system prompt (operationalization, confound analysis, verdict validation)
- [public/queue.html](public/queue.html) — queue UI
- `artifacts/`, `data/` — runtime, gitignored

## Configuration

- [config.example.json](config.example.json) — documented template
- [config.json](config.json) — local config; placeholder model blocks; **never commit real keys**
- `execution.queueLimit` — max queued jobs (default 20)
