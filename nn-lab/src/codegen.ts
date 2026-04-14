/**
 * NN-Lab Codegen — generates custom PyTorch training code via LLM.
 *
 * For experiments that go beyond fixed menus. The LLM can write arbitrary
 * PyTorch code — custom models, loss functions, training loops — while
 * using the runner utilities for datasets, GPU detection, and measurements.
 *
 * Bias isolation is architectural, not menu-based:
 *   1. Lab never sees claim narrative — only hypothesis + parameters
 *   2. Anti-tautology check in spec extractor catches circular specs
 *   3. Lab returns structured data, not verdicts
 *   4. Prior rejection memory prevents retry-until-confirmation
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { callLlm, getConfig, LabError, executePython, type CodegenResult, type ExperimentSpec } from '@lab/core';

// =============================================================================
// PROMPT
// =============================================================================

const PROMPT_PATH = join(process.cwd(), 'prompts', 'codegen.md');

let promptCache: string | null = null;
function getSystemPrompt(): string {
    if (!promptCache) promptCache = readFileSync(PROMPT_PATH, 'utf-8');
    return promptCache;
}

// Parent of the runner package — so `from runner import ...` resolves
const RUNNER_PARENT = process.cwd().replace(/\\/g, '/');

function buildPrompt(spec: ExperimentSpec, artifactDir: string, previousAttempts?: Array<{ code: string; error: string }>): string {
    const setupDesc = spec.setup
        ? Object.entries(spec.setup)
            .map(([k, v]) => `  ${k}: ${JSON.stringify(v)}`)
            .join('\n')
        : '  (no additional parameters)';

    let errorFeedback = '';
    if (previousAttempts?.length) {
        const latest = previousAttempts[previousAttempts.length - 1];
        const earlier = previousAttempts.slice(0, -1);
        const earlierBlock = earlier.length > 0
            ? earlier.map((a, i) => `Attempt ${i + 1} error: ${a.error.slice(0, 300)}`).join('\n') + '\n\n'
            : '';
        errorFeedback = `\n\nPREVIOUS FAILED ATTEMPTS — fix the error. Keep what works, fix what's broken.

${earlierBlock}YOUR PREVIOUS CODE:
\`\`\`python
${latest.code.slice(0, 3000)}
\`\`\`

ERROR:
${latest.error}

Fix the broken line and return the full corrected code.`;
    }

    const constraints = spec.setup?.constraints || {};
    const maxEpochs = constraints.max_epochs ?? 20;
    const maxRuns = constraints.max_runs ?? 3;

    return `${getSystemPrompt()}

## Experiment

HYPOTHESIS: ${spec.hypothesis}

SETUP:
${setupDesc}

BUDGET (guidance — exceed when the hypothesis genuinely needs it, but stay inside the execution timeout):
  max_epochs: ${maxEpochs}     # default cap; raise it in the spec if convergence requires more
  max_runs: ${maxRuns}          # number of seeds per condition; more = better statistics, more GPU
  artifact_dir: ${artifactDir.replace(/\\/g, '/')}
  Dataset: synthetic datasets are fastest and isolate dynamics; MNIST/Fashion-MNIST/CIFAR-10 are all available — pick whichever the hypothesis actually needs.
  Model: pick the smallest architecture that can express the hypothesis. Scale up when scale itself is the variable.
  Target wall time: aim for ~2 minutes per run when possible — longer is fine when justified.
${errorFeedback}`;
}

// =============================================================================
// CODE EXTRACTION
// =============================================================================

function extractCode(rawResponse: string): string {
    // 1. Try JSON: {"code": "..."}
    try {
        const parsed = JSON.parse(rawResponse);
        if (parsed.code) return parsed.code;
    } catch { /* not JSON */ }

    // 2. Try JSON inside markdown fence
    const jsonBlock = rawResponse.match(/```json\s*([\s\S]*?)```/);
    if (jsonBlock) {
        try { const c = JSON.parse(jsonBlock[1]).code; if (c) return c; } catch { /* fall through */ }
    }

    // 3. Try extracting JSON object from anywhere
    const obj = rawResponse.match(/\{[\s\S]*\}/);
    if (obj) {
        try { const c = JSON.parse(obj[0]).code; if (c) return c; } catch { /* fall through */ }
    }

    // 4. Try Python code block
    const pyBlock = rawResponse.match(/```(?:python|py)\s*\n([\s\S]*?)```/);
    if (pyBlock) return pyBlock[1].trim();

    // 5. Try any code block
    const anyBlock = rawResponse.match(/```\s*\n([\s\S]*?)```/);
    if (anyBlock) return anyBlock[1].trim();

    return '';
}

/**
 * Detect chain-of-thought leakage in generated code.
 * Some models (especially smaller ones) embed reasoning, self-corrections,
 * or HTML artifacts directly in the code string. Catching this early avoids
 * wasting a preflight attempt on obviously broken code.
 */
function detectThinkingLeakage(code: string): string | null {
    const lines = code.split('\n');
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const lineNum = i + 1;

        // Self-correction comments: "# Wait", "# Actually", "# I see", "# Hmm"
        if (/^[^#]*#\s*(Wait|Actually|I see|Hmm|Let me|I think|I need to|I will|I should|Oops|Sorry)\b/i.test(line)) {
            return `Line ${lineNum}: chain-of-thought leaked into code: "${line.trim().slice(0, 120)}"`;
        }

        // HTML-like fragments that aren't in string literals
        if (!line.match(/^\s*#/) && !line.match(/['"].*<\/?[a-z].*['"]/) && /<\/?\w+[^>]*>/.test(line)) {
            return `Line ${lineNum}: HTML fragment in code: "${line.trim().slice(0, 120)}"`;
        }

        // Multiple def/class statements on a single line (stream-of-consciousness)
        if (/\)\s+(def |class )\w/.test(line)) {
            return `Line ${lineNum}: multiple definitions crammed on one line: "${line.trim().slice(0, 120)}"`;
        }
    }

    return null;
}

// =============================================================================
// SCRIPT WRAPPING
// =============================================================================

function wrapScript(generatedCode: string, artifactDir: string, preflight: boolean = false): string {
    const preflightPatch = preflight ? `
# ─── PREFLIGHT MODE: patch training to 1 epoch, 2 batches ───
_preflight_mode = True
_max_preflight_batches = 2
_PREFLIGHT_MAX_EPOCHS = 1

class _PreflightDataLoader:
    """Wraps a DataLoader to yield only a few batches."""
    def __init__(self, loader, max_batches=2):
        self._loader = loader
        self._max = max_batches
    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._max: break
            yield batch
    def __len__(self):
        return min(len(self._loader), self._max) if hasattr(self._loader, '__len__') else self._max
    @property
    def dataset(self):
        return self._loader.dataset

# Patch load_dataset to return truncated loaders
try:
    from runner import datasets as _ds
    _orig_load = _ds.load_dataset
    def _patched_load(*a, **kw):
        train, val, test, info = _orig_load(*a, **kw)
        return _PreflightDataLoader(train), _PreflightDataLoader(val), _PreflightDataLoader(test), info
    _ds.load_dataset = _patched_load
except Exception:
    pass

# Patch train_single_run to limit epochs (targeted, not global range())
try:
    from runner import training as _tr
    _orig_train = _tr.train_single_run
    def _patched_train(model, train_loader, val_loader, test_loader, spec, device, seed, measurements, is_regression=False):
        # Override epochs in the spec to 1 for preflight
        if isinstance(spec, dict):
            spec = {**spec}
            if 'training' in spec and isinstance(spec['training'], dict):
                spec['training'] = {**spec['training'], 'epochs': _PREFLIGHT_MAX_EPOCHS}
        return _orig_train(model, _PreflightDataLoader(train_loader), _PreflightDataLoader(val_loader), _PreflightDataLoader(test_loader), spec, device, seed, measurements, is_regression)
    _tr.train_single_run = _patched_train
except Exception:
    pass

# NOTE: We do NOT patch range() globally. The old _PatchedRange replaced ALL
# range(N) where N>2 with range(1), which broke list iteration, data processing,
# and result aggregation — causing "list index out of range" errors.
# The DataLoader patch above (2 batches per epoch) is sufficient: even 50 epochs
# with 2 batches each completes in seconds, which is fast enough for preflight.
# ─── END PREFLIGHT PATCH ───
` : '';

    return `#!/usr/bin/env python3
"""NN-Lab custom experiment — LLM-generated."""

import sys
import os
import json
import time
import math

# Add nn-lab root to path so runner utilities are available
sys.path.insert(0, ${JSON.stringify(RUNNER_PARENT)})

import torch
import torch.nn as nn
import numpy as np

# GPU round-robin — uses runner.gpu to spread across available devices
from runner.gpu import detect_device
_device_info = detect_device()
device = _device_info["device"]

${preflightPatch}
artifact_dir = ${JSON.stringify(artifactDir.replace(/\\/g, '/'))}
os.makedirs(artifact_dir, exist_ok=True)

_start_time = time.time()

try:
${generatedCode.split('\n').map(line => '    ' + line).join('\n')}

    # Ensure meta has execution time
    if isinstance(result, dict) and "meta" not in result and "_error" not in result:
        result["meta"] = {
            "execution_time_ms": int((time.time() - _start_time) * 1000),
            "compute": "gpu" if torch.cuda.is_available() else "cpu",
            "charts": [],
        }
    elif isinstance(result, dict) and "meta" in result:
        result["meta"]["execution_time_ms"] = int((time.time() - _start_time) * 1000)

except Exception as _e:
    import traceback as _tb_mod
    # Write actual error to stderr so the retry loop can see it and feed it back to the LLM
    sys.stderr.write(f"EXPERIMENT ERROR: {type(_e).__name__}: {_e}\\n{_tb_mod.format_exc()}\\n")
    # Re-raise so executor sets exit code 1, triggering server-level retry
    raise
`;
}

// =============================================================================
// PREFLIGHT VALIDATION
// =============================================================================

/**
 * Translate sandbox line numbers to generated-code-relative line numbers.
 * The wrapper adds ~30 lines (non-preflight) or ~80 lines (preflight)
 * before the try: block where generated code is indented.
 */
function enrichNNError(error: string, generatedCode: string, preflight: boolean): string {
    // Count wrapper lines before the generated code insertion point
    const wrapperBefore = wrapScript('__MARKER__', '/tmp', preflight);
    const markerIndex = wrapperBefore.split('\n').findIndex(l => l.includes('__MARKER__'));
    const offset = markerIndex >= 0 ? markerIndex : (preflight ? 80 : 30);

    return error.replace(/line (\d+)/g, (match, num) => {
        const absLine = parseInt(num, 10);
        const relLine = absLine - offset;
        if (relLine < 1) return match; // line is in the wrapper, not generated code

        const codeLines = generatedCode.split('\n');
        if (relLine > codeLines.length) return match;

        // Show the broken line
        const contextStart = Math.max(0, relLine - 3);
        const contextEnd = Math.min(codeLines.length, relLine + 2);
        const snippet = codeLines.slice(contextStart, contextEnd)
            .map((l, i) => {
                const ln = contextStart + i + 1;
                return `${ln === relLine ? ' >>>' : '    '} ${ln}: ${l}`;
            }).join('\n');

        return `line ${relLine} of your code\n\nYour code around the error:\n${snippet}`;
    });
}

/**
 * Run a quick preflight check: syntax validation + 1-epoch dry run.
 * Catches compile errors, import errors, shape mismatches, and API misuse
 * in seconds instead of 30 minutes.
 */
async function preflightValidate(generatedCode: string, artifactDir: string): Promise<string | null> {
    const preflightCode = wrapScript(generatedCode, artifactDir, true);
    const result = await executePython(preflightCode);

    if (result.success && result.parsedOutput) {
        const data = result.parsedOutput.result;
        if (data?._error) return `Preflight runtime error: ${data._error}`;
        return null; // passed
    }

    const stderr = result.stderr || '';
    const lines = stderr.trim().split('\n');
    const raw = lines.length <= 5 ? lines.join('\n') : lines.slice(-8).join('\n');

    // Translate line numbers so the LLM can find the broken code
    return `Preflight failed:\n${enrichNNError(raw, generatedCode, true)}`;
}

// =============================================================================
// PUBLIC API
// =============================================================================

/** Max time for a single codegen LLM call (5 minutes). The model config
 *  timeoutMs is shared with evaluation and can be hours — codegen should
 *  fail fast so the queue doesn't block on a hanging API call. */
const CODEGEN_CALL_TIMEOUT_MS = 5 * 60 * 1000;

export async function generateTrainingCode(
    spec: ExperimentSpec,
    artifactDir: string,
    previousAttempts?: Array<{ code: string; error: string }>,
    options?: { signal?: AbortSignal },
): Promise<CodegenResult> {
    const cfg = getConfig();
    const maxCodegenAttempts = (cfg as any).execution?.retryLimit ?? 2;
    let attempts = previousAttempts ? [...previousAttempts] : [];

    for (let attempt = 0; attempt <= maxCodegenAttempts; attempt++) {
        const prompt = buildPrompt(spec, artifactDir, attempts.length > 0 ? attempts : undefined);

        // Tight timeout for codegen LLM calls — prevents one hanging API call
        // from blocking the entire queue for hours. Combined with the job-level
        // signal so cancellation also works.
        const codegenTimeout = AbortSignal.timeout(CODEGEN_CALL_TIMEOUT_MS);
        const codegenSignal = options?.signal
            ? AbortSignal.any([options.signal, codegenTimeout])
            : codegenTimeout;

        const rawResponse = await callLlm(prompt, {
            role: 'codegen',
            jsonSchema: { name: 'codegen', schema: {} },
            signal: codegenSignal,
        });

        const generatedCode = extractCode(rawResponse);

        if (!generatedCode || generatedCode.trim().length < 20) {
            throw new LabError(
                `Codegen produced empty or trivial code. Raw response (first 500 chars): ${rawResponse.slice(0, 500)}`,
                'llm', true,
            );
        }

        // Thinking leakage detection: catch models that embed reasoning in code
        const leakage = detectThinkingLeakage(generatedCode);
        if (leakage) {
            console.error(`[nn-lab] Thinking leakage detected (attempt ${attempt + 1}/${maxCodegenAttempts + 1}): ${leakage}`);
            attempts.push({
                code: generatedCode,
                error: `EXECUTOR ERROR: ${leakage}\n\nYour code contained reasoning text, HTML fragments, or self-corrections mixed into the Python source. Return ONLY clean, syntactically valid Python in the "code" field. Do not embed your thinking process.`,
            });
            if (attempt < maxCodegenAttempts) continue;
            throw new LabError(
                `Codegen failed all ${maxCodegenAttempts + 1} attempts due to thinking leakage. Last: ${leakage}`,
                'llm', true,
            );
        }

        // Preflight: dry run with 2-batch DataLoaders catches runtime errors in seconds
        const preflightError = await preflightValidate(generatedCode, artifactDir);
        if (preflightError) {
            console.error(`[nn-lab] Preflight failed (attempt ${attempt + 1}/${maxCodegenAttempts + 1}): ${preflightError}`);
            attempts.push({ code: generatedCode, error: preflightError });
            if (attempt < maxCodegenAttempts) continue; // retry with error feedback
            // Final attempt also failed — do NOT submit broken code for full execution.
            // It wastes GPU hours and produces confusing "incomplete" results.
            throw new LabError(
                `Codegen failed all ${maxCodegenAttempts + 1} preflight attempts. Last error: ${preflightError}`,
                'sandbox', true,
            );
        }

        const code = wrapScript(generatedCode, artifactDir);
        return { code, prompt, rawResponse };
    }

    // Should not reach here, but satisfy TS
    throw new LabError('Codegen failed all preflight attempts', 'llm', true);
}
