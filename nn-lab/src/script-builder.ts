/**
 * NN-Lab Script Builder — builds a deterministic Python training script from a spec.
 *
 * No LLM involved. The spec is serialized as JSON, and the Python runner
 * reads it and executes fixed-menu training code.
 */

import { join } from 'path';
import type { ExperimentSpec, CodegenResult } from '@lab/core';

// Parent of the runner package — so `from runner import ...` resolves
const RUNNER_PARENT = process.cwd().replace(/\\/g, '/');

export function buildScript(spec: ExperimentSpec, artifactDir: string): CodegenResult {
    const specJson = JSON.stringify(spec.setup, null, 2);

    // The executor.py wraps whatever `result` is set to inside
    // {"result": <result>, "error": null, "execution_time_ms": ...}
    // We unwrap run_experiment's return so `result` is either the structured
    // data directly (conditions/comparison/meta) or an error marker.
    const code = `#!/usr/bin/env python3
"""NN-Lab training script — deterministic, built from experiment spec."""

import sys
import os
import json

# Add nn-lab root to path so 'from runner import ...' resolves
sys.path.insert(0, ${JSON.stringify(RUNNER_PARENT)})

from runner import run_experiment

spec = json.loads(${JSON.stringify(specJson)})
artifact_dir = ${JSON.stringify(artifactDir.replace(/\\/g, '/'))}

_outcome = run_experiment(spec, artifact_dir)
if _outcome.get("error"):
    result = {"_error": _outcome["error"]}
else:
    result = _outcome["result"]
`;

    return {
        code,
        prompt: `Deterministic script for spec: ${spec.specType}`,
        rawResponse: 'deterministic',
    };
}
