/**
 * NN-Lab — neural network training lab for Podbit verification pipeline.
 *
 * Declarative-in / verdict-out. Trains real networks from fixed menus or
 * LLM-generated code, then uses an evaluation LLM to interpret the results
 * against the hypothesis.
 *
 * Maps onto the lab-core LabPipeline contract:
 *   - generate: deterministic script builder OR LLM codegen
 *   - execute:  Python sandbox with PyTorch
 *   - evaluate: pre-check errors, then core LLM evaluator interprets results
 */

import { join } from 'path';
import { readFileSync } from 'fs';
import {
    createLabServer, loadConfig, executePython, db,
    evaluate as coreEvaluate,
    type LabPipeline, type PipelineContext, type BaseLabConfig,
    type LabVerdict, type SandboxResult,
} from '@lab/core';
import { validateNNSpec } from './spec-validator.js';
import { buildScript } from './script-builder.js';
import { generateTrainingCode } from './codegen.js';

// =============================================================================
// CONFIG
// =============================================================================

interface NNLabConfig extends BaseLabConfig {
    capabilities: {
        specTypes: Record<string, string>;
        queueLimit: number;
        artifactTtlSeconds: number;
        features: string[];
        architectures: string[];
        datasets: string[];
        optimizers: string[];
        lr_schedules: string[];
        measurements: string[];
        comparison_measurements: string[];
        sparsity_methods: string[];
        modifications: string[];
        max_runs: number;
        default_runs: number;
        [key: string]: any;
    };
    execution: BaseLabConfig['execution'] & {
        defaultEpochCeiling?: { real: number; synthetic: number };
    };
}

const config = loadConfig<NNLabConfig>({
    capabilities: {
        specTypes: {
            nn_training_profile: 'Train a single configuration and collect measurements. Prefer synthetic datasets to isolate the mechanism under test. Required setup: architecture{model_type(mlp|cnn|cnn_depthwise_separable|resnet|transformer_tiny), depth(1-50), width(1-4096), optional modifications[bottleneck|dropout|weight_decay|batch_norm|layer_norm]}, training{dataset(mnist|cifar10|fashion_mnist|synthetic_quadratic|synthetic_regression), optimizer(sgd|adam|rmsprop|lbfgs), epochs(1-200 synthetic/1-50 real), optional lr_schedule(constant|cosine|cyclical|step), optional batch_size(1-4096), optional runs(1-10, default 3)}, measurements[(loss_curve|convergence_epoch|final_accuracy|sparsity_by_layer|gradient_stats_by_layer|effective_lr_by_layer|weight_norm_by_layer|hessian_top_eigenvalue|wall_clock_time)].',
            nn_paired_comparison: 'Train 2+ configurations with matched seeds and compare. Prefer synthetic datasets for clean controlled comparisons. Required setup: architecture{model_type, depth, width} (same as nn_training_profile), training{dataset, optimizer, epochs}, measurements[], PLUS conditions[2-10 items, each with label(string) and optional architecture/training overrides], optional comparison_measurements[(convergence_speed_ratio|final_accuracy_difference|sparsity_distribution_divergence|gradient_stat_divergence)]. The base architecture/training define defaults; each condition overrides specific fields.',
            nn_sparsity_profile: 'Train with sparsity and measure per-layer distribution and accuracy impact. Same required setup as nn_training_profile, PLUS architecture.sparsity{ratio(0-0.99), method(unstructured|structured_2_4|magnitude)} is required.',
            nn_optimizer_dynamics: 'Measure optimizer behaviour on synthetic loss surfaces. Same required setup as nn_training_profile. Prefer synthetic datasets (synthetic_quadratic, synthetic_regression).',
            nn_architecture_scaling: 'Measure how metrics change with depth/width variations. Same required setup as nn_training_profile, PLUS conditions[2-10 items with label and architecture/training overrides] required (same as nn_paired_comparison).',
            nn_codegen: 'Generate custom PyTorch training code via LLM. For experiments beyond fixed menus - custom architectures, activations, loss functions, optimizers, training procedures, or novel measurements. Required: hypothesis(string, min 10 chars). Optional setup.constraints{max_epochs(1-200), max_runs(1-10)}. No fixed architecture/training/measurements schema - the LLM writes arbitrary code.',
        },
        queueLimit: 10,
        artifactTtlSeconds: 604800,
        features: ['deterministic-codegen', 'llm-codegen', 'gpu-training', 'chart-generation'],
        architectures: ['mlp', 'cnn', 'cnn_depthwise_separable', 'resnet', 'transformer_tiny'],
        datasets: ['mnist', 'cifar10', 'fashion_mnist', 'synthetic_quadratic', 'synthetic_regression'],
        optimizers: ['sgd', 'adam', 'rmsprop', 'lbfgs'],
        lr_schedules: ['constant', 'cosine', 'cyclical', 'step'],
        measurements: [
            'loss_curve', 'convergence_epoch', 'final_accuracy',
            'sparsity_by_layer', 'gradient_stats_by_layer', 'effective_lr_by_layer',
            'weight_norm_by_layer', 'hessian_top_eigenvalue', 'wall_clock_time',
        ],
        comparison_measurements: [
            'convergence_speed_ratio', 'final_accuracy_difference',
            'sparsity_distribution_divergence', 'gradient_stat_divergence',
        ],
        sparsity_methods: ['unstructured', 'structured_2_4', 'magnitude'],
        modifications: ['bottleneck', 'dropout', 'weight_decay', 'batch_norm', 'layer_norm'],
        max_runs: 10,
        default_runs: 3,
    },
});

// =============================================================================
// EVALUATE — pre-check for errors, then delegate to core LLM evaluator
// =============================================================================

/**
 * Check for execution-level errors before sending to the LLM evaluator.
 * Returns a verdict directly for errors/hardware issues, or null to signal
 * that the core evaluator should interpret the results.
 */
function preCheckResult(_spec: any, sandbox: SandboxResult): LabVerdict | null {
    if (!sandbox.success || sandbox.killed) {
        const stderr = sandbox.stderr || '';
        if (stderr.includes('not testable on current hardware') || stderr.includes('GPU required')) {
            return { verdict: 'not_testable', confidence: 0, details: stderr.slice(0, 500) };
        }
        if (stderr.includes('not supported')) {
            return { verdict: 'not_testable', confidence: 0, details: stderr.slice(0, 500) };
        }
        return { verdict: 'error', confidence: 0, details: stderr?.slice(0, 500) || 'Execution failed' };
    }

    if (!sandbox.parsedOutput) {
        return { verdict: 'error', confidence: 0, details: 'No structured output from training run' };
    }

    const executorOutput = sandbox.parsedOutput;

    // Executor-level error (syntax error, no result variable, etc.)
    if (executorOutput.error) {
        return { verdict: 'error', confidence: 0, details: executorOutput.error };
    }

    const data = executorOutput.result;
    if (!data) {
        return { verdict: 'error', confidence: 0, details: 'No result data from training run' };
    }

    // Runner-level error (GPU unavailable, unsupported config, etc.)
    if (data._error) {
        const err: string = data._error;
        if (err.includes('not testable') || err.includes('not supported')) {
            return { verdict: 'not_testable', confidence: 0, details: err };
        }
        return { verdict: 'error', confidence: 0, details: err };
    }

    // No error — delegate to core evaluator for LLM interpretation
    return null;
}

function applyConvergenceGate(sandbox: SandboxResult, verdict: LabVerdict): LabVerdict {
    const perCondition = sandbox.parsedOutput?.result?.summary?.per_condition;
    if (!perCondition || typeof perCondition !== 'object') return verdict;

    const convergenceStates = Object.values(perCondition)
        .map((condition: any) => condition?.converged)
        .filter((converged): converged is boolean => typeof converged === 'boolean');

    if (convergenceStates.length === 0) return verdict;

    const failedCount = convergenceStates.filter(converged => !converged).length;
    if (failedCount === convergenceStates.length) {
        const note = 'all conditions failed to converge — comparison is between un-trained models';
        return {
            ...verdict,
            verdict: 'inconclusive',
            confidence: 0,
            details: verdict.details ? `${note}. ${verdict.details}` : note,
            structuredDetails: {
                ...verdict.structuredDetails,
                convergenceWarning: note,
            },
        };
    }

    if (failedCount > 0) {
        const note = `${failedCount}/${convergenceStates.length} conditions failed to converge; confidence capped at 0.5`;
        return {
            ...verdict,
            confidence: Math.min(verdict.confidence, 0.5),
            details: verdict.details ? `${verdict.details} [${note}]` : note,
            structuredDetails: {
                ...verdict.structuredDetails,
                convergenceWarning: note,
            },
        };
    }

    return verdict;
}

// =============================================================================
// PIPELINE
// =============================================================================

const pipeline: LabPipeline = {
    async generate(ctx: PipelineContext, previousAttempts?: Array<{ code: string; error: string }>) {
        if (ctx.spec.specType === 'nn_codegen') {
            return generateTrainingCode(ctx.spec, ctx.artifactDir, previousAttempts, { signal: ctx.signal });
        }
        // Fixed-menu specs — deterministic, no LLM
        return buildScript(ctx.spec, ctx.artifactDir);
    },

    async execute(ctx: PipelineContext, code: string) {
        // Cap sandbox timeout to remaining job budget so it can't exceed the job timeout.
        // Leave a buffer for the evaluation LLM call that follows.
        const EVAL_BUFFER_MS = 60_000; // 60s for evaluation LLM
        const elapsed = Date.now() - ctx.jobStartMs;
        const remaining = Math.max(10_000, ctx.jobTimeoutMs - elapsed - EVAL_BUFFER_MS);
        return executePython(code, undefined, ctx.signal, remaining);
    },

    async evaluate(ctx: PipelineContext, sandbox: SandboxResult) {
        // Pre-check for execution errors (hardware, crashes, missing output)
        const errorVerdict = preCheckResult(ctx.spec, sandbox);
        if (errorVerdict) return errorVerdict;

        // Experiment completed — use the core LLM evaluator to interpret
        // results against the hypothesis. Use the job-level signal so
        // evaluation gets the full remaining budget (matching math-lab).
        // Podbit's model registry owns the per-call timeout.
        const verdict = await coreEvaluate(ctx.spec, sandbox, { signal: ctx.signal });
        return applyConvergenceGate(sandbox, verdict);
    },
};

// =============================================================================
// START
// =============================================================================

const lab = createLabServer<NNLabConfig>({
    name: 'nn-lab',
    version: '1.0.0',
    description: 'Neural network training lab for Podbit. Trains real networks from declarative specs, then evaluates results against the hypothesis via LLM to produce supported/refuted/inconclusive verdicts.\n\nIMPORTANT: This is a rapid-experiment lab with strict execution timeouts, not a large-scale training facility. ALWAYS prefer synthetic datasets (synthetic_quadratic, synthetic_regression) or small benchmarks (mnist, fashion_mnist) over larger datasets (cifar10). Synthetic data isolates the mechanism being tested without dataset noise or training time overhead. Only use real datasets when the hypothesis specifically requires natural image statistics or dataset-specific properties.',

    pipeline,
    config,

    buildCapabilities: (cfg) => ({
        specTypes: cfg.capabilities.specTypes,
        environment: {
            runtime: `Python + PyTorch (${cfg.sandbox.pythonPath})`,
            approach: 'Fixed-menu experiments via deterministic script builder, plus LLM codegen for custom experiments beyond the menus.',
            codegenMethod: 'Fixed menus: Spec → JSON → Python runner. Custom (nn_codegen): Spec → LLM → PyTorch script.',
            evaluationMethod: 'LLM evaluator interprets raw measurements against the hypothesis to produce supported/refuted/inconclusive verdicts.',
            limits: {
                maxExecutionSeconds: Math.round(cfg.sandbox.executionTimeoutMs / 1000),
                maxOutputBytes: cfg.sandbox.maxOutputBytes,
                epochCeiling: cfg.execution.defaultEpochCeiling,
                maxRuns: cfg.capabilities.max_runs,
            },
        },
        menus: {
            architectures: cfg.capabilities.architectures,
            datasets: cfg.capabilities.datasets,
            optimizers: cfg.capabilities.optimizers,
            lr_schedules: cfg.capabilities.lr_schedules,
            measurements: cfg.capabilities.measurements,
            comparison_measurements: cfg.capabilities.comparison_measurements,
            sparsity_methods: cfg.capabilities.sparsity_methods,
            modifications: cfg.capabilities.modifications,
        },
        queueLimit: cfg.capabilities.queueLimit,
        artifactTtlSeconds: cfg.capabilities.artifactTtlSeconds,
        features: cfg.capabilities.features,
    }),

    validateSpec: (spec, cfg) => {
        const err = validateNNSpec(spec, cfg);
        if (err && spec.specType !== 'nn_codegen') {
            // Menu validation failed — fall back to codegen instead of rejecting.
            // The hypothesis and setup are still usable; the codegen LLM can work with them.
            console.error(`[nn-lab] Menu validation failed for ${spec.specType}, falling back to nn_codegen: ${err}`);
            spec.specType = 'nn_codegen';
            return validateNNSpec(spec, cfg);
        }
        return err;
    },
});

// =============================================================================
// EXTRA ROUTES — job list + queue UI (not in lab-core)
// =============================================================================

let _queueHtml: string | null = null;
lab.app.get('/ui', (_req, res) => {
    if (!_queueHtml) _queueHtml = readFileSync(join(process.cwd(), 'public', 'queue.html'), 'utf-8');
    res.setHeader('Content-Type', 'text/html');
    res.send(_queueHtml);
});

lab.app.get('/jobs', (_req, res) => {
    const jobs = db.listJobs({ limit: 100 });
    res.json(jobs.map(j => ({
        id: j.id,
        status: j.status,
        priority: j.priority,
        spec: j.spec ? JSON.parse(j.spec) : null,
        created_at: j.created_at,
        started_at: j.started_at,
        completed_at: j.completed_at,
    })));
});

lab.start();
