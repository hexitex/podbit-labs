/**
 * NN-Lab Spec Validator — validates experiment specs against the fixed menus
 * that form the spec-extractor API contract.
 *
 * Fixed menus exist for the deterministic script-builder path only — they
 * give the spec extractor a stable surface to target. They are NOT a
 * limitation on what nn-lab can run: any spec that fails menu validation
 * is rerouted to `nn_codegen` (see index.ts), which lets the LLM write
 * arbitrary PyTorch — custom models, custom losses, custom training loops,
 * custom datasets, custom measurements.
 */

import type { ExperimentSpec, BaseLabConfig } from '@lab/core';

// =============================================================================
// FIXED MENUS
// =============================================================================

const ARCHITECTURES = new Set([
    'mlp', 'cnn', 'cnn_depthwise_separable', 'resnet', 'transformer_tiny',
]);

const DATASETS = new Set([
    'mnist', 'cifar10', 'fashion_mnist', 'synthetic_quadratic', 'synthetic_regression',
]);

const OPTIMIZERS = new Set(['sgd', 'adam', 'rmsprop', 'lbfgs']);

const LR_SCHEDULES = new Set(['constant', 'cosine', 'cyclical', 'step']);

const SPARSITY_METHODS = new Set(['unstructured', 'structured_2_4', 'magnitude']);

const MODIFICATIONS = new Set([
    'bottleneck', 'dropout', 'weight_decay', 'batch_norm', 'layer_norm',
]);

const MEASUREMENTS = new Set([
    'loss_curve', 'convergence_epoch', 'final_accuracy',
    'sparsity_by_layer', 'gradient_stats_by_layer', 'effective_lr_by_layer',
    'weight_norm_by_layer', 'hessian_top_eigenvalue', 'wall_clock_time',
]);

const COMPARISON_MEASUREMENTS = new Set([
    'convergence_speed_ratio', 'final_accuracy_difference',
    'sparsity_distribution_divergence', 'gradient_stat_divergence',
]);

// =============================================================================
// VALIDATION
// =============================================================================

interface NNLabConfig extends BaseLabConfig {
    execution: BaseLabConfig['execution'] & {
        defaultEpochCeiling?: { real: number; synthetic: number };
    };
}

function isSyntheticDataset(dataset: string): boolean {
    return dataset.startsWith('synthetic_');
}

function validateArchitecture(setup: Record<string, any>): string | null {
    const arch = setup.architecture;
    if (!arch) return 'Missing architecture section in setup';
    if (!arch.model_type) return 'Missing architecture.model_type';
    if (!ARCHITECTURES.has(arch.model_type)) {
        return `Unsupported model_type: "${arch.model_type}". Supported: ${[...ARCHITECTURES].join(', ')}`;
    }
    if (typeof arch.depth !== 'number' || arch.depth < 1 || arch.depth > 50) {
        return `architecture.depth must be an integer 1-50, got: ${arch.depth}`;
    }
    if (typeof arch.width !== 'number' || arch.width < 1 || arch.width > 4096) {
        return `architecture.width must be an integer 1-4096, got: ${arch.width}`;
    }
    if (arch.sparsity) {
        if (typeof arch.sparsity.ratio !== 'number' || arch.sparsity.ratio < 0 || arch.sparsity.ratio > 0.99) {
            return `sparsity.ratio must be 0-0.99, got: ${arch.sparsity.ratio}`;
        }
        if (!SPARSITY_METHODS.has(arch.sparsity.method)) {
            return `Unsupported sparsity method: "${arch.sparsity.method}". Supported: ${[...SPARSITY_METHODS].join(', ')}`;
        }
    }
    if (arch.modifications) {
        if (!Array.isArray(arch.modifications)) return 'architecture.modifications must be an array';
        for (const mod of arch.modifications) {
            const name = typeof mod === 'string' ? mod : mod?.name;
            if (!name || !MODIFICATIONS.has(name)) {
                return `Unsupported modification: "${name}". Supported: ${[...MODIFICATIONS].join(', ')}`;
            }
        }
    }
    return null;
}

function validateTraining(setup: Record<string, any>, cfg: NNLabConfig): string | null {
    const tr = setup.training;
    if (!tr) return 'Missing training section in setup';
    if (!tr.dataset) return 'Missing training.dataset';

    const dsName = typeof tr.dataset === 'string' ? tr.dataset : tr.dataset?.name;
    if (!dsName || !DATASETS.has(dsName)) {
        return `Unsupported dataset: "${dsName}". Supported: ${[...DATASETS].join(', ')}`;
    }

    if (!tr.optimizer) return 'Missing training.optimizer';
    const optName = typeof tr.optimizer === 'string' ? tr.optimizer : tr.optimizer?.name;
    if (!optName || !OPTIMIZERS.has(optName)) {
        return `Unsupported optimizer: "${optName}". Supported: ${[...OPTIMIZERS].join(', ')}`;
    }

    if (tr.lr_schedule) {
        const sName = typeof tr.lr_schedule === 'string' ? tr.lr_schedule : tr.lr_schedule?.name;
        if (sName && !LR_SCHEDULES.has(sName)) {
            return `Unsupported lr_schedule: "${sName}". Supported: ${[...LR_SCHEDULES].join(', ')}`;
        }
    }

    const ceiling = isSyntheticDataset(dsName)
        ? (cfg.execution.defaultEpochCeiling?.synthetic ?? 200)
        : (cfg.execution.defaultEpochCeiling?.real ?? 50);

    if (typeof tr.epochs !== 'number' || tr.epochs < 1 || tr.epochs > ceiling) {
        return `training.epochs must be 1-${ceiling} for ${dsName}, got: ${tr.epochs}`;
    }

    if (tr.batch_size !== undefined) {
        if (typeof tr.batch_size !== 'number' || tr.batch_size < 1 || tr.batch_size > 4096) {
            return `training.batch_size must be 1-4096, got: ${tr.batch_size}`;
        }
    }

    const runs = tr.runs ?? 3;
    if (typeof runs !== 'number' || runs < 1 || runs > 10) {
        return `training.runs must be 1-10, got: ${runs}`;
    }

    return null;
}

function validateMeasurements(setup: Record<string, any>): string | null {
    const ms = setup.measurements;
    if (!ms || !Array.isArray(ms) || ms.length === 0) {
        return 'Missing or empty measurements array in setup';
    }
    for (const m of ms) {
        const name = typeof m === 'string' ? m : m?.name;
        if (!name || !MEASUREMENTS.has(name)) {
            return `Unsupported measurement: "${name}". Supported: ${[...MEASUREMENTS].join(', ')}`;
        }
    }
    return null;
}

function validateConditions(setup: Record<string, any>, cfg: NNLabConfig): string | null {
    if (!setup.conditions) return null; // single-condition experiment
    if (!Array.isArray(setup.conditions) || setup.conditions.length < 2) {
        return 'conditions must be an array of 2+ configurations';
    }
    if (setup.conditions.length > 10) {
        return `Too many conditions (max 10, got ${setup.conditions.length})`;
    }
    for (let i = 0; i < setup.conditions.length; i++) {
        const cond = setup.conditions[i];
        if (!cond.label || typeof cond.label !== 'string') {
            return `conditions[${i}] missing label`;
        }
        // Validate overrides — only known fields allowed
        if (cond.architecture) {
            const merged = { ...setup.architecture, ...cond.architecture };
            const err = validateArchitecture({ architecture: merged });
            if (err) return `conditions[${i}]: ${err}`;
        }
        if (cond.training) {
            const merged = { ...setup.training, ...cond.training };
            const err = validateTraining({ training: merged }, cfg);
            if (err) return `conditions[${i}]: ${err}`;
        }
    }

    if (setup.comparison_measurements) {
        if (!Array.isArray(setup.comparison_measurements)) {
            return 'comparison_measurements must be an array';
        }
        for (const m of setup.comparison_measurements) {
            const name = typeof m === 'string' ? m : m?.name;
            if (!name || !COMPARISON_MEASUREMENTS.has(name)) {
                return `Unsupported comparison_measurement: "${name}". Supported: ${[...COMPARISON_MEASUREMENTS].join(', ')}`;
            }
        }
    }

    return null;
}

// =============================================================================
// PUBLIC
// =============================================================================

// Spec types that require conditions (multi-config experiments)
const MULTI_CONDITION_TYPES = new Set([
    'nn_paired_comparison', 'nn_architecture_scaling',
]);

// Spec types that require sparsity configuration
const SPARSITY_REQUIRED_TYPES = new Set(['nn_sparsity_profile']);

// Spec types that should prefer synthetic datasets
const SYNTHETIC_PREFERRED_TYPES = new Set(['nn_optimizer_dynamics']);

const ALL_SPEC_TYPES = new Set([
    'nn_training_profile', 'nn_paired_comparison', 'nn_sparsity_profile',
    'nn_optimizer_dynamics', 'nn_architecture_scaling', 'nn_codegen',
]);

function validateCodegenSpec(spec: ExperimentSpec): string | null {
    const { setup } = spec;
    if (!setup || typeof setup !== 'object') return 'Missing or invalid setup';
    if (!spec.hypothesis || typeof spec.hypothesis !== 'string' || spec.hypothesis.length < 10) {
        return 'nn_codegen requires a hypothesis (min 10 chars)';
    }
    const constraints = setup.constraints || {};
    if (constraints.max_epochs !== undefined) {
        if (typeof constraints.max_epochs !== 'number' || constraints.max_epochs < 1 || constraints.max_epochs > 200) {
            return `constraints.max_epochs must be 1-200, got: ${constraints.max_epochs}`;
        }
    }
    if (constraints.max_runs !== undefined) {
        if (typeof constraints.max_runs !== 'number' || constraints.max_runs < 1 || constraints.max_runs > 10) {
            return `constraints.max_runs must be 1-10, got: ${constraints.max_runs}`;
        }
    }
    return null;
}

export function validateNNSpec(spec: ExperimentSpec, cfg: BaseLabConfig): string | null {
    const nnCfg = cfg as NNLabConfig;

    if (!ALL_SPEC_TYPES.has(spec.specType)) {
        return `Unsupported specType: "${spec.specType}". Supported: ${[...ALL_SPEC_TYPES].join(', ')}`;
    }

    // Codegen specs have minimal validation — the LLM handles the rest
    if (spec.specType === 'nn_codegen') {
        return validateCodegenSpec(spec);
    }

    const { setup } = spec;
    if (!setup || typeof setup !== 'object') return 'Missing or invalid setup';

    let err = validateArchitecture(setup);
    if (err) return err;

    err = validateTraining(setup, nnCfg);
    if (err) return err;

    err = validateMeasurements(setup);
    if (err) return err;

    // Multi-condition types require conditions array
    if (MULTI_CONDITION_TYPES.has(spec.specType)) {
        if (!setup.conditions) return `specType "${spec.specType}" requires conditions array`;
        err = validateConditions(setup, nnCfg);
        if (err) return err;
    }

    // Sparsity types require sparsity config in architecture
    if (SPARSITY_REQUIRED_TYPES.has(spec.specType)) {
        if (!setup.architecture?.sparsity) {
            return `specType "${spec.specType}" requires architecture.sparsity configuration`;
        }
    }

    // Warn-level: optimizer dynamics types should use synthetic datasets
    // (not a hard error, just noted — the spec extractor context prompt guides this)

    // Optional conditions on non-required types are still valid (e.g., nn_sparsity_profile with conditions)
    if (setup.conditions && !MULTI_CONDITION_TYPES.has(spec.specType)) {
        err = validateConditions(setup, nnCfg);
        if (err) return err;
    }

    return null;
}
