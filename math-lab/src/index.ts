/**
 * Math Lab — computational verification server for mathematical claims.
 *
 * Thin wrapper around @lab/core that provides:
 *   - Math-specific codegen (template + LLM prompt)
 *   - Math-specific capabilities and spec validation
 */

import { join } from 'path';
import { readFileSync } from 'fs';
import {
    createLabServer, loadConfig, executePython, evaluate, db,
    type LabPipeline, type PipelineContext, type BaseLabConfig,
} from '@lab/core';
import { generateCode } from './codegen.js';

// =============================================================================
// CONFIG
// =============================================================================

interface MathLabConfig extends BaseLabConfig {
    capabilities: {
        specTypes: string[] | Record<string, string>;
        queueLimit: number;
        artifactTtlSeconds: number;
        features: string[];
        [key: string]: any;
    };
}

const config = loadConfig<MathLabConfig>({
    capabilities: {
        specTypes: ['math', 'parameter_sweep', 'convergence_analysis', 'curve_shape', 'quantum_model', 'wave_system', 'coupled_dynamics', 'many_body_model'],
        queueLimit: 20,
        artifactTtlSeconds: 604800,
        features: ['codegen', 'sandbox', 'llm-eval'],
    },
});

// =============================================================================
// PIPELINE
// =============================================================================

const pipeline: LabPipeline = {
    async generate(ctx: PipelineContext, previousAttempts) {
        return generateCode(ctx.spec, previousAttempts, { signal: ctx.signal });
    },

    async execute(_ctx: PipelineContext, code: string) {
        return executePython(code);
    },

    async evaluate(ctx: PipelineContext, sandbox) {
        return evaluate(ctx.spec, sandbox, { signal: ctx.signal });
    },
};

// =============================================================================
// START
// =============================================================================

const lab = createLabServer<MathLabConfig>({
    name: 'math-lab',
    version: '1.0.0',
    description: 'Computational verification lab for mathematical claims. Generates Python from experiment specs, executes in sandbox, evaluates via numerical comparison or LLM.',

    pipeline,
    config,

    buildCapabilities: (cfg) => ({
        specTypes: cfg.capabilities.specTypes,
        environment: {
            runtime: `Python sandbox (${cfg.sandbox.pythonPath})`,
            approach: 'Crafted template with LLM-generated computation. Network kill switch for isolation.',
            codegenMethod: 'LLM generates computation code, wrapped in pre-built template with all imports and utilities',
            evaluationMethod: 'Direct verdict from code, or LLM interpretation fallback',
            limits: {
                maxExecutionSeconds: Math.round(cfg.sandbox.executionTimeoutMs / 1000),
                maxOutputBytes: cfg.sandbox.maxOutputBytes,
                networkAccess: !cfg.sandbox.networkKillSwitch,
            },
        },
        queueLimit: cfg.capabilities.queueLimit,
        artifactTtlSeconds: cfg.capabilities.artifactTtlSeconds,
        features: cfg.capabilities.features,
    }),

    validateSpec: (spec, cfg) => {
        const specTypes = cfg.capabilities.specTypes;
        const supported = Array.isArray(specTypes) ? specTypes : Object.keys(specTypes);
        if (!supported.includes(spec.specType)) {
            return `Unsupported specType: "${spec.specType}". Supported: ${supported.join(', ')}`;
        }
        return null;
    },
});

// Queue UI + job list
let _queueHtml: string | null = null;
lab.app.get('/ui', (_req, res) => {
    if (!_queueHtml) _queueHtml = readFileSync(join(process.cwd(), 'public', 'queue.html'), 'utf-8');
    res.setHeader('Content-Type', 'text/html');
    res.send(_queueHtml);
});
lab.app.get('/jobs', (_req, res) => {
    const jobs = db.listJobs({ limit: 100 });
    res.json(jobs.map(j => ({
        id: j.id, status: j.status, priority: j.priority,
        spec: j.spec ? JSON.parse(j.spec) : null,
        created_at: j.created_at, started_at: j.started_at, completed_at: j.completed_at,
    })));
});

lab.start();
