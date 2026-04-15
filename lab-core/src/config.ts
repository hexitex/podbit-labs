/**
 * Generic lab config loader.
 * Reads config.json from cwd with env var overrides. Labs extend BaseLabConfig.
 *
 * On startup we ALSO load a `.env` file from the lab's working directory (if present)
 * into process.env. This lets each lab specify PODBIT_URL, PORT, and model API keys
 * in a per-lab .env without hard-coding them in config.json.
 */

import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import type { BaseLabConfig, ModelSlotConfig, LlmProvider } from './types.js';

let _config: BaseLabConfig | null = null;
let _envLoaded = false;

/**
 * Tiny .env loader (no external dependency). Reads a file of `KEY=value` lines,
 * skipping `#` comments and blank lines. Variables already present in process.env
 * are NOT overridden — actual environment wins over the file.
 *
 * Supports surrounding double or single quotes on values: `KEY="value with spaces"`.
 * Does NOT support multi-line values, command substitution, or variable interpolation.
 */
function loadEnvFile(filePath: string): void {
    if (!existsSync(filePath)) return;
    try {
        const text = readFileSync(filePath, 'utf-8');
        for (const rawLine of text.split(/\r?\n/)) {
            const line = rawLine.trim();
            if (!line || line.startsWith('#')) continue;
            const eq = line.indexOf('=');
            if (eq < 1) continue;
            const key = line.slice(0, eq).trim();
            let value = line.slice(eq + 1).trim();
            // Strip surrounding quotes
            if ((value.startsWith('"') && value.endsWith('"')) ||
                (value.startsWith("'") && value.endsWith("'"))) {
                value = value.slice(1, -1);
            }
            if (!(key in process.env)) {
                process.env[key] = value;
            }
        }
    } catch { /* non-fatal — fall through to whatever is in process.env */ }
}

export function loadModelSlot(fileSlot: any, envPrefix: string, defaults: Partial<ModelSlotConfig>): ModelSlotConfig {
    return {
        provider: (fileSlot?.provider || defaults.provider || 'openai-compatible') as LlmProvider,
        endpoint: process.env[`${envPrefix}_ENDPOINT`] || fileSlot?.endpoint || defaults.endpoint || 'http://localhost:11434/v1',
        model: process.env[`${envPrefix}_MODEL`] || fileSlot?.model || defaults.model || 'qwen2.5-coder:14b',
        apiKey: process.env[`${envPrefix}_API_KEY`] || fileSlot?.apiKey || defaults.apiKey || 'ollama',
        maxTokens: parseInt(process.env[`${envPrefix}_MAX_TOKENS`] || '', 10) || fileSlot?.maxTokens || defaults.maxTokens || undefined,
        temperature: parseFloat(process.env[`${envPrefix}_TEMPERATURE`] || '') || (fileSlot?.temperature ?? (defaults.temperature ?? 0.15)),
        timeoutMs: fileSlot?.timeoutMs || defaults.timeoutMs || 0, // 0 = use job remaining time
        systemPrompt: fileSlot?.systemPrompt || defaults.systemPrompt || undefined,
        mergeSystemPrompt: fileSlot?.mergeSystemPrompt ?? defaults.mergeSystemPrompt ?? false,
        stripThinking: fileSlot?.stripThinking ?? defaults.stripThinking ?? false,
        noThink: fileSlot?.noThink ?? defaults.noThink ?? false,
    };
}

/**
 * Load config from config.json in the given directory (defaults to cwd).
 * Labs call this with their own defaults for the capabilities section.
 */
export function loadConfig<T extends BaseLabConfig>(labDefaults: Partial<T>, configDir?: string): T {
    const dir = configDir || process.cwd();

    // Load .env from the lab's working directory ONCE per process. Real env wins over file.
    if (!_envLoaded) {
        loadEnvFile(join(dir, '.env'));
        _envLoaded = true;
    }

    let f: any = {};
    try {
        f = JSON.parse(readFileSync(join(dir, 'config.json'), 'utf-8'));
    } catch { /* use defaults */ }

    const envTokens = process.env.LAB_AUTH_TOKENS?.split(',').map(t => t.trim()).filter(Boolean);
    const codegenSlot = loadModelSlot(f.models?.codegen, 'LLM', {});

    const base: BaseLabConfig = {
        port: parseInt(process.env.PORT || '', 10) || f.port || 3580,
        auth: {
            token: process.env.LAB_AUTH_TOKEN || f.auth?.token || undefined,
            tokens: envTokens || f.auth?.tokens || undefined,
        },
        sandbox: {
            pythonPath: process.env.PYTHON_PATH || f.sandbox?.pythonPath || 'python',
            executionTimeoutMs: f.sandbox?.executionTimeoutMs || 120000,
            maxOutputBytes: f.sandbox?.maxOutputBytes || 1048576,
            maxCodeLength: f.sandbox?.maxCodeLength || 50000,
            networkKillSwitch: f.sandbox?.networkKillSwitch ?? true,
        },
        execution: {
            retryLimit: f.execution?.retryLimit ?? 2,
            artifactTtlSeconds: f.execution?.artifactTtlSeconds ?? 604800,
            maxConcurrent: f.execution?.maxConcurrent ?? 5,
            jobTimeoutMs: f.execution?.jobTimeoutMs ?? 300000,
            maxArtifactBytes: f.execution?.maxArtifactBytes ?? 52428800,
            resourceLock: f.execution?.resourceLock ?? false,
        },
        database: { path: f.database?.path || undefined },
        models: {
            codegen: codegenSlot,
            evaluation: f.models?.evaluation
                ? loadModelSlot(f.models.evaluation, 'EVAL_LLM', { maxTokens: 1500, temperature: 0.1 })
                : undefined,
            fallback: f.models?.fallback
                ? loadModelSlot(f.models.fallback, 'FALLBACK_LLM', {})
                : undefined,
        },
        capabilities: f.capabilities || labDefaults.capabilities || {},
        // Podbit URL resolution order:
        //   1. PODBIT_URL env var (set this when running multiple labs, or when Podbit
        //      is on a non-standard port — single source of truth across all labs)
        //   2. config.json `podbit.url`  (per-lab override)
        // The subsystem ID stays in config.json since it's lab-specific.
        podbit: (process.env.PODBIT_URL || f.podbit?.url)
            ? { url: process.env.PODBIT_URL || f.podbit.url, subsystem: process.env.PODBIT_SUBSYSTEM || f.podbit?.subsystem }
            : undefined,
    };

    // Merge lab-specific defaults
    const config = { ...labDefaults, ...base, capabilities: f.capabilities || labDefaults.capabilities || {} } as T;
    _config = config;
    return config;
}

export function getConfig<T extends BaseLabConfig = BaseLabConfig>(): T {
    if (!_config) throw new Error('Config not loaded — call loadConfig() first');
    return _config as T;
}

export function setConfig<T extends BaseLabConfig>(config: T): void {
    _config = config;
}
