/**
 * Core types shared across all lab implementations.
 */

// =============================================================================
// EXPERIMENT & RESULTS
// =============================================================================

export interface ExperimentSpec {
    specType: string;
    hypothesis: string;
    setup: Record<string, any>;
    nodeId: string;
    claimType?: string;
    hints?: Record<string, any>;
}

export interface SandboxResult {
    success: boolean;
    stdout: string;
    stderr: string;
    exitCode: number;
    executionTimeMs: number;
    killed: boolean;
    parsedOutput?: any;
}

export interface LabVerdict {
    verdict: 'supported' | 'refuted' | 'inconclusive' | 'not_testable' | 'error';
    confidence: number;
    hypothesis?: string;
    testCategory?: string;
    /** Plain prose explanation — short, render-as-text. */
    details?: string;
    /**
     * Optional structured payload accompanying the verdict.
     *
     * Use this for rich data the host should render as fields, lists, or tables —
     * not as a stringified blob. Critique-lab puts its action / corrected verdict /
     * issues / guidance / rewritten claim here. Math-lab can put computed
     * measurements here. Anything that's "more than prose" goes here, NOT inside
     * `details` as escaped JSON.
     */
    structuredDetails?: Record<string, unknown>;
    evalPrompt?: string;
    evalRawResponse?: string;
}

export interface Artifact {
    label: string;
    type: string;
    url: string;
    sizeBytes?: number;
    category?: string;
}

export interface LabResult {
    verdict: LabVerdict['verdict'];
    confidence: number;
    hypothesis?: string;
    testCategory?: string;
    details?: string;
    artifacts: Artifact[];
    executionTimeMs?: number;
    error?: string;
}

// =============================================================================
// PIPELINE — the abstraction labs implement
// =============================================================================

export interface CodegenResult {
    code: string;
    prompt: string;
    rawResponse: string;
}

export interface PipelineContext {
    jobId: string;
    spec: ExperimentSpec;
    artifactDir: string;
    signal: AbortSignal;
}

export interface LabPipeline {
    /** Generate executable code from a spec. Called with retry context. */
    generate(
        ctx: PipelineContext,
        previousAttempts?: Array<{ code: string; error: string }>,
    ): Promise<CodegenResult>;

    /** Execute the generated code. */
    execute(ctx: PipelineContext, code: string): Promise<SandboxResult>;

    /** Evaluate execution results against the spec. Returns a verdict. */
    evaluate(ctx: PipelineContext, sandbox: SandboxResult): Promise<LabVerdict>;
}

// =============================================================================
// SERVER FACTORY OPTIONS
// =============================================================================

export interface LabServerOptions<TConfig extends BaseLabConfig = BaseLabConfig> {
    name: string;
    version: string;
    description?: string;
    pipeline: LabPipeline;
    config: TConfig;
    /** Build the /capabilities response body */
    buildCapabilities(config: TConfig): Record<string, any>;
    /** Validate an incoming spec. Return error string if invalid, null if ok. */
    validateSpec?(spec: ExperimentSpec, config: TConfig): string | null;
}

// =============================================================================
// CONFIG TYPES
// =============================================================================

export interface SandboxConfig {
    pythonPath: string;
    executionTimeoutMs: number;
    maxOutputBytes: number;
    maxCodeLength: number;
    networkKillSwitch: boolean;
}

export interface ExecutionConfig {
    retryLimit: number;
    artifactTtlSeconds: number;
    maxConcurrent: number;
    jobTimeoutMs: number;
    maxArtifactBytes: number;
    resourceLock: boolean;
}

export type LlmProvider = 'openai-compatible' | 'anthropic' | 'openai';

export interface ModelSlotConfig {
    provider: LlmProvider;
    endpoint: string;
    model: string;
    apiKey: string;
    maxTokens: number;
    temperature: number;
    timeoutMs: number;
    systemPrompt?: string;
    mergeSystemPrompt?: boolean;
    stripThinking?: boolean;
    noThink?: boolean;
}

export interface ModelsConfig {
    codegen: ModelSlotConfig;
    evaluation?: ModelSlotConfig;
    fallback?: ModelSlotConfig;
}

export interface AuthConfig {
    token?: string;
    tokens?: string[];
}

export interface DatabaseConfig {
    path?: string;
}

/** Podbit LLM proxy config — routes LLM calls through Podbit's subsystem model routing */
export interface PodbitProxyConfig {
    /** Podbit server URL (e.g., "http://localhost:3000") */
    url: string;
    /** Subsystem name (e.g., "lab:math-lab") — auto-detected from lab ID if not set */
    subsystem?: string;
}

export interface BaseLabConfig {
    port: number;
    auth: AuthConfig;
    sandbox: SandboxConfig;
    execution: ExecutionConfig;
    database?: DatabaseConfig;
    models: ModelsConfig;
    /** Route LLM calls through Podbit instead of direct API calls (local labs only) */
    podbit?: PodbitProxyConfig;
    /** Lab-specific capabilities — each lab defines its own shape */
    capabilities: Record<string, any>;
    [key: string]: any;
}
