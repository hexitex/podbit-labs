/**
 * Lab Server Factory — creates a fully-wired Express server implementing the Podbit lab contract.
 *
 * Labs call createLabServer() with their pipeline, config, and capabilities builder.
 * The factory handles: HTTP routes, auth, rate limiting, job queue, artifact management,
 * metrics, health, cleanup, graceful shutdown.
 */

import express from 'express';
import rateLimit from 'express-rate-limit';
import { randomUUID, createHash } from 'crypto';
import { existsSync, mkdirSync, readdirSync, rmSync, statSync, writeFileSync } from 'fs';
import { join } from 'path';
import path from 'path';
import archiver from 'archiver';
import * as db from './db.js';
import { JobQueue } from './queue.js';
import { logger } from './logger.js';
import { LabError } from './errors.js';
import type { LabServerOptions, BaseLabConfig, ExperimentSpec, Artifact, LabResult, LabVerdict } from './types.js';

// =============================================================================
// ARTIFACT HELPERS
// =============================================================================

const MIME_MAP: Record<string, string> = {
    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml', '.gif': 'image/gif',
    '.csv': 'text/csv', '.tsv': 'text/tab-separated-values',
    '.json': 'application/json', '.txt': 'text/plain', '.log': 'text/plain',
    '.py': 'text/x-python', '.jl': 'text/x-julia', '.r': 'text/x-r',
    '.pt': 'application/octet-stream', '.pth': 'application/octet-stream',
    '.npy': 'application/octet-stream', '.npz': 'application/octet-stream',
    '.pdf': 'application/pdf', '.html': 'text/html',
};

function getMime(filename: string): string {
    return MIME_MAP[path.extname(filename).toLowerCase()] || 'application/octet-stream';
}

function saveArtifact(jobId: string, artifactDir: string, filename: string, content: string, category: string): void {
    const subDir = path.dirname(filename);
    const fullDir = subDir !== '.' ? join(artifactDir, subDir) : artifactDir;
    if (!existsSync(fullDir)) mkdirSync(fullDir, { recursive: true });
    const filePath = join(artifactDir, filename);
    writeFileSync(filePath, content, 'utf-8');
    const size = Buffer.byteLength(content, 'utf-8');
    const label = path.basename(filename, path.extname(filename));
    db.registerArtifact(jobId, label, filename, getMime(filename), category, size);
}

function collectNewArtifacts(jobId: string, artifactDir: string, maxBytes: number): void {
    if (!existsSync(artifactDir)) return;
    const registered = new Set(db.getJobArtifacts(jobId).map(a => a.filename));
    let totalSize = 0;
    const scanDir = (dir: string, prefix: string) => {
        try {
            for (const entry of readdirSync(dir)) {
                const fullPath = join(dir, entry);
                const relPath = prefix ? `${prefix}/${entry}` : entry;
                const stat = statSync(fullPath);
                if (stat.isDirectory()) { scanDir(fullPath, relPath); }
                else if (stat.isFile()) {
                    totalSize += stat.size;
                    if (totalSize > maxBytes) { logger.warn({ jobId, totalSize, maxBytes }, 'Artifact size limit exceeded'); return; }
                    if (!registered.has(relPath)) {
                        db.registerArtifact(jobId, path.basename(entry, path.extname(entry)), relPath, getMime(entry), 'output', stat.size);
                    }
                }
            }
        } catch { /* non-fatal */ }
    };
    scanDir(artifactDir, '');
}

function buildArtifactList(jobId: string): Artifact[] {
    return db.getJobArtifacts(jobId).map(a => ({
        label: a.label, type: a.type,
        url: `/artifacts/${jobId}/${a.filename}`,
        sizeBytes: a.size_bytes, category: a.category,
    }));
}

// =============================================================================
// FACTORY
// =============================================================================

export function createLabServer<TConfig extends BaseLabConfig>(options: LabServerOptions<TConfig>) {
    const { name, version, description, pipeline, config: cfg, buildCapabilities, validateSpec } = options;

    const ARTIFACTS_DIR = join(process.cwd(), 'artifacts');
    if (!existsSync(ARTIFACTS_DIR)) mkdirSync(ARTIFACTS_DIR, { recursive: true });

    // ── SSE event bus ─────────────────────────────────────────────────────
    const sseClients = new Set<express.Response>();
    function broadcast(event: string, data: Record<string, any>): void {
        const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
        for (const res of sseClients) {
            try { res.write(payload); } catch { sseClients.delete(res); }
        }
    }

    // ── Execution pipeline ────────────────────────────────────────────────
    async function executeSpec(jobId: string, signal: AbortSignal): Promise<void> {
        const jobRow = db.getJob(jobId);
        if (!jobRow) return;

        const spec: ExperimentSpec = JSON.parse(jobRow.spec);
        broadcast('job:started', { jobId, specType: spec.specType });
        const artifactDir = jobRow.artifact_dir;
        const startMs = Date.now();
        const maxAttempts = (cfg.execution.retryLimit ?? 2) + 1;
        const previousAttempts: Array<{ code: string; error: string }> = [];
        const ctx = { jobId, spec, artifactDir, signal };

        try {
            let code = '';
            for (let attempt = 1; attempt <= maxAttempts; attempt++) {
                if (signal.aborted) { finishJob(jobId, { verdict: 'error', confidence: 0, artifacts: buildArtifactList(jobId), executionTimeMs: Date.now() - startMs, error: 'Job cancelled' }); return; }

                // GENERATE
                broadcast('job:stage', { jobId, stage: 'codegen', attempt, maxAttempts });
                try {
                    const result = await pipeline.generate(ctx, previousAttempts.length > 0 ? previousAttempts : undefined);
                    code = result.code;
                    saveArtifact(jobId, artifactDir, `prompts/codegen_attempt_${attempt}.txt`, result.prompt, 'prompt');
                    saveArtifact(jobId, artifactDir, `responses/codegen_raw_${attempt}.txt`, result.rawResponse, 'response');
                    saveArtifact(jobId, artifactDir, `code/experiment_${attempt}.py`, code, 'code');
                } catch (err: any) {
                    saveArtifact(jobId, artifactDir, `errors/codegen_attempt_${attempt}.txt`, err.message, 'log');
                    broadcast('job:stage', { jobId, stage: 'codegen_failed', attempt, maxAttempts, error: err.message.slice(0, 200) });
                    if ((err instanceof LabError && !err.retryable) || attempt === maxAttempts) {
                        finishJob(jobId, { verdict: 'error', confidence: 0, artifacts: buildArtifactList(jobId), executionTimeMs: Date.now() - startMs, error: `Codegen failed: ${err.message}` });
                        return;
                    }
                    continue;
                }

                // EXECUTE
                broadcast('job:stage', { jobId, stage: 'executing', attempt, maxAttempts });
                const sandbox = await pipeline.execute(ctx, code);
                saveArtifact(jobId, artifactDir, 'stdout.txt', sandbox.stdout || '(empty)', 'stdout');
                saveArtifact(jobId, artifactDir, 'stderr.txt', sandbox.stderr || '(empty)', 'stderr');
                saveArtifact(jobId, artifactDir, 'execution_meta.json', JSON.stringify({
                    exitCode: sandbox.exitCode, executionTimeMs: sandbox.executionTimeMs,
                    killed: sandbox.killed, success: sandbox.success, attempt,
                }, null, 2), 'log');

                // executor.py writes tracebacks into parsedOutput.error; some labs only use stderr.
                // Prefer the structured field, fall back to stderr — either way the retry LLM gets a real traceback.
                const parsedError = (sandbox.parsedOutput && typeof sandbox.parsedOutput === 'object')
                    ? (sandbox.parsedOutput as any).error
                    : undefined;
                if (!sandbox.success || !sandbox.parsedOutput || (parsedError && (sandbox.parsedOutput as any).result == null)) {
                    const errorMsg = (typeof parsedError === 'string' && parsedError.length > 0)
                        ? parsedError.slice(0, 2000)
                        : (sandbox.stderr?.slice(0, 500) || 'Execution failed (no stderr)');
                    broadcast('job:stage', { jobId, stage: 'exec_failed', attempt, maxAttempts, exitCode: sandbox.exitCode, error: errorMsg.slice(0, 200) });
                    if (attempt < maxAttempts) { previousAttempts.push({ code, error: errorMsg }); logger.warn({ jobId, attempt, maxAttempts, exitCode: sandbox.exitCode, killed: sandbox.killed, error: errorMsg }, 'Attempt failed'); continue; }
                    finishJob(jobId, { verdict: 'error', confidence: 0, artifacts: buildArtifactList(jobId), executionTimeMs: Date.now() - startMs, error: errorMsg });
                    return;
                }

                // Check if all measurements failed (error objects instead of values)
                const resultData = sandbox.parsedOutput?.result;
                if (resultData && typeof resultData === 'object') {
                    const entries = Object.entries(resultData);
                    const errorCount = entries.filter(([, v]: [string, any]) => v && typeof v === 'object' && v.error).length;
                    if (entries.length > 0 && errorCount === entries.length) {
                        const firstError = (entries[0][1] as any).error;
                        const errorMsg = `All ${errorCount} measurements failed. First: ${firstError}`;
                        if (attempt < maxAttempts) { previousAttempts.push({ code, error: errorMsg }); logger.warn({ jobId, attempt, maxAttempts, error: errorMsg }, 'All measurements failed'); continue; }
                        finishJob(jobId, { verdict: 'error', confidence: 0, artifacts: buildArtifactList(jobId), executionTimeMs: Date.now() - startMs, error: errorMsg });
                        return;
                    }
                }

                collectNewArtifacts(jobId, artifactDir, cfg.execution.maxArtifactBytes);

                // EVALUATE
                broadcast('job:stage', { jobId, stage: 'evaluating', attempt, maxAttempts });
                const verdict = await pipeline.evaluate(ctx, sandbox);
                saveArtifact(jobId, artifactDir, 'verdict.json', JSON.stringify(verdict, null, 2), 'evaluation');

                finishJob(jobId, { ...verdict, artifacts: buildArtifactList(jobId), executionTimeMs: Date.now() - startMs });

                // Webhook
                if ((spec as any).webhookUrl || spec.hints?.webhookUrl) {
                    fireWebhook((spec as any).webhookUrl || spec.hints!.webhookUrl, jobId, verdict).catch(() => {});
                }
                return;
            }

            finishJob(jobId, { verdict: 'error', confidence: 0, artifacts: buildArtifactList(jobId), executionTimeMs: Date.now() - startMs, error: 'All attempts exhausted' });
        } catch (err: any) {
            const isAbort = err.name === 'AbortError' || signal.aborted;
            finishJob(jobId, { verdict: 'error', confidence: 0, artifacts: buildArtifactList(jobId), executionTimeMs: Date.now() - startMs, error: isAbort ? 'Job cancelled or timed out' : err.message });
        }
    }

    function finishJob(jobId: string, result: LabResult): void {
        const status = result.verdict === 'error' ? 'failed' : 'completed';
        db.updateJob(jobId, { status, result: JSON.stringify(result), completed_at: new Date().toISOString() });
        logger.info({ jobId, verdict: result.verdict, confidence: result.confidence, ms: result.executionTimeMs }, 'Job finished');
        broadcast('job:finished', { jobId, status, verdict: result.verdict, confidence: result.confidence, error: result.error ?? null, executionTimeMs: result.executionTimeMs });
    }

    async function fireWebhook(url: string, jobId: string, verdict: LabVerdict): Promise<void> {
        try { await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ jobId, ...verdict }), signal: AbortSignal.timeout(10000) }); }
        catch (err: any) { logger.warn({ jobId, url, error: err.message }, 'Webhook delivery failed'); }
    }

    function findCachedResult(specHash: string): db.JobRow | undefined {
        const recent = db.listJobs({ status: 'completed', limit: 100 });
        for (const job of recent) {
            if (createHash('sha256').update(job.spec).digest('hex') === specHash && job.result) return job;
        }
        return undefined;
    }

    function runCleanup(): void {
        const maxAgeDays = cfg.execution.artifactTtlSeconds / 86400;
        const { count, artifactDirs } = db.pruneOldJobs(maxAgeDays);
        if (count > 0) {
            logger.info({ count }, 'Pruned old jobs');
            for (const dir of artifactDirs) { try { rmSync(dir, { recursive: true, force: true }); } catch { /* best effort */ } }
        }
    }

    // ── Auth middleware ────────────────────────────────────────────────────
    function authMiddleware(req: express.Request, res: express.Response, next: express.NextFunction): void {
        const validTokens: string[] = [];
        if (cfg.auth.tokens?.length) validTokens.push(...cfg.auth.tokens);
        else if (cfg.auth.token) validTokens.push(cfg.auth.token);
        if (validTokens.length === 0) { next(); return; }
        const authHeader = req.headers.authorization;
        if (!authHeader) { res.status(401).json({ error: 'Authorization required' }); return; }
        const provided = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : authHeader;
        if (!validTokens.includes(provided)) { res.status(403).json({ error: 'Invalid token' }); return; }
        next();
    }

    // ── Express app ───────────────────────────────────────────────────────
    const app = express();
    app.use(express.json({ limit: '10mb' }));
    const submitLimiter = rateLimit({ windowMs: 60_000, max: 10, standardHeaders: true, legacyHeaders: false, message: { error: 'Too many submissions, try again later' } });

    // Auth — skip for unauthenticated endpoints
    app.use((req, res, next) => {
        if (['/health', '/', '/capabilities', '/metrics'].includes(req.path)) next();
        else authMiddleware(req, res, next);
    });

    // Health
    app.get('/health', (_req, res) => {
        let dbOk = true;
        try { db.getQueueDepth(); } catch { dbOk = false; }
        res.json({ service: name, status: dbOk ? 'ok' : 'degraded', queueDepth: dbOk ? db.getQueueDepth() : -1, running: dbOk ? db.getRunningCount() : -1, db: dbOk ? 'ok' : 'error', avgExecutionMs: dbOk ? db.getAvgCompletionMs() : -1, uptime: Math.round(process.uptime()) });
    });

    app.get('/', (_req, res) => {
        res.json({ service: name, status: 'ok', queueDepth: db.getQueueDepth(), running: db.getRunningCount() });
    });

    // Capabilities
    app.get('/capabilities', (_req, res) => {
        res.json({ name, version, ...(description ? { description } : {}), ...buildCapabilities(cfg) });
    });

    // Metrics
    app.get('/metrics', (_req, res) => {
        const running = db.getRunningCount();
        const depth = db.getQueueDepth();
        res.setHeader('Content-Type', 'text/plain; version=0.0.4');
        res.send([
            `# HELP lab_jobs_queued Number of queued jobs`, `lab_jobs_queued ${depth - running}`,
            `# HELP lab_jobs_running Number of running jobs`, `lab_jobs_running ${running}`,
            `# HELP lab_jobs_completed_total Total completed jobs`, `lab_jobs_completed_total ${db.getCompletedCount()}`,
            `# HELP lab_jobs_failed_total Total failed jobs`, `lab_jobs_failed_total ${db.getFailedCount()}`,
            `# HELP lab_execution_avg_ms Average execution time`, `lab_execution_avg_ms ${db.getAvgCompletionMs()}`,
            '',
        ].join('\n'));
    });

    // Submit
    app.post('/submit', submitLimiter, (req, res) => {
        const { spec, priority } = req.body;
        if (!spec?.hypothesis) { res.status(400).json({ error: 'Missing spec or spec.hypothesis' }); return; }

        // Lab-specific validation
        if (validateSpec) {
            const err = validateSpec(spec, cfg);
            if (err) { res.status(400).json({ error: err }); return; }
        }

        // Generic validation
        const specJson = JSON.stringify(spec);
        if (specJson.length > 500_000) { res.status(400).json({ error: 'Spec too large (max 500KB)' }); return; }
        if (spec.hypothesis.length > 5000) { res.status(400).json({ error: 'Hypothesis too long (max 5000 chars)' }); return; }
        if (!spec.setup || typeof spec.setup !== 'object') { res.status(400).json({ error: 'Missing or invalid spec.setup' }); return; }

        const queueDepth = db.getQueueDepth();
        if (queueDepth >= (cfg.capabilities.queueLimit ?? 20)) { res.status(429).json({ error: 'Queue full', queueDepth }); return; }

        // Caching
        const specHash = createHash('sha256').update(specJson).digest('hex');
        const cached = findCachedResult(specHash);
        if (cached) {
            logger.info({ specHash }, 'Returning cached result');
            const result = JSON.parse(cached.result!) as LabResult;
            result.artifacts = buildArtifactList(cached.id);
            res.json({ jobId: cached.id, cached: true, ...result });
            return;
        }

        const jobId = randomUUID();
        const artifactDir = join(ARTIFACTS_DIR, jobId);
        mkdirSync(artifactDir, { recursive: true });
        const jobPriority = Math.max(0, Math.min(2, parseInt(priority, 10) || 0));
        db.createJob(jobId, spec, artifactDir, jobPriority);
        saveArtifact(jobId, artifactDir, 'spec.json', JSON.stringify(spec, null, 2), 'log');

        queue.enqueue(jobId);

        const queuePosition = db.getQueuePosition(jobId);
        const avgMs = db.getAvgCompletionMs();
        broadcast('job:queued', { jobId, specType: spec.specType, hypothesis: spec.hypothesis?.slice(0, 100), queuePosition });
        res.status(202).json({ jobId, accepted: true, queuePosition, estimatedCompletionMs: Math.round(queuePosition * avgMs), resourceLock: cfg.execution.resourceLock });
    });

    // Cancel
    const handleCancel = (req: express.Request, res: express.Response) => {
        const job = db.getJob(req.params.jobId);
        if (!job) { res.status(404).json({ error: 'Job not found' }); return; }
        if (job.status === 'completed' || job.status === 'failed') { res.status(409).json({ error: 'Job already finished', status: job.status }); return; }
        const cancelled = queue.cancel(req.params.jobId);
        if (cancelled) res.json({ jobId: req.params.jobId, cancelled: true });
        else res.status(409).json({ error: 'Could not cancel job' });
    };
    app.post('/jobs/:jobId/cancel', handleCancel);
    app.delete('/jobs/:jobId', handleCancel);

    // Job list
    app.get('/jobs', (req, res) => {
        const status = req.query.status as string | undefined;
        const limit = Math.min(parseInt(req.query.limit as string) || 100, 500);
        const offset = parseInt(req.query.offset as string) || 0;
        const jobs = db.listJobs({ status, limit, offset }).map(j => {
            const result = j.result ? JSON.parse(j.result) : null;
            return {
                id: j.id,
                status: j.status,
                spec: j.spec,
                verdict: result?.verdict ?? null,
                confidence: result?.confidence ?? null,
                error: result?.error ?? null,
                executionTimeMs: result?.executionTimeMs ?? null,
                created_at: j.created_at,
                started_at: j.started_at,
                completed_at: j.completed_at,
            };
        });
        res.json(jobs);
    });

    // SSE event stream
    app.get('/events', (req, res) => {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');
        res.flushHeaders();
        res.write(`event: connected\ndata: ${JSON.stringify({ service: name })}\n\n`);
        sseClients.add(res);
        req.on('close', () => { sseClients.delete(res); });
    });

    // Status
    app.get('/status/:jobId', (req, res) => {
        const job = db.getJob(req.params.jobId);
        if (!job) { res.status(404).json({ error: 'Job not found' }); return; }
        const isDone = job.status === 'completed' || job.status === 'failed';
        const queuePosition = isDone ? 0 : db.getQueuePosition(job.id);
        res.json({ status: job.status, progress: isDone ? 1.0 : undefined, queuePosition, estimatedCompletionMs: isDone ? 0 : Math.round(queuePosition * db.getAvgCompletionMs()), startedAt: job.started_at || undefined, artifacts: buildArtifactList(job.id), resourceState: isDone ? 'idle' : db.getRunningCount() > 0 ? 'active' : 'idle' });
    });

    // Result
    app.get('/result/:jobId', (req, res) => {
        const job = db.getJob(req.params.jobId);
        if (!job) { res.status(404).json({ error: 'Job not found' }); return; }
        if (job.status === 'queued' || job.status === 'running') { res.status(202).json({ status: job.status, message: 'Job still running', queuePosition: db.getQueuePosition(job.id) }); return; }
        if (job.result) { const result = JSON.parse(job.result) as LabResult; result.artifacts = buildArtifactList(job.id); res.json(result); }
        else { res.json({ verdict: 'error', confidence: 0, artifacts: buildArtifactList(job.id), error: 'No result available' }); }
    });

    // Artifact download
    app.get('/artifacts/:jobId/:filename(*)', (req, res) => {
        const job = db.getJob(req.params.jobId);
        if (!job) { res.status(404).json({ error: 'Job not found' }); return; }
        const normalized = path.posix.normalize(req.params.filename);
        if (normalized.includes('..') || path.isAbsolute(normalized)) { res.status(403).json({ error: 'Path traversal denied' }); return; }
        const filePath = join(job.artifact_dir, normalized);
        const resolved = path.resolve(filePath);
        const artifactRoot = path.resolve(job.artifact_dir);
        if (!resolved.startsWith(artifactRoot + path.sep) && resolved !== artifactRoot) { res.status(403).json({ error: 'Path traversal denied' }); return; }
        if (!existsSync(filePath)) { res.status(404).json({ error: 'Artifact not found' }); return; }
        res.sendFile(resolved);
    });

    // Artifact archive
    app.get('/artifacts/:jobId', (req, res) => {
        const job = db.getJob(req.params.jobId);
        if (!job) { res.status(404).json({ error: 'Job not found' }); return; }
        if (!existsSync(job.artifact_dir)) { res.status(404).json({ error: 'No artifacts found' }); return; }
        res.setHeader('Content-Type', 'application/zip');
        res.setHeader('Content-Disposition', `attachment; filename="artifacts-${req.params.jobId.slice(0, 8)}.zip"`);
        const archive = archiver('zip', { zlib: { level: 6 } });
        archive.on('error', (err) => { logger.error({ jobId: req.params.jobId, error: err.message }, 'Archive error'); if (!res.headersSent) res.status(500).json({ error: 'Archive failed' }); });
        archive.pipe(res);
        archive.directory(job.artifact_dir, false);
        archive.finalize();
    });

    // ── Lifecycle ─────────────────────────────────────────────────────────
    const queue = new JobQueue(cfg.execution.maxConcurrent, cfg.execution.jobTimeoutMs, executeSpec);
    let cleanupTimer: ReturnType<typeof setInterval>;
    let httpServer: ReturnType<typeof app.listen>;

    function start(): void {
        db.init(cfg.database?.path);
        queue.start();
        cleanupTimer = setInterval(runCleanup, 3600_000);
        httpServer = app.listen(cfg.port, () => {
            logger.info({
                service: name, port: cfg.port, python: cfg.sandbox.pythonPath,
                codegenModel: `${cfg.models.codegen.endpoint} / ${cfg.models.codegen.model}`,
                evalModel: cfg.models.evaluation ? `${cfg.models.evaluation.endpoint} / ${cfg.models.evaluation.model}` : '(using codegen)',
                fallback: cfg.models.fallback ? `${cfg.models.fallback.endpoint} / ${cfg.models.fallback.model}` : '(none)',
                auth: (cfg.auth.tokens?.length || cfg.auth.token) ? 'ENABLED' : 'disabled',
                maxConcurrent: cfg.execution.maxConcurrent, jobTimeoutMs: cfg.execution.jobTimeoutMs,
                queueDepth: db.getQueueDepth(),
            }, `${name} server started`);
        });

        async function gracefulShutdown(signal: string): Promise<void> {
            logger.info({ signal }, 'Shutdown initiated');
            httpServer.close();
            clearInterval(cleanupTimer);
            await queue.stop();
            db.close();
            process.exit(0);
        }
        process.on('SIGINT', () => gracefulShutdown('SIGINT'));
        process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    }

    return { app, start };
}
