/**
 * Job Queue — concurrency controller with priority, cancellation, and timeouts.
 */

import * as db from './db.js';
import { logger } from './logger.js';

export type JobExecutor = (jobId: string, signal: AbortSignal) => Promise<void>;

export class JobQueue {
    private running = new Map<string, AbortController>();
    private maxConcurrent: number;
    private jobTimeoutMs: number;
    private pollIntervalMs: number;
    private timer: ReturnType<typeof setInterval> | null = null;
    private executor: JobExecutor;
    private stopping = false;

    constructor(maxConcurrent: number, jobTimeoutMs: number, executor: JobExecutor) {
        this.maxConcurrent = maxConcurrent;
        this.jobTimeoutMs = jobTimeoutMs;
        this.pollIntervalMs = 1000;
        this.executor = executor;
    }

    start(): void {
        logger.info({ maxConcurrent: this.maxConcurrent, jobTimeoutMs: this.jobTimeoutMs }, 'Job queue started');
        this.timer = setInterval(() => this.processQueue(), this.pollIntervalMs);
        this.processQueue();
    }

    async stop(): Promise<void> {
        this.stopping = true;
        if (this.timer) { clearInterval(this.timer); this.timer = null; }
        if (this.running.size === 0) return;
        logger.info({ inFlight: this.running.size }, 'Waiting for in-flight jobs to complete');
        const deadline = Date.now() + 30000;
        while (this.running.size > 0 && Date.now() < deadline) {
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        if (this.running.size > 0) {
            logger.warn({ remaining: this.running.size }, 'Aborting remaining jobs on shutdown');
            for (const [, controller] of this.running) controller.abort();
        }
    }

    enqueue(_jobId: string): void {
        this.processQueue();
    }

    cancel(jobId: string): boolean {
        const controller = this.running.get(jobId);
        if (controller) {
            controller.abort();
            this.running.delete(jobId);
            db.cancelJob(jobId);
            logger.info({ jobId }, 'Cancelled running job');
            return true;
        }
        const cancelled = db.cancelJob(jobId);
        if (cancelled) logger.info({ jobId }, 'Cancelled queued job');
        return cancelled;
    }

    get activeCount(): number { return this.running.size; }

    private processQueue(): void {
        if (this.stopping) return;
        while (this.running.size < this.maxConcurrent) {
            const next = db.getNextQueuedJob();
            if (!next) break;
            const jobId = next.id;
            const controller = new AbortController();
            this.running.set(jobId, controller);
            const timeoutHandle = setTimeout(() => {
                logger.warn({ jobId, timeoutMs: this.jobTimeoutMs }, 'Job timed out');
                controller.abort();
            }, this.jobTimeoutMs);
            this.executor(jobId, controller.signal)
                .catch(err => logger.error({ jobId, error: err.message }, 'Job execution error'))
                .finally(() => {
                    clearTimeout(timeoutHandle);
                    this.running.delete(jobId);
                    if (!this.stopping) this.processQueue();
                });
            db.updateJob(jobId, { status: 'running', started_at: new Date().toISOString() });
        }
    }
}
