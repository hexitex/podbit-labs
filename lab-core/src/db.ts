/**
 * Lab Database — persistent SQLite storage for jobs and artifacts.
 */

import Database from 'better-sqlite3';
import { join, dirname } from 'path';
import { mkdirSync } from 'fs';
import { logger } from './logger.js';

// =============================================================================
// TYPES
// =============================================================================

export interface JobRow {
    id: string;
    status: 'queued' | 'running' | 'completed' | 'failed';
    priority: number;
    spec: string;
    result: string | null;
    artifact_dir: string;
    created_at: string;
    started_at: string | null;
    completed_at: string | null;
}

export interface ArtifactRow {
    id: number;
    job_id: string;
    label: string;
    filename: string;
    type: string;
    category: string;
    size_bytes: number;
    created_at: string;
}

// =============================================================================
// SINGLETON
// =============================================================================

let db: Database.Database | null = null;

export function getDb(): Database.Database {
    if (!db) throw new Error('Database not initialized — call init() first');
    return db;
}

// =============================================================================
// INIT
// =============================================================================

export function init(dbPath?: string): void {
    const resolvedPath = dbPath || join(process.cwd(), 'data', 'lab.db');
    mkdirSync(dirname(resolvedPath), { recursive: true });
    db = new Database(resolvedPath);
    db.pragma('journal_mode = WAL');
    db.pragma('foreign_keys = ON');

    db.exec(`
        CREATE TABLE IF NOT EXISTS jobs (
            id              TEXT PRIMARY KEY,
            status          TEXT NOT NULL DEFAULT 'queued',
            priority        INTEGER NOT NULL DEFAULT 0,
            spec            TEXT NOT NULL,
            result          TEXT,
            artifact_dir    TEXT NOT NULL,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            started_at      TEXT,
            completed_at    TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
        CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs (created_at);

        CREATE TABLE IF NOT EXISTS artifacts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            label           TEXT NOT NULL,
            filename        TEXT NOT NULL,
            type            TEXT NOT NULL DEFAULT 'application/octet-stream',
            category        TEXT NOT NULL DEFAULT 'output',
            size_bytes      INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_artifacts_job ON artifacts (job_id);
    `);

    // Migration: add priority column if missing (existing DBs)
    try {
        db.prepare('SELECT priority FROM jobs LIMIT 1').get();
    } catch {
        db.exec('ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 0');
        logger.info('Migrated jobs table: added priority column');
    }

    db.exec('CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs (priority DESC, created_at ASC)');

    // Recovery: reset any jobs stuck as 'running' from a previous crash
    const stuck = db.prepare("UPDATE jobs SET status = 'queued', started_at = NULL WHERE status = 'running'").run();
    if (stuck.changes > 0) {
        logger.info({ count: stuck.changes }, 'Recovered stuck jobs → queued');
    }
}

export function createJob(id: string, spec: any, artifactDir: string, priority: number = 0): JobRow {
    const specJson = JSON.stringify(spec);
    getDb().prepare(`
        INSERT INTO jobs (id, status, priority, spec, artifact_dir) VALUES (?, 'queued', ?, ?, ?)
    `).run(id, priority, specJson, artifactDir);
    return getDb().prepare('SELECT * FROM jobs WHERE id = ?').get(id) as JobRow;
}

export function getJob(id: string): JobRow | undefined {
    return getDb().prepare('SELECT * FROM jobs WHERE id = ?').get(id) as JobRow | undefined;
}

export function updateJob(id: string, updates: Partial<Pick<JobRow, 'status' | 'result' | 'started_at' | 'completed_at'>>): void {
    const sets: string[] = [];
    const vals: any[] = [];
    if (updates.status !== undefined) { sets.push('status = ?'); vals.push(updates.status); }
    if (updates.result !== undefined) { sets.push('result = ?'); vals.push(updates.result); }
    if (updates.started_at !== undefined) { sets.push('started_at = ?'); vals.push(updates.started_at); }
    if (updates.completed_at !== undefined) { sets.push('completed_at = ?'); vals.push(updates.completed_at); }
    if (sets.length === 0) return;
    vals.push(id);
    getDb().prepare(`UPDATE jobs SET ${sets.join(', ')} WHERE id = ?`).run(...vals);
}

export function listJobs(filters?: { status?: string; limit?: number; offset?: number }): JobRow[] {
    let sql = 'SELECT * FROM jobs';
    const params: any[] = [];
    if (filters?.status) { sql += ' WHERE status = ?'; params.push(filters.status); }
    sql += ' ORDER BY created_at DESC';
    if (filters?.limit) { sql += ' LIMIT ?'; params.push(filters.limit); }
    if (filters?.offset) { sql += ' OFFSET ?'; params.push(filters.offset); }
    return getDb().prepare(sql).all(...params) as JobRow[];
}

export function getQueueDepth(): number {
    return (getDb().prepare("SELECT COUNT(*) as count FROM jobs WHERE status IN ('queued', 'running')").get() as { count: number }).count;
}

export function getQueuePosition(jobId: string): number {
    const row = getDb().prepare(`
        SELECT COUNT(*) as pos FROM jobs
        WHERE status IN ('queued', 'running')
        AND (priority > (SELECT priority FROM jobs WHERE id = ?)
             OR (priority = (SELECT priority FROM jobs WHERE id = ?)
                 AND created_at <= (SELECT created_at FROM jobs WHERE id = ?)))
    `).get(jobId, jobId, jobId) as { pos: number } | undefined;
    return row?.pos ?? 0;
}

export function getRunningCount(): number {
    return (getDb().prepare("SELECT COUNT(*) as count FROM jobs WHERE status = 'running'").get() as { count: number }).count;
}

export function getNextQueuedJob(): JobRow | undefined {
    return getDb().prepare(`
        SELECT * FROM jobs WHERE status = 'queued'
        ORDER BY priority DESC, created_at ASC LIMIT 1
    `).get() as JobRow | undefined;
}

export function cancelJob(id: string): boolean {
    const result = getDb().prepare(`
        UPDATE jobs SET status = 'failed', result = ?, completed_at = datetime('now')
        WHERE id = ? AND status IN ('queued', 'running')
    `).run(JSON.stringify({ verdict: 'error', confidence: 0, error: 'Job cancelled' }), id);
    return result.changes > 0;
}

export function getCompletedCount(): number {
    return (getDb().prepare("SELECT COUNT(*) as count FROM jobs WHERE status = 'completed'").get() as { count: number }).count;
}

export function getFailedCount(): number {
    return (getDb().prepare("SELECT COUNT(*) as count FROM jobs WHERE status = 'failed'").get() as { count: number }).count;
}

export function getAvgCompletionMs(): number {
    const row = getDb().prepare(`
        SELECT AVG((julianday(completed_at) - julianday(started_at)) * 86400000) as avg_ms
        FROM jobs WHERE status IN ('completed', 'failed')
        AND started_at IS NOT NULL AND completed_at IS NOT NULL
        ORDER BY completed_at DESC LIMIT 50
    `).get() as { avg_ms: number | null };
    return row?.avg_ms ?? 60000;
}

export function registerArtifact(jobId: string, label: string, filename: string, type: string, category: string, sizeBytes: number): number {
    return getDb().prepare(`
        INSERT INTO artifacts (job_id, label, filename, type, category, size_bytes) VALUES (?, ?, ?, ?, ?, ?)
    `).run(jobId, label, filename, type, category, sizeBytes).lastInsertRowid as number;
}

export function getJobArtifacts(jobId: string): ArtifactRow[] {
    return getDb().prepare('SELECT * FROM artifacts WHERE job_id = ? ORDER BY id').all(jobId) as ArtifactRow[];
}

export function pruneOldJobs(maxAgeDays: number = 7): { count: number; artifactDirs: string[] } {
    const rows = getDb().prepare(`
        SELECT artifact_dir FROM jobs WHERE status IN ('completed', 'failed')
        AND completed_at < datetime('now', '-' || ? || ' days')
    `).all(maxAgeDays) as { artifact_dir: string }[];
    const result = getDb().prepare(`
        DELETE FROM jobs WHERE status IN ('completed', 'failed')
        AND completed_at < datetime('now', '-' || ? || ' days')
    `).run(maxAgeDays);
    return { count: result.changes, artifactDirs: rows.map(r => r.artifact_dir) };
}

export function close(): void {
    if (db) { db.close(); db = null; }
}
