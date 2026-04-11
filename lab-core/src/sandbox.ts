/**
 * Generic Python sandbox execution.
 * Spawns Python with executor.py, enforces timeout, captures JSON output.
 */

import { spawn } from 'child_process';
import { writeFile, unlink, mkdtemp, rmdir } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { fileURLToPath } from 'url';
import path from 'path';
import { getConfig } from './config.js';
import type { SandboxResult } from './types.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_EXECUTOR = join(__dirname, '..', 'executor.py');

export { type SandboxResult };

export async function executePython(code: string, executorPath?: string, signal?: AbortSignal, timeoutMs?: number): Promise<SandboxResult> {
    const cfg = getConfig();
    const startTime = Date.now();
    const executor = executorPath || DEFAULT_EXECUTOR;

    // If the job was already aborted before we even start, return immediately
    if (signal?.aborted) {
        return { success: false, stdout: '', stderr: 'Job aborted before sandbox execution', exitCode: -1, executionTimeMs: 0, killed: true };
    }

    let tempDir: string;
    let codePath: string;
    try {
        tempDir = await mkdtemp(join(tmpdir(), 'lab-'));
        codePath = join(tempDir, 'run.py');
        await writeFile(codePath, code, 'utf-8');
    } catch (err: any) {
        return { success: false, stdout: '', stderr: `Failed to create temp file: ${err.message}`, exitCode: -1, executionTimeMs: Date.now() - startTime, killed: false };
    }

    const sb = cfg.sandbox;
    const networkKill = sb.networkKillSwitch ? '1' : '0';

    return new Promise<SandboxResult>((resolve) => {
        let stdout = '';
        let stderr = '';
        let killed = false;
        let resolved = false;

        const finish = (exitCode: number, signal: string | null) => {
            if (resolved) return;
            resolved = true;
            const executionTimeMs = Date.now() - startTime;
            if (signal === 'SIGTERM' || signal === 'SIGKILL' || killed) killed = true;

            let parsedOutput: any;
            try {
                const t = stdout.trim();
                if (t) {
                    // Handle Python's Infinity/NaN which are not valid JSON
                    const sanitized = t.replace(/\bInfinity\b/g, 'null').replace(/\bNaN\b/g, 'null').replace(/\b-Infinity\b/g, 'null');
                    parsedOutput = JSON.parse(sanitized);
                }
            } catch { /* not JSON */ }

            unlink(codePath).catch(() => {});
            rmdir(tempDir!).catch(() => {});

            resolve({
                success: exitCode === 0 && !killed,
                stdout: stdout.slice(0, sb.maxOutputBytes),
                stderr: stderr.slice(0, 4096),
                exitCode, executionTimeMs, killed, parsedOutput,
            });
        };

        let proc;
        try {
            proc = spawn(sb.pythonPath, [executor, codePath, networkKill], {
                timeout: timeoutMs ?? sb.executionTimeoutMs,
                env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
                windowsHide: true,
                stdio: ['ignore', 'pipe', 'pipe'],
            });
        } catch { finish(-1, null); return; }

        proc.stdout!.on('data', (data: Buffer) => {
            stdout += data.toString();
            if (stdout.length > sb.maxOutputBytes) { killed = true; proc.kill('SIGTERM'); }
        });
        proc.stderr!.on('data', (data: Buffer) => { stderr += data.toString(); });
        proc.on('close', (exitCode: number | null, sig: string | null) => finish(exitCode ?? -1, sig));
        proc.on('error', (err: Error) => { stderr += err.message; finish(-1, null); });

        // Kill subprocess when the job's abort signal fires (job timeout or cancellation)
        if (signal) {
            const onAbort = () => {
                if (!resolved) {
                    killed = true;
                    stderr += '\nJob aborted - killing sandbox process';
                    proc.kill('SIGTERM');
                }
            };
            signal.addEventListener('abort', onAbort, { once: true });
            // Clean up the listener when the process finishes normally
            proc.on('close', () => signal.removeEventListener('abort', onAbort));
        }
    });
}
