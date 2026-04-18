/**
 * Multi-provider LLM client with role-based model selection and fallback.
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { Agent, setGlobalDispatcher } from 'undici';
import { getConfig } from './config.js';
import type { ModelSlotConfig } from './types.js';
import { LabError, TruncatedError } from './errors.js';
import { logger } from './logger.js';

// Extend undici's default headers/body timeouts for all fetch calls in this process.
// Reasoning models can think for minutes before sending the first response byte —
// the default 60s headersTimeout kills these calls. The abort signal handles cancellation.
setGlobalDispatcher(new Agent({
    headersTimeout: 3_600_000,  // 1 hour — reasoning models can think this long before first byte
    bodyTimeout: 3_600_000,     // 1 hour — streaming responses from slow thinking models
    connectTimeout: 30_000,     // 30s — connection establishment should be fast
    keepAliveTimeout: 30_000,
}));

export type LlmRole = 'codegen' | 'evaluation';

export interface LlmCallOptions {
    role?: LlmRole;
    maxTokens?: number;
    temperature?: number;
    systemPrompt?: string;
    jsonSchema?: { name: string; schema: Record<string, any> };
    signal?: AbortSignal;
}

// System prompt cache
const promptCache = new Map<string, string>();

export function loadSystemPrompt(pathOrInline?: string): string | undefined {
    if (!pathOrInline) return undefined;
    if (promptCache.has(pathOrInline)) return promptCache.get(pathOrInline)!;

    if (pathOrInline.includes('/') || pathOrInline.endsWith('.txt') || pathOrInline.endsWith('.md')) {
        try {
            const fullPath = pathOrInline.startsWith('/') || pathOrInline.includes(':')
                ? pathOrInline
                : join(process.cwd(), pathOrInline);
            const content = readFileSync(fullPath, 'utf-8');
            promptCache.set(pathOrInline, content);
            return content;
        } catch {
            logger.warn({ path: pathOrInline }, 'Failed to load system prompt file, using as inline');
        }
    }
    promptCache.set(pathOrInline, pathOrInline);
    return pathOrInline;
}

// Provider internals
interface ProviderCallOptions {
    maxTokens?: number;
    temperature: number;
    jsonMode: boolean;
    signal?: AbortSignal;
    timeoutMs: number;
    noThink?: boolean;
}

/** Delay that resolves immediately if the abort signal fires. */
function abortableDelay(ms: number, signal: AbortSignal): Promise<void> {
    if (signal.aborted) return Promise.resolve();
    return new Promise(resolve => {
        const timer = setTimeout(resolve, ms);
        signal.addEventListener('abort', () => { clearTimeout(timer); resolve(); }, { once: true });
    });
}

function stripThinkingBlocks(text: string): string {
    let cleaned = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
    cleaned = cleaned.replace(/<thinking>[\s\S]*?<\/thinking>/gi, '').trim();
    cleaned = cleaned.replace(/^<think>[\s\S]*$/gi, '').trim();
    return cleaned || text;
}

function mergeSignals(signal?: AbortSignal, timeoutMs?: number): AbortSignal {
    // timeoutMs=0 means "no independent LLM timeout — rely on the job's abort signal."
    // Only create a timeout signal when a positive value is explicitly configured.
    if (!timeoutMs || timeoutMs <= 0) return signal || AbortSignal.timeout(900_000); // 15 min safety net if no signal at all
    const timeout = AbortSignal.timeout(timeoutMs);
    if (!signal) return timeout;
    return AbortSignal.any([signal, timeout]);
}

function buildOpenAICompatibleChatUrl(endpoint: string): string {
    const trimmed = endpoint.trim();
    const strip = (s: string) => s.replace(/\/+$/, '');
    if (trimmed.includes('/chat/completions')) return trimmed;
    try {
        const parsed = new URL(trimmed);
        const base = strip(parsed.pathname);
        parsed.pathname = (base.length > 0 && base !== '/') ? `${base}/chat/completions` : '/v1/chat/completions';
        return parsed.toString();
    } catch {
        const cleaned = strip(trimmed);
        return /^https?:\/\/[^/]+\/.+/.test(cleaned) ? `${cleaned}/chat/completions` : `${cleaned}/v1/chat/completions`;
    }
}

async function callOpenAICompatible(
    endpoint: string, apiKey: string, model: string,
    messages: Array<{ role: string; content: string }>,
    opts: ProviderCallOptions,
): Promise<string> {
    const body: Record<string, any> = { model, messages, temperature: opts.temperature, stream: true };
    if (opts.maxTokens != null) body.max_tokens = opts.maxTokens;
    if (opts.noThink) { body.enable_thinking = false; body.chat_template_kwargs = { enable_thinking: false }; }

    const url = buildOpenAICompatibleChatUrl(endpoint);
    const signal = mergeSignals(opts.signal, opts.timeoutMs);
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}) },
        body: JSON.stringify(body),
        signal,
    });

    if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new LabError(`LLM call failed (${response.status}): ${text.slice(0, 300)}`, 'llm', response.status === 429 || response.status >= 500);
    }

    // Read SSE stream and concatenate content deltas
    const reader = response.body?.getReader();
    if (!reader) {
        throw new LabError('No response body for streaming', 'llm', true);
    }

    const decoder = new TextDecoder();
    let content = '';
    let buffer = '';
    let finishReason = '';

    const processLine = (line: string) => {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]') return;
        if (!trimmed.startsWith('data: ')) return;
        try {
            const chunk = JSON.parse(trimmed.slice(6));
            const delta = chunk.choices?.[0]?.delta?.content;
            if (delta) content += delta;
            const fr = chunk.choices?.[0]?.finish_reason;
            if (fr) finishReason = fr;
        } catch { /* skip malformed chunk */ }
    };

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) processLine(line);
        }
        // Flush remaining buffer - final SSE chunk may lack trailing \n
        if (buffer.trim()) processLine(buffer);
    } finally {
        reader.releaseLock();
    }

    if (finishReason === 'length') {
        logger.warn({ model, contentLen: content.length }, 'LLM response truncated (finish_reason: length)');
        throw new TruncatedError(content);
    }

    return content;
}

async function callAnthropic(
    endpoint: string, apiKey: string, model: string,
    messages: Array<{ role: string; content: string }>,
    opts: ProviderCallOptions,
): Promise<string> {
    const systemMsgs = messages.filter(m => m.role === 'system');
    const nonSystemMsgs = messages.filter(m => m.role !== 'system');
    const body: Record<string, any> = { model, max_tokens: opts.maxTokens || 16384, messages: nonSystemMsgs };
    if (systemMsgs.length > 0) body.system = systemMsgs.map(m => m.content).join('\n\n');

    const url = endpoint.endsWith('/v1/messages') ? endpoint : `${endpoint.replace(/\/+$/, '')}/v1/messages`;
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
        body: JSON.stringify(body),
        signal: mergeSignals(opts.signal, opts.timeoutMs),
    });

    if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new LabError(`Anthropic call failed (${response.status}): ${text.slice(0, 300)}`, 'llm', response.status === 429 || response.status >= 500);
    }
    const data = await response.json() as any;
    const text = data.content?.[0]?.text || '';
    if (data.stop_reason === 'max_tokens') {
        logger.warn({ model, contentLen: text.length }, 'Anthropic response truncated (stop_reason: max_tokens)');
        throw new TruncatedError(text);
    }
    return text;
}

function dispatchCall(slot: ModelSlotConfig, messages: Array<{ role: string; content: string }>, opts: ProviderCallOptions): Promise<string> {
    switch (slot.provider) {
        case 'anthropic': return callAnthropic(slot.endpoint, slot.apiKey, slot.model, messages, opts);
        default: return callOpenAICompatible(slot.endpoint, slot.apiKey, slot.model, messages, opts);
    }
}

function resolveSlot(role: LlmRole): ModelSlotConfig {
    const cfg = getConfig();
    if (role === 'evaluation' && cfg.models.evaluation) return cfg.models.evaluation;
    return cfg.models.codegen;
}

/**
 * Call Podbit's LLM proxy endpoint. Routes through Podbit's subsystem model routing,
 * rate limiting, budget tracking, and streaming. No direct external API calls needed.
 */
async function callPodbit(
    podbitUrl: string, subsystem: string, role: LlmRole,
    messages: Array<{ role: string; content: string }>,
    opts: { maxTokens?: number; temperature?: number; jsonSchema?: any; signal?: AbortSignal; timeoutMs?: number },
): Promise<string> {
    const url = `${podbitUrl.replace(/\/+$/, '')}/api/llm/call`;
    const bodyObj = {
        subsystem,
        messages,
        role: role === 'evaluation' ? 'consultant' : 'primary',
        options: {
            maxTokens: opts.maxTokens,
            temperature: opts.temperature,
            jsonSchema: opts.jsonSchema,
        },
    };
    const bodyJson = JSON.stringify(bodyObj);

    // Podbit's model registry requestTimeout is the real timeout for the LLM call.
    // Lab side just relies on the job's abort signal (opts.signal) to stay within budget.
    const signal = mergeSignals(opts.signal, opts.timeoutMs);
    const maxRetries = 10;
    let lastError = '';

    for (let attempt = 0; attempt < maxRetries; attempt++) {
        // Don't bother fetching if the signal is already dead — every call would fail instantly
        if (signal.aborted) {
            throw new LabError(`Podbit LLM call aborted before attempt ${attempt + 1} for "${subsystem}"`, 'llm', false);
        }

        let response: Response;
        try {
            response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: bodyJson,
                signal,
            });
        } catch (fetchErr: any) {
            // Connection refused, DNS failure, Podbit restarting, network blip
            const detail = describeNetworkError(fetchErr) || fetchErr.message || 'fetch failed';
            lastError = `Podbit unreachable at ${podbitUrl}: ${detail}`;

            // AbortError = caller cancelled, TimeoutError = AbortSignal.timeout() fired,
            // or signal already aborted = all retries will fail instantly. Don't retry any of these.
            if (fetchErr.name === 'AbortError' || fetchErr.name === 'TimeoutError' || signal.aborted) {
                throw new LabError(lastError, 'llm', false);
            }

            // Transient network error — wait and retry
            const delay = Math.min(3000 * (attempt + 1), 15000);
            logger.warn({ subsystem, attempt, delay, error: detail }, 'Podbit connection failed — retrying');
            if (attempt >= maxRetries - 1) break; // fall through to "retries exhausted"
            await abortableDelay(delay, signal);
            continue;
        }

        if (response.status === 429) {
            // Model at capacity or rate-limited — wait and retry
            let retryMs = 5000;
            try {
                const errData = await response.json() as any;
                retryMs = errData.retryAfterMs || 5000;
                lastError = errData.error || 'rate limited';
                logger.info({ subsystem, attempt, retryMs, model: errData.model }, 'Podbit 429 — waiting for capacity');
            } catch { /* ignore parse error */ }
            await abortableDelay(retryMs, signal);
            continue;
        }

        if (!response.ok) {
            const text = await response.text().catch(() => '');
            const retryable = response.status === 503 || response.status >= 500;
            throw new LabError(`Podbit LLM proxy failed (${response.status}): ${text.slice(0, 300)}`, 'llm', retryable);
        }

        const data = await response.json() as any;
        if (!data.content) {
            throw new LabError(`Podbit LLM proxy returned empty content for subsystem "${subsystem}"`, 'llm', false);
        }
        return data.content;
    }

    // All retries exhausted
    throw new LabError(`Podbit LLM proxy: ${maxRetries} attempts failed for "${subsystem}". Last error: ${lastError}`, 'llm', true);
}

export async function callLlm(prompt: string, options: LlmCallOptions = {}): Promise<string> {
    const cfg = getConfig();
    const role = options.role || 'codegen';

    // Route through Podbit if configured (local labs)
    if (cfg.podbit?.url) {
        const subsystem = cfg.podbit.subsystem || 'lab:unknown';
        const slot = resolveSlot(role);
        const sysPrompt = options.systemPrompt || loadSystemPrompt(slot.systemPrompt);

        let messages: Array<{ role: string; content: string }> = [];
        if (sysPrompt && slot.mergeSystemPrompt) {
            messages.push({ role: 'user', content: sysPrompt + '\n\n' + prompt });
        } else {
            if (sysPrompt) messages.push({ role: 'system', content: sysPrompt });
            messages.push({ role: 'user', content: prompt });
        }

        // When routing through Podbit, don't override Podbit's model registry settings
        // with the lab's local (often placeholder) model slot config. Podbit owns:
        //   - maxTokens (from model_registry.max_tokens)
        //   - timeout (from model_registry.request_timeout + retry_window_minutes)
        // Only pass values the caller explicitly set in options (e.g. the codegen call
        // may want a specific temperature). The local slot defaults are irrelevant.
        logger.debug({ role, subsystem, podbitUrl: cfg.podbit.url, promptLen: prompt.length }, 'Podbit LLM call starting');
        let result = await callPodbit(cfg.podbit.url, subsystem, role, messages, {
            maxTokens: options.maxTokens,      // only if caller explicitly set it
            temperature: options.temperature,   // only if caller explicitly set it
            jsonSchema: options.jsonSchema,
            signal: options.signal,
        });
        {
            const stripped = stripThinkingBlocks(result);
            if (stripped !== result) { logger.info({ role, originalLen: result.length, strippedLen: stripped.length }, 'Stripped thinking blocks'); result = stripped; }
        }
        logger.debug({ role, subsystem, responseLen: result.length }, 'Podbit LLM call completed');
        return result;
    }

    // Direct provider call (remote labs or no podbit config)
    const slot = resolveSlot(role);

    let messages: Array<{ role: string; content: string }> = [];
    const sysPrompt = options.systemPrompt || loadSystemPrompt(slot.systemPrompt);
    if (sysPrompt && slot.mergeSystemPrompt) {
        messages.push({ role: 'user', content: sysPrompt + '\n\n' + prompt });
    } else {
        if (sysPrompt) messages.push({ role: 'system', content: sysPrompt });
        messages.push({ role: 'user', content: prompt });
    }

    const callOpts: ProviderCallOptions = {
        maxTokens: options.maxTokens ?? slot.maxTokens,
        temperature: options.temperature ?? slot.temperature,
        jsonMode: !!options.jsonSchema,
        signal: options.signal,
        timeoutMs: slot.timeoutMs,
        noThink: slot.noThink,
    };

    try {
        logger.debug({ role, model: slot.model, endpoint: slot.endpoint, promptLen: prompt.length }, 'LLM call starting');
        let result = await dispatchCall(slot, messages, callOpts);
        {
            const stripped = stripThinkingBlocks(result);
            if (stripped !== result) { logger.info({ role, originalLen: result.length, strippedLen: stripped.length }, 'Stripped thinking blocks'); result = stripped; }
        }
        logger.debug({ role, responseLen: result.length }, 'LLM call completed');
        return result;
    } catch (err: any) {
        // Network errors from undici fetch arrive as TypeError("fetch failed") with the
        // real reason wrapped on err.cause. Unwrap so retries and logs aren't "fetch failed".
        const networkInfo = describeNetworkError(err);
        const enrichedMessage = networkInfo
            ? `LLM call network error (${slot.endpoint}, model=${slot.model}): ${networkInfo}`
            : (err instanceof LabError ? err.message : (err.message || 'LLM call failed'));

        // Bare network failures are RETRYABLE — DNS blips, connection resets, transient
        // upstream outages all deserve a second chance, both at the fallback-model level
        // here and at the lab-core retry-loop level above us.
        const isRetryable = networkInfo !== null
            ? true
            : (err instanceof LabError ? err.retryable : false);

        if (cfg.models.fallback && isRetryable) {
            logger.warn({ role, model: slot.model, error: enrichedMessage }, 'Primary LLM failed, trying fallback');
            try { return await dispatchCall(cfg.models.fallback, messages, { ...callOpts, timeoutMs: cfg.models.fallback.timeoutMs }); }
            catch (fbErr: any) {
                const fbNetwork = describeNetworkError(fbErr);
                const fbMsg = fbNetwork ? `network error (${cfg.models.fallback.endpoint}): ${fbNetwork}` : (fbErr.message || 'fallback failed');
                throw new LabError(`Primary and fallback LLM failed. Primary: ${enrichedMessage}. Fallback: ${fbMsg}`, 'llm', isRetryable);
            }
        }
        if (err instanceof LabError && !networkInfo) throw err;
        throw new LabError(enrichedMessage, 'llm', isRetryable);
    }
}

/**
 * Unwrap a Node fetch / undici network error into a human-readable description.
 * Returns null if `err` is not a network error (so callers can fall back to other handling).
 *
 * Node's undici fetch surfaces network failures as `TypeError('fetch failed')` with the
 * actual cause on `err.cause` — typically an Error with `code` (e.g. ENOTFOUND, ECONNREFUSED,
 * ECONNRESET, ETIMEDOUT, UND_ERR_SOCKET, CERT_HAS_EXPIRED) and a `message`. AbortError shows
 * up as `err.name === 'AbortError'` (timeout / cancellation).
 */
function describeNetworkError(err: any): string | null {
    if (!err) return null;
    if (err.name === 'AbortError') return 'AbortError (request timed out or was cancelled)';
    const isFetchFailed = err instanceof TypeError && /fetch failed/i.test(err.message || '');
    const cause = err.cause;
    if (isFetchFailed || cause) {
        if (cause && typeof cause === 'object') {
            const code = (cause as any).code ? `[${(cause as any).code}] ` : '';
            const msg = (cause as any).message || String(cause);
            const errno = (cause as any).errno != null ? ` (errno ${(cause as any).errno})` : '';
            return `${code}${msg}${errno}`;
        }
        if (isFetchFailed) return 'fetch failed (no cause exposed — DNS, connection reset, or upstream unreachable)';
    }
    return null;
}

export async function callLlmJson<T = any>(prompt: string, options: LlmCallOptions = {}): Promise<T> {
    const raw = await callLlm(prompt, { ...options, jsonSchema: options.jsonSchema || { name: 'r', schema: {} } });
    try { return JSON.parse(raw); } catch { /* fall through */ }
    const block = raw.match(/```json\s*([\s\S]*?)```/);
    if (block) { try { return JSON.parse(block[1]); } catch { /* fall through */ } }
    const obj = raw.match(/\{[\s\S]*\}/);
    if (obj) { try { return JSON.parse(obj[0]); } catch { /* fall through */ } }
    throw new LabError(`Failed to parse LLM JSON: ${raw.slice(0, 200)}`, 'llm', false);
}
