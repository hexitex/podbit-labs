/**
 * Generic result evaluator — interprets sandbox output and produces verdicts.
 * Works for any lab that produces structured JSON output with numeric/boolean values.
 */

import { callLlm } from './llm.js';
import type { ExperimentSpec, SandboxResult, LabVerdict } from './types.js';

export { type LabVerdict };

export async function evaluate(
    spec: ExperimentSpec,
    sandbox: SandboxResult,
    options?: { signal?: AbortSignal },
): Promise<LabVerdict> {
    if (!sandbox.success || sandbox.killed) {
        return { verdict: 'error', confidence: 0, details: sandbox.stderr || 'Execution failed' };
    }

    if (!sandbox.parsedOutput) {
        return { verdict: 'inconclusive', confidence: 0, details: 'No structured output from experiment' };
    }

    const output = sandbox.parsedOutput;

    // Check for per-computation errors
    const errors = Object.entries(output)
        .filter(([, v]) => v && typeof v === 'object' && 'error' in (v as any))
        .map(([k, v]) => `${k}: ${(v as any).error}`);

    if (errors.length > 0 && errors.length === Object.keys(output).length) {
        return { verdict: 'error', confidence: 0, details: `All computations failed: ${errors.join('; ')}` };
    }

    // Check if the code produced a direct verdict
    if ('supported' in output && typeof output.supported === 'boolean') {
        // Count real measurements (non-error, non-metadata entries)
        const metaKeys = new Set(['supported', 'confidence', 'explanation', 'error', 'verified', 'testability']);
        const measurements = Object.entries(output)
            .filter(([k, v]) => !metaKeys.has(k) && !(v && typeof v === 'object' && 'error' in (v as any)));
        const errorEntries = Object.entries(output)
            .filter(([k, v]) => !metaKeys.has(k) && v && typeof v === 'object' && 'error' in (v as any));

        // If "supported: false" but no real measurements were computed, the code
        // likely crashed and set supported=false as a fallback. That's an execution
        // failure, not a physics refutation — don't let it poison the knowledge graph.
        if (!output.supported && measurements.length === 0) {
            const errorSummary = errorEntries.length > 0
                ? errorEntries.map(([k, v]) => `${k}: ${(v as any).error}`).join('; ')
                : 'Code set supported=false but produced no measurements';
            return {
                verdict: 'error', confidence: 0,
                details: `Computation failed — no measurements produced. ${errorSummary}`,
            };
        }

        const details = Object.entries(output)
            .filter(([k]) => k !== 'supported' && k !== 'confidence')
            .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
            .join('; ');

        // Don't blindly trust the code's own verdict at high confidence.
        // Send to the LLM evaluator for proper analysis when measurements exist.
        // Only use the direct path for simple boolean results with high signal.
        if (measurements.length >= 3) {
            // Enough data for the LLM to reason about — let it evaluate properly
            // instead of trusting the code's one-bit summary
        } else {
            return {
                verdict: output.supported ? 'supported' : 'refuted',
                confidence: Math.min(output.confidence ?? 0.7, 0.7),
                details: details || `Code verdict: ${output.supported ? 'supported' : 'refuted'}`,
            };
        }
    }

    // Fall back to LLM interpretation
    const result = await llmEvaluate(spec, output, errors, options?.signal);

    // Cap confidence when measurements partially failed
    if (errors.length > 0 && result.verdict !== 'inconclusive') {
        const failRatio = errors.length / Object.keys(output).length;
        const maxConfidence = Math.max(0, 1 - failRatio);
        if (result.confidence > maxConfidence) {
            result.details += ` [confidence capped from ${result.confidence.toFixed(2)} to ${maxConfidence.toFixed(2)} — ${errors.length}/${Object.keys(output).length} measurements failed]`;
            result.confidence = maxConfidence;
        }
    }

    return result;
}

async function llmEvaluate(spec: ExperimentSpec, output: Record<string, any>, measurementErrors: string[], signal?: AbortSignal): Promise<LabVerdict> {
    const errorWarning = measurementErrors.length > 0
        ? `\n\nNote — ${measurementErrors.length} measurement(s) errored and contain no valid data:\n${measurementErrors.map(e => `  - ${e}`).join('\n')}\nThese error objects are not evidence either way. If a critical measurement is missing, prefer "inconclusive" over a verdict drawn from the surviving measurements alone — but use your judgment about which measurements were actually load-bearing for the hypothesis.\n`
        : '';

    const prompt = `You are a scientific referee evaluating the results of a computational experiment against a hypothesis. Your job is to determine what was actually tested, what was assumed, and what remains open.

EXPERIMENT TYPE: ${spec.specType}

HYPOTHESIS:
${spec.hypothesis}

SETUP:
${JSON.stringify(spec.setup, null, 2)}

COMPUTED RESULTS:
${JSON.stringify(output, null, 2)}
${errorWarning}
## Your evaluation must cover:

1. **What was actually tested** — describe in concrete terms what the code computed and compared. Name the quantities.
2. **What was assumed or encoded** — if the code defined a formula that mirrors the hypothesis and then verified arithmetic consistency, say so. A test that encodes the claimed answer in its setup is tautological — it tells you the code is correct, not that the physics is right.
3. **Which sub-claims of the hypothesis are addressed** — break the hypothesis into its constituent claims and state which ones the experiment bears on and which it doesn't touch.
4. **What would a stronger test look like** — if the experiment is weak or tautological, describe what a non-circular test of the same hypothesis would measure.
5. **Verdict** — supported, refuted, or inconclusive. "Supported" requires that the experiment could have produced a different outcome under plausible alternative physics. If every test passes by construction, the verdict is "inconclusive" regardless of how many measurements succeeded.

Respond with JSON:
{
    "verdict": "supported" | "refuted" | "inconclusive",
    "confidence": 0.0-1.0,
    "details": "Your full evaluation covering all 5 points above. Be specific — quote measurement names and values. This is the primary record of what the experiment means.",
    "testedClaims": ["list of sub-claims the experiment actually bears on"],
    "untestedClaims": ["list of sub-claims the experiment does not address"],
    "tautologyRisk": "none | low | high — whether the code encodes the expected answer in its setup",
    "suggestedFollowUp": "what a better experiment would test, or null if the current one is sufficient"
}`;

    try {
        const rawResponse = await callLlm(prompt, { role: 'evaluation', jsonSchema: { name: 'eval', schema: {} }, signal });

        let result: any;
        try { result = JSON.parse(rawResponse); } catch {
            const block = rawResponse.match(/```json\s*([\s\S]*?)```/);
            if (block) { result = JSON.parse(block[1]); }
            else {
                const obj = rawResponse.match(/\{[\s\S]*\}/);
                if (obj) { result = JSON.parse(obj[0]); }
                else throw new Error(`Failed to parse: ${rawResponse.slice(0, 200)}`);
            }
        }

        // If the evaluator flagged high tautology risk, downgrade to inconclusive
        // regardless of what verdict it gave — a tautological test proves nothing.
        let verdict = (['supported', 'refuted', 'inconclusive'].includes(result.verdict)
            ? result.verdict : 'inconclusive') as LabVerdict['verdict'];
        const tautologyRisk = result.tautologyRisk?.toLowerCase?.() || 'none';
        if (tautologyRisk === 'high' && verdict === 'supported') {
            verdict = 'inconclusive';
            result.details = `[Downgraded from "supported" — evaluator flagged high tautology risk] ${result.details || ''}`;
        }

        // Build structured details from the rich evaluation fields
        const structuredDetails: Record<string, unknown> = {};
        if (Array.isArray(result.testedClaims) && result.testedClaims.length > 0) {
            structuredDetails.testedClaims = result.testedClaims;
        }
        if (Array.isArray(result.untestedClaims) && result.untestedClaims.length > 0) {
            structuredDetails.untestedClaims = result.untestedClaims;
        }
        if (tautologyRisk !== 'none') {
            structuredDetails.tautologyRisk = tautologyRisk;
        }
        if (result.suggestedFollowUp) {
            structuredDetails.suggestedFollowUp = result.suggestedFollowUp;
        }

        return {
            verdict,
            confidence: Math.max(0, Math.min(1, result.confidence ?? 0.5)),
            details: result.details || `LLM evaluation: ${verdict}`,
            structuredDetails: Object.keys(structuredDetails).length > 0 ? structuredDetails : undefined,
            evalPrompt: prompt,
            evalRawResponse: rawResponse,
        };
    } catch (err: any) {
        return {
            verdict: 'inconclusive', confidence: 0,
            details: `LLM evaluation failed: ${err.message}`,
            evalPrompt: prompt, evalRawResponse: err.message,
        };
    }
}

function parseNum(val: any): number | null {
    if (typeof val === 'number' && !isNaN(val)) return val;
    if (typeof val === 'string') { const n = parseFloat(val); if (!isNaN(n)) return n; }
    return null;
}
