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
        const details = Object.entries(output)
            .filter(([k]) => k !== 'supported' && k !== 'confidence')
            .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
            .join('; ');
        return {
            verdict: output.supported ? 'supported' : 'refuted',
            confidence: output.confidence ?? 0.85,
            details: details || `Code verdict: ${output.supported ? 'supported' : 'refuted'}`,
        };
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
        ? `\n\nWARNING — FAILED MEASUREMENTS:\nThe following ${measurementErrors.length} measurement(s) threw runtime errors and contain NO valid data:\n${measurementErrors.map(e => `  - ${e}`).join('\n')}\nDo NOT treat error objects as evidence for or against the hypothesis. If any failed measurement is critical to the verdict, you MUST return "inconclusive".\n`
        : '';

    const prompt = `You are evaluating the results of a computational experiment.

EXPERIMENT TYPE: ${spec.specType}

HYPOTHESIS:
${spec.hypothesis}

SETUP:
${JSON.stringify(spec.setup, null, 2)}

COMPUTED RESULTS:
${JSON.stringify(output, null, 2)}
${errorWarning}
Based on the computed results, determine whether the hypothesis is supported or refuted.

Respond with JSON:
{
    "verdict": "supported" | "refuted" | "inconclusive",
    "confidence": 0.0-1.0,
    "details": "explanation of your reasoning"
}`;

    try {
        const rawResponse = await callLlm(prompt, { role: 'evaluation', jsonSchema: { name: 'eval', schema: {} }, signal });

        let result: { verdict: string; confidence: number; details: string };
        try { result = JSON.parse(rawResponse); } catch {
            const block = rawResponse.match(/```json\s*([\s\S]*?)```/);
            if (block) { result = JSON.parse(block[1]); }
            else {
                const obj = rawResponse.match(/\{[\s\S]*\}/);
                if (obj) { result = JSON.parse(obj[0]); }
                else throw new Error(`Failed to parse: ${rawResponse.slice(0, 200)}`);
            }
        }

        const verdict = (['supported', 'refuted', 'inconclusive'].includes(result.verdict)
            ? result.verdict : 'inconclusive') as LabVerdict['verdict'];

        return {
            verdict,
            confidence: Math.max(0, Math.min(1, result.confidence ?? 0.5)),
            details: result.details || `LLM evaluation: ${verdict}`,
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
