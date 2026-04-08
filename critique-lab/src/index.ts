/**
 * Critique Lab — pure LLM critique of knowledge graph nodes and experiments.
 *
 * No computation, no codegen, no sandbox. Receives content and context,
 * asks an LLM to evaluate, and returns a structured verdict.
 *
 * Spec types:
 *   - node_critique:      Evaluate a node for quality, specificity, novelty, etc.
 *   - experiment_review:  Review a lab experiment's methodology and verdict validity
 *
 * Pipeline:
 *   - generate: returns the claim/context text (no code)
 *   - execute:  no-op (returns structured output for artifact tracing)
 *   - evaluate: LLM critiques/reviews based on spec type
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import {
    createLabServer, loadConfig, callLlm, db,
    type LabPipeline, type PipelineContext, type BaseLabConfig,
    type LabVerdict, type SandboxResult, type CodegenResult,
} from '@lab/core';

// =============================================================================
// CONFIG
// =============================================================================

interface CritiqueLabConfig extends BaseLabConfig {
    capabilities: {
        specTypes: Record<string, string>;
        queueLimit: number;
        artifactTtlSeconds: number;
        features: string[];
    };
}

const config = loadConfig<CritiqueLabConfig>({
    capabilities: {
        specTypes: {
            node_critique: 'Evaluate a knowledge graph node for quality, specificity, novelty, grounding, and falsifiability.',
            experiment_review: 'Review a lab experiment\'s methodology. Assesses whether the experiment actually tests the claim, checks for confounds and operationalization gaps, and validates the verdict. Returns: confirm (methodology sound), correct (verdict wrong), or retest (needs redesign). Receives full context: original claim, experiment spec, lab verdict, code, and output.',
        },
        queueLimit: 20,
        artifactTtlSeconds: 604800,
        features: ['llm-critique', 'no-compute', 'experiment-review'],
    },
});

// =============================================================================
// PROMPTS
// =============================================================================

const PROMPT_DIR = join(process.cwd(), 'prompts');
const promptCache: Record<string, string> = {};

function loadPrompt(filename: string): string {
    if (!promptCache[filename]) {
        promptCache[filename] = readFileSync(join(PROMPT_DIR, filename), 'utf-8');
    }
    return promptCache[filename];
}

// =============================================================================
// NODE CRITIQUE PROMPT BUILDER
// =============================================================================

function buildCritiquePrompt(spec: any): string {
    const s = spec.setup || {};
    const sections: string[] = [loadPrompt('critique.md')];

    // ── Node metadata ──
    sections.push(`\n## Node Under Review`);
    sections.push(`Domain: ${s.domain || 'unknown'}`);
    sections.push(`Type: ${s.nodeType || 'unknown'}`);
    if (s.contributor) sections.push(`Contributor: ${s.contributor}`);
    if (s.trajectory) sections.push(`Trajectory: ${s.trajectory}`);
    if (s.weight != null) sections.push(`Weight: ${s.weight}`);
    if (s.salience != null) sections.push(`Salience: ${s.salience}`);
    if (s.specificity != null) sections.push(`Specificity: ${s.specificity}`);
    if (s.createdAt) sections.push(`Created: ${s.createdAt}`);
    if (s.verificationStatus) sections.push(`Verification status: ${s.verificationStatus} (score: ${s.verificationScore ?? 'N/A'})`);

    sections.push(`\n### Content`);
    sections.push(s.nodeContent || s.claim_content || spec.hypothesis || '(no content)');

    // ── Parent nodes ──
    const parents = s.parentContent;
    if (parents && Array.isArray(parents) && parents.length > 0) {
        sections.push(`\n## Parent Nodes (${parents.length} sources)`);
        sections.push('These are the nodes this node was synthesized from. Compare the node content against its sources to assess novelty.');
        for (const p of parents) {
            if (typeof p === 'string') {
                sections.push(`- ${p}`);
            } else {
                sections.push(`- [${p.id || '?'}] (${p.type || '?'}, w:${p.weight?.toFixed(3) ?? '?'}, ${p.domain || '?'}): ${p.content}`);
            }
        }
    }

    // ── Child nodes ──
    const children = s.childSample;
    if (children && Array.isArray(children) && children.length > 0) {
        sections.push(`\n## Child Nodes (${s.childCount ?? children.length} total, showing ${children.length})`);
        sections.push('These nodes were derived from the node under review. Many productive children suggest the node is generative.');
        for (const c of children) {
            sections.push(`- [${c.id || '?'}] (${c.type || '?'}, w:${c.weight?.toFixed(3) ?? '?'}): ${c.content}`);
        }
    }

    // ── Prior verification attempts ──
    const priors = s.priorVerifications;
    if (priors && Array.isArray(priors) && priors.length > 0) {
        sections.push(`\n## Prior Verification Attempts (${priors.length})`);
        for (const v of priors) {
            sections.push(`- [${v.status}] ${v.claimType || ''} — ${v.hypothesis || '(no hypothesis)'}`);
            if (v.confidence != null) sections.push(`  Confidence: ${v.confidence}`);
            if (v.error) sections.push(`  Error: ${v.error}`);
            if (v.when) sections.push(`  When: ${v.when}`);
        }
    }

    return sections.join('\n');
}

// =============================================================================
// EXPERIMENT REVIEW PROMPT BUILDER
// =============================================================================

function buildExperimentReviewPrompt(spec: any): string {
    const s = spec.setup || {};
    const sections: string[] = [loadPrompt('experiment-review.md')];

    // ── The Claim Being Tested ──
    sections.push(`\n## Claim Under Test`);
    sections.push(`Domain: ${s.claimDomain || 'unknown'}`);
    sections.push(`Node Type: ${s.claimType || 'unknown'}`);
    if (s.claimContributor) sections.push(`Contributor: ${s.claimContributor}`);
    sections.push(`\n### Claim Content`);
    sections.push(s.claim || spec.hypothesis || '(no content)');

    // ── Parent nodes (source material for the claim) ──
    const parents = s.parents;
    if (parents && Array.isArray(parents) && parents.length > 0) {
        sections.push(`\n## Source Nodes (${parents.length})`);
        sections.push('The claim was synthesized from these nodes. Understanding the sources helps assess whether the claim\'s concepts were correctly operationalized.');
        for (const p of parents) {
            if (typeof p === 'string') {
                sections.push(`- ${p}`);
            } else {
                sections.push(`- [${p.id || '?'}] (${p.type || '?'}, w:${p.weight?.toFixed(3) ?? '?'}, ${p.domain || '?'}): ${p.content}`);
            }
        }
    }

    // ── The Experiment ──
    sections.push(`\n## Experiment Specification`);
    const origSpec = s.originalSpec;
    if (origSpec) {
        sections.push(`Spec Type: ${origSpec.specType || 'unknown'}`);
        sections.push(`Hypothesis: ${origSpec.hypothesis || '(none)'}`);
        sections.push(`\n### Setup Parameters`);
        sections.push('```json');
        sections.push(JSON.stringify(origSpec.setup || {}, null, 2));
        sections.push('```');
        if (origSpec.claimType) sections.push(`Claim Type: ${origSpec.claimType}`);
    }

    // ── Lab Result ──
    sections.push(`\n## Lab Result`);
    sections.push(`Lab: ${s.labName || 'unknown'}`);
    sections.push(`Verdict: **${s.labVerdict || 'unknown'}**`);
    sections.push(`Confidence: ${s.labConfidence ?? 'N/A'}`);
    if (s.labDetails) {
        sections.push(`\n### Verdict Details`);
        sections.push(s.labDetails);
    }

    // ── Experiment Code ──
    if (s.experimentCode) {
        sections.push(`\n## Generated Experiment Code`);
        sections.push('This is the code the lab generated and executed. Review it to understand exactly what was tested.');
        sections.push('```python');
        sections.push(s.experimentCode);
        sections.push('```');
    }

    // ── Experiment Output ──
    if (s.experimentStdout) {
        sections.push(`\n## Experiment Output (stdout)`);
        sections.push('```');
        sections.push(s.experimentStdout);
        sections.push('```');
    }

    return sections.join('\n');
}

// =============================================================================
// PIPELINE
// =============================================================================

const pipeline: LabPipeline = {
    async generate(ctx: PipelineContext): Promise<CodegenResult> {
        const isReview = ctx.spec.specType === 'experiment_review';
        const prompt = isReview
            ? buildExperimentReviewPrompt(ctx.spec)
            : buildCritiquePrompt(ctx.spec);
        return {
            code: ctx.spec.hypothesis || '',
            prompt,
            rawResponse: `(critique-lab: no codegen — LLM ${isReview ? 'experiment review' : 'critique'} happens in evaluate stage)`,
        };
    },

    async execute(_ctx: PipelineContext, _code: string): Promise<SandboxResult> {
        return {
            success: true,
            stdout: JSON.stringify({ result: { claim: _code } }),
            stderr: '',
            exitCode: 0,
            executionTimeMs: 0,
            killed: false,
            parsedOutput: { result: { claim: _code }, error: null, execution_time_ms: 0 },
        };
    },

    async evaluate(ctx: PipelineContext, _sandbox: SandboxResult): Promise<LabVerdict> {
        const isReview = ctx.spec.specType === 'experiment_review';
        const prompt = isReview
            ? buildExperimentReviewPrompt(ctx.spec)
            : buildCritiquePrompt(ctx.spec);

        const TIMEOUT_MS = 3 * 60 * 1000;
        const signal = AbortSignal.any([
            ctx.signal,
            AbortSignal.timeout(TIMEOUT_MS),
        ]);

        try {
            const rawResponse = await callLlm(prompt, {
                role: 'codegen',
                jsonSchema: { name: isReview ? 'experiment_review' : 'critique', schema: {} },
                signal,
            });

            let result: any;
            try {
                result = JSON.parse(rawResponse);
            } catch {
                const match = rawResponse.match(/\{[\s\S]*\}/);
                if (match) {
                    try { result = JSON.parse(match[0]); } catch { /* fall through */ }
                }
            }

            if (!result) {
                return {
                    verdict: 'inconclusive',
                    confidence: 0,
                    details: `Failed to parse response: ${rawResponse.slice(0, 300)}`,
                    evalPrompt: prompt,
                    evalRawResponse: rawResponse,
                };
            }

            if (isReview) {
                return buildExperimentReviewVerdict(result, prompt, rawResponse);
            } else {
                return buildNodeCritiqueVerdict(result, prompt, rawResponse);
            }
        } catch (err: any) {
            return {
                verdict: 'inconclusive',
                confidence: 0,
                details: `${isReview ? 'Experiment review' : 'Critique'} failed: ${err.message}`,
                evalPrompt: prompt,
                evalRawResponse: err.message,
            };
        }
    },
};

// =============================================================================
// VERDICT BUILDERS
// =============================================================================

function buildNodeCritiqueVerdict(result: any, prompt: string, rawResponse: string): LabVerdict {
    const verdict = mapNodeVerdict(result.verdict || result.quality);
    const confidence = Math.max(0, Math.min(1, result.confidence ?? 0.5));

    const scores = result.scores || {};

    // Structured payload — fielded data the GUI renders as a table.
    const structuredDetails: Record<string, unknown> = {};
    if (result.recommendation) structuredDetails.recommendation = result.recommendation;
    if (result.critique) structuredDetails.critique = result.critique;
    if (Object.keys(scores).length > 0) structuredDetails.scores = scores;
    if (Array.isArray(result.issues) && result.issues.length > 0) structuredDetails.issues = result.issues;
    if (result.suggestion) structuredDetails.suggestion = result.suggestion;

    // Short prose summary for the `details` text field — never mash structured
    // data in here. Prefer the LLM's own critique if it gave one, otherwise
    // synthesise a one-line headline from the recommendation/scores.
    let detailsProse = '';
    if (result.critique) {
        const trimmed = String(result.critique).replace(/\s+/g, ' ').trim();
        detailsProse = trimmed.length > 400 ? trimmed.slice(0, 400) + '…' : trimmed;
    } else if (result.recommendation) {
        detailsProse = `Recommendation: ${result.recommendation}`;
    } else {
        detailsProse = `Verdict: ${verdict}`;
    }

    return {
        verdict,
        confidence,
        details: detailsProse,
        structuredDetails: Object.keys(structuredDetails).length > 0 ? structuredDetails : undefined,
        evalPrompt: prompt,
        evalRawResponse: rawResponse,
    };
}

function buildExperimentReviewVerdict(result: any, prompt: string, rawResponse: string): LabVerdict {
    // Map the action to a lab verdict that Podbit's chaining module can parse
    const action = result.action?.toLowerCase() || 'confirm';
    let verdict: LabVerdict['verdict'];

    switch (action) {
        case 'confirm':
            verdict = 'supported'; // Methodology is sound → original verdict stands
            break;
        case 'correct':
            verdict = 'refuted';   // Methodology flawed → original verdict is wrong
            break;
        case 'retest':
            verdict = 'inconclusive'; // Needs redesign → inconclusive
            break;
        default:
            verdict = 'inconclusive';
    }

    const confidence = Math.max(0, Math.min(1, result.methodologyScore ?? result.confidence ?? 0.5));

    // The full critique decision rides in `structuredDetails` as a real object —
    // Podbit's chaining module reads it field-by-field and the GUI renders it as
    // labelled fields. The prose `details` field carries a short human-readable
    // summary that's safe to render anywhere as plain text.
    const structuredDetails: Record<string, unknown> = {
        action: result.action || 'confirm',
        correctedVerdict: result.correctedVerdict,
        correctedConfidence: result.correctedConfidence,
        methodologyScore: result.methodologyScore,
        issues: result.issues || [],
        guidance: result.guidance,
        critique: result.critique,
        rewrittenClaim: result.rewrittenClaim || undefined,
    };

    // Build a short prose summary for the `details` text field. Prefer the LLM's
    // own narrative critique if it gave one, otherwise stitch together the action
    // and a hint at the methodology score.
    const proseParts: string[] = [];
    proseParts.push(`Action: ${(result.action || 'confirm').toUpperCase()}`);
    if (typeof result.methodologyScore === 'number') {
        proseParts.push(`Methodology score: ${result.methodologyScore.toFixed(2)}`);
    }
    if (result.critique) {
        // Trim to a reasonable summary length — full critique stays in structuredDetails
        const trimmed = String(result.critique).replace(/\s+/g, ' ').trim();
        proseParts.push(trimmed.length > 400 ? trimmed.slice(0, 400) + '…' : trimmed);
    } else if (Array.isArray(result.issues) && result.issues.length > 0) {
        proseParts.push(`Issues: ${result.issues.slice(0, 3).join('; ')}${result.issues.length > 3 ? ` (+${result.issues.length - 3} more)` : ''}`);
    }
    const detailsProse = proseParts.join(' — ');

    return {
        verdict,
        confidence,
        details: detailsProse,
        structuredDetails,
        evalPrompt: prompt,
        evalRawResponse: rawResponse,
    };
}

function mapNodeVerdict(raw: string | undefined): LabVerdict['verdict'] {
    if (!raw) return 'inconclusive';
    const v = raw.toLowerCase().trim();
    if (v === 'high' || v === 'valuable' || v === 'supported' || v === 'strong') return 'supported';
    if (v === 'low' || v === 'weak' || v === 'refuted' || v === 'poor') return 'refuted';
    return 'inconclusive';
}

// =============================================================================
// SERVER
// =============================================================================

const lab = createLabServer<CritiqueLabConfig>({
    name: 'critique-lab',
    version: '2.0.0',
    description: 'LLM Critique Lab for Podbit. Evaluates knowledge graph nodes for quality AND reviews experiment methodology for the lab chaining pipeline. No computation — pure LLM analysis.',

    pipeline,
    config,

    buildCapabilities: (cfg) => ({
        specTypes: cfg.capabilities.specTypes,
        environment: {
            runtime: 'LLM-only (no sandbox, no compute)',
            approach: 'Pure LLM analysis — node critique and experiment methodology review.',
            codegenMethod: 'None — no code is generated.',
            evaluationMethod: 'LLM evaluates node quality or experiment methodology depending on spec type.',
        },
        queueLimit: cfg.capabilities.queueLimit,
        artifactTtlSeconds: cfg.capabilities.artifactTtlSeconds,
        features: cfg.capabilities.features,
    }),
});

// Queue UI
let _queueHtml: string | null = null;
lab.app.get('/ui', (_req, res) => {
    if (!_queueHtml) _queueHtml = readFileSync(join(process.cwd(), 'public', 'queue.html'), 'utf-8');
    res.setHeader('Content-Type', 'text/html');
    res.send(_queueHtml);
});

// Job list endpoint
lab.app.get('/jobs', (_req, res) => {
    const jobs = db.listJobs({ limit: 50 });
    res.json(jobs.map(j => ({
        id: j.id,
        status: j.status,
        spec: j.spec ? JSON.parse(j.spec) : null,
        result: j.result ? JSON.parse(j.result) : null,
        created_at: j.created_at,
        completed_at: j.completed_at,
    })));
});

lab.start();
