/**
 * Math Lab Codegen — generates computation code from experiment specs via LLM.
 *
 * The LLM generates ONLY the computation logic. We wrap it in a template
 * that has all math imports and utility functions pre-loaded.
 *
 * Helper sections in the prompt are filtered by spec type to keep prompts
 * compact and avoid truncation on models with limited output budgets.
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { callLlm, getConfig, LabError, TruncatedError, type CodegenResult, type ExperimentSpec } from '@lab/core';

const TEMPLATE_PATH = join(process.cwd(), 'prompts', 'template.py');

let templateCache: string | null = null;
function getTemplate(): string {
    if (!templateCache) templateCache = readFileSync(TEMPLATE_PATH, 'utf-8');
    return templateCache;
}

interface CodegenOptions {
    signal?: AbortSignal;
}

function buildTemplate(computation: string, spec: ExperimentSpec): string {
    const cfg = getConfig();
    const precision = spec.setup?.precision ?? (cfg as any).execution?.defaultPrecision ?? 50;
    return getTemplate()
        .replace('{{precision}}', String(precision))
        .replace('{{computation}}', computation);
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER SECTIONS — each section documents functions available in template.py.
// Only sections relevant to the spec type are included in the prompt.
// ═══════════════════════════════════════════════════════════════════════════

const SECTION_CORE = `MATH HELPER FUNCTIONS:
  measure(label, fn)         — safely compute fn() and store in result[label]; catches errors automatically
  values_close(a, b, rel_tol=1e-9)  — check if two numbers are close
  relative_error(measured, expected) — compute relative error
  hp_eval(expr_str)          — evaluate math expression string with mpmath precision (e.g., hp_eval("pi**2/6"))
  test_convergence(seq_fn, n_terms=1000) — test if sequence_fn(n) converges; returns {limit_estimate, converged, rate}
  partial_sums(term_fn, n)   — compute partial sums of series; term_fn(n) returns n-th term
  analyze_curve(fn, start, end, n=500) — analyze function shape; returns {monotonic, extrema, inflection_points, ...}
  sweep(fn, param_name, values) — sweep parameter; fn(value) returns measurement
  linspace_sweep(fn, name, start, end, n=20) — sweep with linearly spaced values
  sym_to_float(expr)         — convert sympy expression to float
  sym_limit(expr, var, point) — symbolic limit as float
  sym_sum_to_float(term, var, start, end) — symbolic sum as float`;

const SECTION_QUANTUM = `QUANTUM MECHANICS HELPERS (numpy required):
  PAULI_I, PAULI_X, PAULI_Y, PAULI_Z — 2x2 Pauli matrices
  KET_0, KET_1                — qubit basis states (column vectors)
  tensor(*matrices)           — Kronecker product of matrices/vectors
  partial_trace(rho, dims, trace_over) — partial trace of density matrix
  density_matrix(state_vec)   — |psi><psi| from column vector
  expect(operator, rho)       — Tr(operator @ rho)
  commutator(A, B)            — [A, B]
  von_neumann_entropy(rho)    — S = -Tr(rho log2 rho) in bits
  fidelity(rho, sigma)        — quantum fidelity F(rho, sigma)
  bell_state(which)           — Bell state vector: 'phi+', 'phi-', 'psi+', 'psi-'
  chsh_value(rho, a1, a2, b1, b2) — CHSH expectation value from measurement operators
  time_evolve(H, rho, t)      — unitary evolution rho(t) = U rho U†
  lindblad_rhs(rho, H, L_ops, gamma) — Lindblad master equation drho/dt`;

const SECTION_WAVE = `WAVE SYSTEM HELPERS (numpy required):
  tight_binding_1d(N, on_site, hopping, periodic) — build 1D TB Hamiltonian
  band_structure_1d(H_cell, H_hop, k_points) — compute band structure E(k)
  transfer_matrix(omega, layers) — 1D wave transfer matrix from [(n, d), ...]
  transmission_coefficient(M)  — |t|^2 from transfer matrix
  disorder_sweep(build_H_fn, N, strengths, n_real) — robustness to disorder statistics`;

const SECTION_QUANTUM_OPTICS = `QUANTUM OPTICS PROTOCOL HELPERS (numpy required):
  creation_op(n_dim)           — bosonic a† in truncated Fock space
  annihilation_op(n_dim)       — bosonic a in truncated Fock space
  number_op(n_dim)             — number operator n = a†a
  squeeze_operator(n_dim, r, phi) — single-mode squeeze S(z)
  two_mode_squeeze(n_dim, r, phi) — SPDC model: two-mode squeezed vacuum density matrix
  jaynes_cummings_H(omega_q, omega_c, g, n_dim) — circuit-QED / cavity-QED Hamiltonian
  heisenberg_exchange_H(J, N_spins, periodic) — Heisenberg spin-exchange Hamiltonian
  concurrence(rho_2qubit)      — concurrence entanglement measure
  entanglement_of_formation(rho_2qubit) — EoF from concurrence`;

const SECTION_MANY_BODY = `MANY-BODY / CONDENSED MATTER HELPERS (numpy required):
  lattice_2d(Nx, Ny, t_hop, mu, periodic) — 2D square lattice tight-binding Hamiltonian
  lattice_momentum_grid(Nx, Ny) — Brillouin zone (kx, ky) meshgrid
  bdg_hamiltonian(H_normal, Delta) — Bogoliubov-de Gennes Hamiltonian from normal H + gap
  bcs_gap_equation(H_k_fn, V, kgrid, T) — self-consistent BCS gap solver → {gap, converged}
  greens_function_retarded(H, omega, eta) — G^R(omega) = (omega+i*eta-H)^{-1}
  spectral_function(H, omega_array, eta) — A(omega) = -1/pi Im Tr G^R
  spectral_function_k(H_k_fn, kx, ky, omega_array, eta) — momentum-resolved A(k,omega)
  density_of_states(H, omega_array, eta) — DOS = A(omega)/N
  superfluid_weight(H_k_fn, Delta, kgrid, direction) — D_s from BdG eigenvalue derivatives
  self_energy(H_full, H_0, omega, eta) — Sigma = G0^{-1} - G^{-1}
  quasiparticle_weight(H_full, H_0, omega_0, eta) — Z = (1 - dReSigma/domega)^{-1}
  matsubara_greens(H, beta, n_matsubara) — Matsubara Green's function G(i*omega_n)`;

const SECTION_RELATIVISTIC = `RELATIVISTIC KINEMATICS HELPERS (numpy required):
  four_vector(E, px, py, pz)   — create 4-momentum
  mass_from_4vec(p)            — invariant mass sqrt(p·p)
  minkowski_dot(p1, p2)        — Minkowski inner product (+,-,-,-)
  mandelstam_s(p1, p2)         — s = (p1+p2)^2
  mandelstam_t(p1, p3)         — t = (p1-p3)^2
  mandelstam_u(p1, p4)         — u = (p1-p4)^2
  lorentz_boost(bx, by, bz)   — 4x4 Lorentz boost matrix
  boost_4vec(p, bx, by, bz)   — apply boost to 4-vector
  cm_frame_boost(p1, p2)      — boost velocity to CM frame
  two_body_phase_space(sqrt_s, m1, m2) — CM momentum |p*| for 2-body kinematics
  verify_lorentz_invariance(fn, p_list, n_boosts) — test scalar is Lorentz-invariant`;

const SECTION_RG = `RG FLOW / MULTI-SUBSYSTEM / PROTOCOL HELPERS (scipy required):
  polchinski_rg_flow(beta_fn, g_init, t_span) — integrate Polchinski/Wetterstein RG equations
  rg_fixed_points(beta_fn, g_init_list, t_end) — find RG fixed points from multiple initial conditions
  couple_subsystems(H_A, H_B, V_AB, g) — H_total = H_A⊗I + I⊗H_B + g*V_AB
  coupling_sweep(H_A, H_B, V_AB, obs_fn, g_values) — sweep coupling strength, measure observable
  time_ordered_evolution(H_list, t_list, rho0) — sequential Hamiltonian stages
  compare_orderings(H_list, t_list, rho0, obs_fn) — test if temporal ORDER of stages matters
  chern_number_2d(H_k_fn, Nk) — Chern number from discretized Berry curvature
  compare_topological_invariants(H_k_fn_A, H_k_fn_B) — compare Chern numbers of two systems
  randomized_benchmarking(n_qubits, gate_fn, noise_fn, lengths) — simulate RB protocol
  test_causal_chain(step_fns, initial_state, obs_fn) — test multi-step chain, skip each link
  with_and_without(H_base, H_extra, obs_fn) — test whether adding a term changes an observable
  verify_symplectic(omega_matrix) — check if 2-form is antisymmetric + non-degenerate
  critical_exponents(obs_fn, params, p_c, fit_range) — extract power-law exponent near transition`;

const SECTION_COUPLED = `COUPLED DYNAMICS HELPERS (scipy required):
  solve_ode(rhs, y0, t_span, n_points) — integrate dy/dt = rhs(t,y)
  stability_eigenvalues(jac_fn, fixed_pt) — eigenvalues of Jacobian
  find_threshold(param_fn, range, criterion_fn) — binary search for critical parameter
  solve_master_equation(H, rho0, t_span, L_ops, gamma) — integrate Lindblad equation
  wigner_function(rho, xvec, pvec) — Wigner function W(x,p) from Fock-basis density matrix`;

// Map each spec type to the helper sections it needs.
// Unknown spec types get all sections (safe default).
const SPEC_TYPE_SECTIONS: Record<string, string[]> = {
    math:                  ['core'],
    parameter_sweep:       ['core'],
    convergence_analysis:  ['core'],
    curve_shape:           ['core'],
    quantum_model:         ['core', 'quantum', 'quantum_optics', 'coupled'],
    wave_system:           ['core', 'wave', 'rg'],
    coupled_dynamics:      ['core', 'quantum', 'quantum_optics', 'coupled', 'rg'],
    many_body_model:       ['core', 'quantum', 'many_body'],
};

const ALL_SECTIONS: Record<string, string> = {
    core: SECTION_CORE,
    quantum: SECTION_QUANTUM,
    wave: SECTION_WAVE,
    quantum_optics: SECTION_QUANTUM_OPTICS,
    many_body: SECTION_MANY_BODY,
    relativistic: SECTION_RELATIVISTIC,
    rg: SECTION_RG,
    coupled: SECTION_COUPLED,
};

function getHelperSections(specType: string): string {
    const needed = SPEC_TYPE_SECTIONS[specType] || Object.keys(ALL_SECTIONS);
    return needed.map(key => ALL_SECTIONS[key]).filter(Boolean).join('\n\n');
}

// Physics guidance — only include lines relevant to the spec type
const PHYSICS_GUIDANCE: Record<string, string[]> = {
    quantum_model: [
        '- Quantum mechanics: use density matrices (numpy arrays), Hamiltonians, tensor products, partial traces. Use the quantum helpers above.',
        '- Causal ordering: use time_ordered_evolution + compare_orderings to test if stage order matters.',
        '- Multi-subsystem coupling: use couple_subsystems + coupling_sweep to test if coupling produces the claimed effect.',
    ],
    wave_system: [
        '- Wave physics / topological: use tight-binding models, transfer matrices, eigenvalue problems. Use the wave helpers above.',
        '- Topological equivalence: use chern_number_2d + compare_topological_invariants for both systems.',
    ],
    coupled_dynamics: [
        '- Coupled dynamics / optomechanics: use ODE integration, stability analysis. Use the dynamics helpers above.',
        '- Quantum mechanics: use density matrices (numpy arrays), Hamiltonians, tensor products, partial traces. Use the quantum helpers above.',
        '- Causal ordering: use time_ordered_evolution + compare_orderings to test if stage order matters.',
    ],
    many_body_model: [
        '- Many-body / condensed matter: use BdG Hamiltonians, self-consistent BCS gap equations, Green\'s functions, spectral functions. BCS CAN converge to zero gap (falsifiable). Use the many-body helpers above.',
        '- Quantum mechanics: use density matrices (numpy arrays), Hamiltonians, tensor products, partial traces. Use the quantum helpers above.',
    ],
    math: [],
    parameter_sweep: [],
    convergence_analysis: [],
    curve_shape: [],
};

// Performance hints — only include lines relevant to the spec type
function getPerformanceHints(specType: string): string {
    const always = [
        '- **Parameter sweeps**: Limit to ≤ 20 sweep points. Each point may involve a full ODE integration. 5-10 points per parameter is usually enough to detect thresholds.',
        '- **Matrix operations**: Operations on N×N dense matrices cost O(N³). At N=500, one matmul takes ~0.1s. At N=3000, it takes ~30s. Keep N ≤ 500 for any matrix that gets multiplied repeatedly.',
        '- **Vectorize, don\'t loop**: Use numpy broadcasting and vectorized operations instead of Python for-loops over arrays. If you need nested loops over large arrays, use `@njit` from numba.',
        '- **Memory**: A complex128 matrix of size N×N uses 16N² bytes. Keep dense matrices under N=3000. Use scipy.sparse for larger systems.',
        '- **Eigendecomposition in loops**: `np.linalg.eigh(H)` is O(N³). Plan your loop counts × matrix sizes to stay under 60s total.',
        '- **Sympy**: NEVER use sympy for numerical computation — it is 100-1000x slower than numpy. Use numpy/scipy for all numerical work.',
        '- **scipy.optimize**: Always pass `options={\'maxiter\': 500}` or equivalent. Never optimize without bounds on unbounded landscapes.',
    ];

    const quantum = [
        '- **Hilbert space**: Total tensor product dimension MUST stay ≤ 500. For N subsystems of dimension d, total = d^N. Use n_cut = 5-8 for multi-mode quantum systems.',
    ];
    const ode = [
        '- **ODE integration**: For density matrix evolution (Lindblad), the state vector has dim² elements. Keep integration time short (T ≤ 20/κ) and use max_step ≥ 1.0. If dim > 200, use sparse matrices.',
        '- **No bare solve_ivp**: Use the helper `solve_ode(rhs, y0, t_span, n_points)` or `solve_master_equation(H, rho0, t_span, L_ops, gamma)` — they are already imported.',
    ];
    const selfConsistent = [
        '- **Self-consistent loops**: BCS gap equations, mean-field, Hartree-Fock — ALWAYS set max_iter ≤ 200 and check convergence. Use `for i in range(max_iter):` not `while not converged:`.',
    ];
    const kGrid = [
        '- **k-point grids**: For Brillouin zone sampling, use ≤ 50×50 grid (2500 points). Each k-point requires an eigendecomposition.',
    ];

    const extra: string[] = [];
    if (['quantum_model', 'coupled_dynamics', 'many_body_model'].includes(specType)) extra.push(...quantum);
    if (['quantum_model', 'coupled_dynamics'].includes(specType)) extra.push(...ode);
    if (['many_body_model'].includes(specType)) extra.push(...selfConsistent, ...kGrid);

    return [...always, ...extra].join('\n');
}

function getPhysicsGuidance(specType: string): string {
    const lines = PHYSICS_GUIDANCE[specType] || Object.values(PHYSICS_GUIDANCE).flat();
    const common = [
        '- Do NOT say "requires QuTiP/Qiskit/FDTD/DMFT" — these problems reduce to linear algebra and ODEs that numpy/scipy handle.',
        '- A model is NOT tautological if coupling/ordering/threshold parameters can be varied to DISPROVE the claim.',
    ];
    return [...lines, ...common].join('\n');
}

// ═══════════════════════════════════════════════════════════════════════════
// PROMPT BUILDER
// ═══════════════════════════════════════════════════════════════════════════

interface PromptOptions {
    /** If true, add a conciseness instruction (used on truncation retry) */
    concise?: boolean;
}

function buildPrompt(spec: ExperimentSpec, previousAttempts?: Array<{ code: string; error: string }>, opts?: PromptOptions): string {
    const setupDesc = Object.entries(spec.setup)
        .map(([k, v]) => `  ${k}: ${JSON.stringify(v)}`)
        .join('\n');

    let errorFeedback = '';
    if (previousAttempts?.length) {
        // Show the latest attempt in full + the error verbatim. Earlier attempts are summarised
        // (just their error) so the model can see whether it is repeating the same mistake.
        const latest = previousAttempts[previousAttempts.length - 1];
        // Cap the resent code at ~30KB. The LLM's generated computation is appended at the END of
        // the wrapped template (boilerplate/helpers are at the top), so when truncating we keep the
        // TAIL — that's where any error site lives and what needs fixing.
        const latestCode = latest.code.length > 30000
            ? '# ... (template/imports/helpers truncated above — assume standard math-lab template is in scope)\n' + latest.code.slice(latest.code.length - 30000)
            : latest.code;
        const earlier = previousAttempts.slice(0, -1);
        const earlierBlock = earlier.length > 0
            ? earlier.map((a, i) => `Earlier attempt ${i + 1} error: ${a.error.slice(0, 400)}`).join('\n') + '\n\n'
            : '';
        errorFeedback = `\n\nPREVIOUS FAILED ATTEMPTS — your code raised an error. READ THE TRACEBACK CAREFULLY and fix the SPECIFIC issue. Do NOT regenerate from scratch — keep what worked, fix what broke. Common mistakes to check: typos in variable names (e.g. \`disperion\` vs \`dispersion\`), referencing variables before defining them, off-by-one in indices, wrong number of arguments to a helper function.

${earlierBlock}LATEST ATTEMPT (attempt ${previousAttempts.length}) — full code:
\`\`\`python
${latestCode}
\`\`\`

LATEST ATTEMPT ERROR (verbatim traceback):
${latest.error}

YOUR JOB: Identify the exact line/identifier the error points at, fix it, and return the full corrected code. Preserve everything else from the latest attempt.`;
    }

    const cfg = getConfig();
    let specTypeDesc = spec.specType;
    const specTypes = cfg.capabilities?.specTypes;
    if (specTypes && typeof specTypes === 'object' && !Array.isArray(specTypes)) {
        const desc = (specTypes as Record<string, string>)[spec.specType];
        if (desc) specTypeDesc = `${spec.specType} — ${desc}`;
    }

    const precision = spec.hints?.precision ?? spec.setup?.precision ?? (cfg as any).execution?.defaultPrecision ?? 50;
    const helpers = getHelperSections(spec.specType);
    const perfHints = getPerformanceHints(spec.specType);
    const physicsGuide = getPhysicsGuidance(spec.specType);

    const conciseHint = opts?.concise
        ? '\n\nCRITICAL: Your previous response was TRUNCATED because it exceeded the output token limit. Write CONCISE code — avoid verbose variable names, minimize comments, combine setup steps. Focus on the essential computation.'
        : '';

    return `Write ONLY the computation code to test this hypothesis. Do NOT write imports or boilerplate.

EVERYTHING BELOW IS ALREADY IN SCOPE — just use it:

LIBRARIES: math, numpy (as np), scipy (.optimize, .integrate, .special, .stats, .interpolate, .linalg, .signal, .fft, .sparse), sympy (symbols, oo, Sum, Product, limit, series, simplify, expand, pi, E, sqrt), mpmath (mpf, mpc, dps=${precision}), networkx (as nx), numba (njit, prange — falls back to no-op if unavailable), statistics, decimal (Decimal), fractions (Fraction), itertools, functools, collections, operator, re, json, hashlib, random, cmath

${helpers}

RESULT DICT: result = {} is pre-defined. Populate it with your findings.

EXPERIMENT TYPE: ${specTypeDesc}

HYPOTHESIS TO TEST:
${spec.hypothesis}

SETUP:
${setupDesc}
${errorFeedback}${conciseHint}

YOUR TASK:
1. Design measurements that would support or refute the hypothesis
2. Compute them using the setup parameters and available libraries
3. Store results in the result dict using measure("label", lambda: computation)
4. Include a "supported" key: measure("supported", lambda: True/False) based on your measurements

RULES:
- Write top-level code only — no function wrappers, no if __name__ blocks, no imports
- Use the measure() helper for safe error handling
- Values must be numbers, lists, or booleans
- No file I/O, no network, no os/sys/subprocess
- ONLY use functions listed in the HELPER FUNCTIONS sections above or from the LIBRARIES listed above
- Do NOT call functions that don't exist — there is no simulate_*, compute_*, run_*, or test_* function unless you define it inline
- If you need a computation, write it directly using numpy/scipy operations — do not reference phantom helper functions
- You MAY define small local helper functions (def my_fn(...): ...) at the top of your code if needed for clarity
${precision > 50 ? `- Use mpmath with mp.dps = ${precision} for high-precision arithmetic` : ''}

PERFORMANCE — HARD LIMITS (violations will timeout and waste compute):
${perfHints}

PHYSICS CLAIMS — reduce to math:
${physicsGuide}

Respond with JSON: {"code": "your computation code here"}`;
}

// ═══════════════════════════════════════════════════════════════════════════
// RESPONSE PARSING + TRUNCATION RECOVERY
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Try all standard parsers to extract computation code from an LLM response.
 *
 * Codegen LLMs are wildly inconsistent about output format — sometimes raw JSON,
 * sometimes JSON inside a ```json fence, sometimes raw Python inside a ```python
 * fence, sometimes JSON-inside-a-```python-fence (the worst), sometimes plain
 * Python with no fence at all. Each step below catches one of those variants.
 *
 * If a fenced block is found, we ALWAYS attempt JSON-extraction on the body
 * before treating the body as raw code — this is the fix for the
 * "```python {"code": "..."}``` " case the LLM occasionally emits.
 */
function parseComputation(rawResponse: string): string {
    const tryExtractJsonCode = (text: string): string => {
        const trimmed = text.trim();
        try {
            const parsed = JSON.parse(trimmed);
            if (parsed && typeof parsed.code === 'string') return parsed.code;
        } catch { /* not whole-text JSON */ }
        // Try to find an enclosed { ... "code": "..." ... } object inside text
        const objMatch = matchBalancedObject(trimmed);
        if (objMatch) {
            try {
                const parsed = JSON.parse(objMatch);
                if (parsed && typeof parsed.code === 'string') return parsed.code;
            } catch { /* fall through */ }
        }
        return '';
    };

    // 1. Whole-response JSON
    const fromWhole = tryExtractJsonCode(rawResponse);
    if (fromWhole) return fromWhole;

    // 2. ```json fenced block
    const jsonBlock = rawResponse.match(/```json\s*([\s\S]*?)```/);
    if (jsonBlock) {
        const fromJsonBlock = tryExtractJsonCode(jsonBlock[1]);
        if (fromJsonBlock) return fromJsonBlock;
    }

    // 3. ```python (or ```py) fenced block — body may be raw Python OR a
    //    JSON object that the LLM mislabelled as python. Try JSON first; if
    //    that fails, treat the body as raw Python.
    const pyBlock = rawResponse.match(/```(?:python|py)\s*\n?([\s\S]*?)```/);
    if (pyBlock) {
        const fromPyAsJson = tryExtractJsonCode(pyBlock[1]);
        if (fromPyAsJson) return fromPyAsJson;
        return pyBlock[1].trim();
    }

    // 4. Any ``` block — same JSON-first, raw-after fallback
    const anyBlock = rawResponse.match(/```[a-z]*\s*\n?([\s\S]*?)```/);
    if (anyBlock) {
        const fromAnyAsJson = tryExtractJsonCode(anyBlock[1]);
        if (fromAnyAsJson) return fromAnyAsJson;
        return anyBlock[1].trim();
    }

    return '';
}

/**
 * Find the first balanced top-level {…} object inside `text`. Walks character
 * by character respecting JSON string escapes. Returns null if no balanced
 * object is found. This replaces the old `\{[\s\S]*\}` greedy match, which
 * grabbed everything up to the LAST `}` in the response — wrong when the
 * response had multiple braces or a brace inside Python code.
 */
function matchBalancedObject(text: string): string | null {
    const start = text.indexOf('{');
    if (start === -1) return null;
    let depth = 0;
    let inString = false;
    let escape = false;
    for (let i = start; i < text.length; i++) {
        const c = text[i];
        if (inString) {
            if (escape) { escape = false; continue; }
            if (c === '\\') { escape = true; continue; }
            if (c === '"') inString = false;
            continue;
        }
        if (c === '"') { inString = true; continue; }
        if (c === '{') depth++;
        else if (c === '}') {
            depth--;
            if (depth === 0) return text.slice(start, i + 1);
        }
    }
    return null;
}

/**
 * Attempt to recover Python code from a truncated JSON response.
 *
 * When the LLM's output is cut off, the JSON is never closed, so standard
 * parsers fail. This locates the `"code": "...` field and rebuilds the
 * Python source by reversing JSON string escapes.
 *
 * IMPORTANT: escape order matters. `\\` (literal backslash) MUST be unescaped
 * AFTER the meaningful escapes (`\n`, `\t`, `\"`) — otherwise `\\n` in the wire
 * data first becomes `\n` (single backslash + n), which the next pass would
 * mistakenly turn into a newline. We do a single forward pass instead so the
 * order is unambiguous and we never re-process produced characters.
 */
function recoverTruncatedCode(rawResponse: string): string {
    const codeMatch = rawResponse.match(/"code"\s*:\s*"/);
    if (!codeMatch || codeMatch.index === undefined) return '';

    let raw = rawResponse.slice(codeMatch.index + codeMatch[0].length);
    // Strip a trailing closing quote + brace if the response is actually complete
    raw = raw.replace(/"\s*\}\s*`*\s*$/, '');

    // Single forward pass — avoids the double-pass ordering bug.
    let out = '';
    for (let i = 0; i < raw.length; i++) {
        const c = raw[i];
        if (c !== '\\') { out += c; continue; }
        const next = raw[i + 1];
        if (next === undefined) break; // dangling backslash → drop it
        switch (next) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case '"': out += '"'; break;
            case "'": out += "'"; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'u': {
                // \uXXXX — decode 4 hex digits if present
                const hex = raw.slice(i + 2, i + 6);
                if (/^[0-9a-fA-F]{4}$/.test(hex)) {
                    out += String.fromCharCode(parseInt(hex, 16));
                    i += 4;
                } else {
                    out += next;
                }
                break;
            }
            default: out += next;
        }
        i++; // skip the escaped char
    }

    // Drop trailing whitespace lines, but ONLY drop the very last line if it
    // looks unfinished (no terminator, dangling open quote/bracket). The old
    // code unconditionally popped the last line, which often killed perfectly
    // good single-line completions and pushed valid recoveries below the
    // codegen length threshold.
    const lines = out.split('\n');
    while (lines.length > 0 && lines[lines.length - 1].trim() === '') lines.pop();
    if (lines.length > 1) {
        const last = lines[lines.length - 1];
        const looksTruncated =
            // open string that never closes on this line
            (last.match(/(?<!\\)"/g)?.length ?? 0) % 2 === 1
            // line ends mid-identifier or mid-operator
            || /[+\-*\/=<>%&|^,(\[{]\s*$/.test(last)
            // bare backslash continuation
            || /\\$/.test(last);
        if (looksTruncated) lines.pop();
    }

    return lines.join('\n').trim();
}

// ═══════════════════════════════════════════════════════════════════════════
// CODE GENERATION
// ═══════════════════════════════════════════════════════════════════════════

export async function generateCode(
    spec: ExperimentSpec,
    previousAttempts?: Array<{ code: string; error: string }>,
    options?: CodegenOptions,
): Promise<CodegenResult> {
    const prompt = buildPrompt(spec, previousAttempts);

    let rawResponse: string;
    try {
        rawResponse = await callLlm(prompt, {
            role: 'codegen',
            jsonSchema: { name: 'codegen', schema: {} },
            signal: options?.signal,
        });
    } catch (err: unknown) {
        if (err instanceof TruncatedError) {
            // Attempt 1: recover usable code from the truncated response
            const recovered = recoverTruncatedCode(err.partialContent);
            if (recovered && recovered.length > 50 && /measure\(|result\[/.test(recovered)) {
                const code = buildTemplate(recovered, spec);
                return { code, prompt, rawResponse: err.partialContent };
            }

            // Attempt 2: retry with a compact prompt (conciseness hint + filtered helpers)
            const compactPrompt = buildPrompt(spec, previousAttempts, { concise: true });
            try {
                rawResponse = await callLlm(compactPrompt, {
                    role: 'codegen',
                    jsonSchema: { name: 'codegen', schema: {} },
                    signal: options?.signal,
                });
            } catch (retryErr: unknown) {
                if (retryErr instanceof TruncatedError) {
                    // Still truncated — try recovery from compact attempt
                    const recovered2 = recoverTruncatedCode(retryErr.partialContent);
                    if (recovered2 && recovered2.length > 50) {
                        const code = buildTemplate(recovered2, spec);
                        return { code, prompt: compactPrompt, rawResponse: retryErr.partialContent };
                    }
                    // Give up with a clear message
                    throw new LabError(
                        `Codegen truncated twice. Model output limit too low for this experiment. Recovered ${recovered2.length} chars but insufficient. Raw (first 500 chars): ${retryErr.partialContent.slice(0, 500)}`,
                        'llm', false,
                    );
                }
                throw retryErr;
            }
        } else {
            throw err;
        }
    }

    // Normal parsing flow
    let computation = parseComputation(rawResponse);

    // Fallback: try truncation recovery even on non-TruncatedError responses
    // (some providers don't report finish_reason: length)
    if (!computation) {
        computation = recoverTruncatedCode(rawResponse);
    }

    if (!computation || computation.trim().length < 10) {
        throw new LabError(
            `Codegen produced empty or trivial code. Raw response (first 500 chars): ${rawResponse.slice(0, 500)}`,
            'llm', true,
        );
    }

    const code = buildTemplate(computation, spec);
    return { code, prompt, rawResponse };
}
