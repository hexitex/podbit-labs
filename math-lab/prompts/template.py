# === MATH-LAB EXECUTION TEMPLATE ===
# All imports and utilities are pre-loaded. The LLM-generated computation
# code is inserted at the bottom. It just populates the `result` dict.

# ─── Resilient imports — missing packages don't crash the lab ─────────────────
import math
import json
import cmath
import statistics
import decimal
from decimal import Decimal
import fractions
from fractions import Fraction
import itertools
import functools
import collections
import operator
import re
import hashlib
import random

def _try_import(name, fromlist=None):
    """Import a module, returning None if unavailable."""
    try:
        if fromlist:
            return __import__(name, fromlist=fromlist)
        # For dotted names like 'scipy.integrate', __import__ returns the
        # top-level package unless fromlist is set. Use fromlist=[''] to
        # get the actual submodule.
        if '.' in name:
            return __import__(name, fromlist=[''])
        return __import__(name)
    except ImportError:
        return None

np = _try_import('numpy')
scipy = _try_import('scipy')
if scipy:
    scipy.optimize = _try_import('scipy.optimize')
    scipy.integrate = _try_import('scipy.integrate')
    scipy.special = _try_import('scipy.special')
    scipy.stats = _try_import('scipy.stats')
    scipy.interpolate = _try_import('scipy.interpolate')
    scipy.linalg = _try_import('scipy.linalg')
    scipy.signal = _try_import('scipy.signal')
    scipy.fft = _try_import('scipy.fft')
    scipy.sparse = _try_import('scipy.sparse')
    if scipy.sparse:
        scipy.sparse.linalg = _try_import('scipy.sparse.linalg')
    scipy.spatial = _try_import('scipy.spatial')

    # Compatibility aliases — LLM code often writes these wrong
    if scipy.integrate:
        scipy.solve_ivp = scipy.integrate.solve_ivp
        scipy.odeint = scipy.integrate.odeint
        scipy.quad = scipy.integrate.quad
        scipy.dblquad = scipy.integrate.dblquad
    if scipy.optimize:
        scipy.minimize = scipy.optimize.minimize
        scipy.root = scipy.optimize.root
        scipy.curve_fit = scipy.optimize.curve_fit
    if scipy.linalg:
        scipy.expm = scipy.linalg.expm
        scipy.sqrtm = scipy.linalg.sqrtm
        scipy.eigh = scipy.linalg.eigh
    if scipy.special:
        scipy.jv = scipy.special.jv
        scipy.spherical_jn = getattr(scipy.special, 'spherical_jn', None)

nx = _try_import('networkx')

numba = _try_import('numba')
if numba:
    from numba import njit, prange
else:
    # Fallback: njit is a no-op decorator, prange is just range
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f
    prange = range

sympy = _try_import('sympy')
if sympy:
    from sympy import symbols, oo, Sum, Product, limit, series, simplify, expand
    from sympy import pi as sym_pi, E as sym_E, sqrt as sym_sqrt, Rational, Integer
    from sympy import sin as sym_sin, cos as sym_cos, log as sym_log, exp as sym_exp
    from sympy import integrate as sym_integrate, diff as sym_diff, solve as sym_solve

mpmath = _try_import('mpmath')
if mpmath:
    from mpmath import mpf, mpc, inf as mp_inf

# Track what's available so the LLM computation can check
_available = {
    'numpy': np is not None,
    'scipy': scipy is not None,
    'sympy': sympy is not None,
    'mpmath': mpmath is not None,
    'networkx': nx is not None,
    'numba': numba is not None,
}

# ─── Precision setup ─────────────────────────────────────────────────────────
if mpmath:
    mpmath.mp.dps = {{precision}}

# ─── Result dict — computation code populates this ───────────────────────────
result = {}

# ─── Utility: safe measurement ───────────────────────────────────────────────
def measure(label, fn):
    """Safely compute a measurement. On error, store the error in result."""
    try:
        val = fn()
        result[label] = _to_json_safe(val)
    except Exception as e:
        result[label] = {"error": str(e)}

def _to_json_safe(val):
    """Convert any numeric type to JSON-serializable Python type."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float, str)):
        return val
    if mpmath and isinstance(val, mpmath.mpf):
        return float(val)
    if mpmath and isinstance(val, mpmath.mpc):
        return {"real": float(val.real), "imag": float(val.imag)}
    if sympy and isinstance(val, (sympy.Float, sympy.Integer, sympy.Rational)):
        return float(val)
    if sympy and isinstance(val, sympy.Basic):
        # Try to convert to float, fall back to string
        try:
            return float(sympy.N(val))
        except (TypeError, ValueError):
            return str(val)
    if np is not None and isinstance(val, np.ndarray):
        return val.tolist()
    if np is not None and isinstance(val, (np.integer,)):
        return int(val)
    if np is not None and isinstance(val, (np.floating,)):
        return float(val)
    if np is not None and isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, Fraction):
        return float(val)
    if isinstance(val, complex):
        return {"real": val.real, "imag": val.imag}
    if isinstance(val, dict):
        return {str(k): _to_json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_json_safe(v) for v in val]
    return str(val)

# ─── Utility: numerical comparison ───────────────────────────────────────────
def values_close(a, b, rel_tol=1e-9, abs_tol=1e-15):
    """Check if two values are close, handling edge cases."""
    a, b = float(a), float(b)
    if a == b:
        return True
    if abs(b) > abs_tol:
        return abs(a - b) / abs(b) < rel_tol
    return abs(a - b) < abs_tol

def relative_error(measured, expected):
    """Compute relative error, handling zero expected."""
    measured, expected = float(measured), float(expected)
    if expected == 0:
        return abs(measured)
    return abs(measured - expected) / abs(expected)

# ─── Utility: high-precision evaluation ──────────────────────────────────────
def hp_eval(expr_str):
    """Evaluate a mathematical expression string with mpmath precision."""
    if not mpmath:
        raise RuntimeError("mpmath not available")
    ns = {
        'pi': mpmath.pi, 'e': mpmath.e, 'inf': mpmath.inf,
        'sqrt': mpmath.sqrt, 'cbrt': mpmath.cbrt, 'log': mpmath.log, 'ln': mpmath.log,
        'log10': mpmath.log10, 'log2': lambda x: mpmath.log(x)/mpmath.log(2),
        'exp': mpmath.exp, 'sin': mpmath.sin, 'cos': mpmath.cos,
        'tan': mpmath.tan, 'asin': mpmath.asin, 'acos': mpmath.acos,
        'atan': mpmath.atan, 'atan2': mpmath.atan2,
        'sinh': mpmath.sinh, 'cosh': mpmath.cosh, 'tanh': mpmath.tanh,
        'gamma': mpmath.gamma, 'zeta': mpmath.zeta, 'factorial': mpmath.factorial,
        'binomial': mpmath.binomial, 'fsum': mpmath.fsum, 'power': mpmath.power,
        'mpf': mpmath.mpf, 'fib': mpmath.fibonacci,
        'abs': abs, 'sum': sum, 'range': range, 'float': float, 'int': int,
        'max': max, 'min': min,
    }
    return mpmath.mpf(eval(expr_str, {"__builtins__": {}}, ns))

# ─── Utility: convergence testing ────────────────────────────────────────────
def test_convergence(sequence_fn, n_terms=1000, window=50):
    """
    Test if a sequence converges.
    sequence_fn(n) → n-th term or n-th partial sum.
    Returns: {limit_estimate, converged, stdev, rate, last_values}
    """
    values = [float(sequence_fn(n)) for n in range(1, n_terms + 1)]
    tail = values[-window:]
    mean_tail = statistics.mean(tail)
    stdev_tail = statistics.stdev(tail) if len(tail) > 1 else 0.0

    converged = stdev_tail < 1e-8 * (abs(mean_tail) + 1e-15)

    rate = None
    if len(values) > 10:
        diffs = [abs(values[i] - values[i-1]) for i in range(-10, 0)]
        if diffs[0] > 1e-15:
            ratios = [diffs[i+1] / diffs[i] for i in range(len(diffs)-1) if diffs[i] > 1e-15]
            if ratios:
                rate = statistics.mean(ratios)

    return {
        "limit_estimate": mean_tail,
        "converged": converged,
        "stdev": stdev_tail,
        "rate": rate,
        "last_values": tail[-5:],
    }

def partial_sums(term_fn, n_terms=1000):
    """Compute partial sums of a series. term_fn(n) → n-th term (0-indexed)."""
    if mpmath:
        s = mpmath.mpf(0)
        sums = []
        for n in range(n_terms):
            s += term_fn(n)
            sums.append(float(s))
        return sums
    else:
        s = 0.0
        sums = []
        for n in range(n_terms):
            s += float(term_fn(n))
            sums.append(s)
        return sums

# ─── Utility: curve analysis ─────────────────────────────────────────────────
def analyze_curve(fn, domain_start, domain_end, n_points=500):
    """
    Analyze a function over a domain.
    fn(x) → float.
    Returns: {monotonic, direction, concave_up, concave_down, has_inflection,
              inflection_points, extrema, min_y, max_y, y_at_start, y_at_end}
    """
    if np is None:
        raise RuntimeError("numpy not available for curve analysis")

    xs = np.linspace(float(domain_start), float(domain_end), n_points)
    ys = np.array([float(fn(x)) for x in xs])

    dy = np.diff(ys)
    ddy = np.diff(dy)

    all_increasing = np.all(dy >= -1e-10)
    all_decreasing = np.all(dy <= 1e-10)
    monotonic = bool(all_increasing or all_decreasing)
    direction = "increasing" if all_increasing else ("decreasing" if all_decreasing else "non-monotonic")

    concave_up = bool(np.all(ddy >= -1e-10))
    concave_down = bool(np.all(ddy <= 1e-10))

    inflections = []
    for i in range(len(ddy) - 1):
        if ddy[i] * ddy[i+1] < 0:
            inflections.append(float(xs[i+1]))

    extrema = []
    for i in range(len(dy) - 1):
        if dy[i] * dy[i+1] < 0:
            extrema.append({
                "x": float(xs[i+1]),
                "y": float(ys[i+1]),
                "type": "maximum" if dy[i] > 0 else "minimum"
            })

    return {
        "monotonic": monotonic,
        "direction": direction,
        "concave_up": concave_up,
        "concave_down": concave_down,
        "has_inflection": len(inflections) > 0,
        "inflection_points": inflections[:10],
        "extrema": extrema[:10],
        "min_y": float(np.min(ys)),
        "max_y": float(np.max(ys)),
        "y_at_start": float(ys[0]),
        "y_at_end": float(ys[-1]),
    }

# ─── Utility: parameter sweep ────────────────────────────────────────────────
def sweep(fn, param_name, values):
    """
    Sweep a parameter across values.
    fn(value) → measurement (number or dict).
    Returns list of {param_name: value, "result": measurement}.
    """
    results_list = []
    for v in values:
        try:
            r = _to_json_safe(fn(v))
            results_list.append({param_name: _to_json_safe(v), "result": r})
        except Exception as e:
            results_list.append({param_name: _to_json_safe(v), "error": str(e)})
    return results_list

def linspace_sweep(fn, param_name, start, end, n_points=20):
    """Convenience: sweep linearly spaced values."""
    if np is None:
        values = [start + (end - start) * i / (n_points - 1) for i in range(n_points)]
    else:
        values = np.linspace(float(start), float(end), n_points)
    return sweep(fn, param_name, values)

# ─── Utility: symbolic helpers ───────────────────────────────────────────────
def sym_to_float(expr):
    """Convert a sympy expression to float."""
    if not sympy:
        raise RuntimeError("sympy not available")
    dps = mpmath.mp.dps if mpmath else 15
    return float(sympy.N(expr, dps))

def sym_limit(expr, var, point):
    """Compute symbolic limit and return as float."""
    return sym_to_float(sympy.limit(expr, var, point))

def sym_sum_to_float(term_expr, var, start, end):
    """Evaluate a symbolic sum and return as float."""
    return sym_to_float(sympy.Sum(term_expr, (var, start, end)).doit())

# ─── Utility: quantum mechanics (Hilbert space linear algebra) ────────────────
if np is not None:
    # Pauli matrices
    PAULI_I = np.eye(2, dtype=complex)
    PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
    PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Common states
    KET_0 = np.array([[1], [0]], dtype=complex)
    KET_1 = np.array([[0], [1]], dtype=complex)

    def tensor(*matrices):
        """Tensor (Kronecker) product of multiple matrices/vectors."""
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result

    def partial_trace(rho, dims, trace_over):
        """
        Partial trace of density matrix rho over specified subsystems.
        dims: list of subsystem dimensions, e.g. [2, 2] for two qubits.
        trace_over: index (or list of indices) of subsystem(s) to trace out.
        """
        if isinstance(trace_over, int):
            trace_over = [trace_over]
        n = len(dims)
        rho_t = rho.reshape(dims + dims)
        # Trace from highest index down so indices stay valid
        for idx in sorted(trace_over, reverse=True):
            rho_t = np.trace(rho_t, axis1=idx, axis2=idx + n)
            n -= 1
            dims = dims[:idx] + dims[idx+1:]
        remaining_dim = int(np.prod(dims))
        return rho_t.reshape(remaining_dim, remaining_dim)

    def density_matrix(state_vec):
        """Outer product |psi><psi| from a column vector."""
        psi = np.asarray(state_vec, dtype=complex).reshape(-1, 1)
        return psi @ psi.conj().T

    def expect(operator, rho):
        """Expectation value Tr(operator @ rho)."""
        return np.real(np.trace(operator @ rho))

    def commutator(A, B):
        """[A, B] = AB - BA."""
        return A @ B - B @ A

    def von_neumann_entropy(rho):
        """Von Neumann entropy S = -Tr(rho log rho)."""
        eigenvalues = np.real(np.linalg.eigvalsh(rho))
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def fidelity(rho, sigma):
        """Quantum fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
        if scipy:
            sqrt_rho = scipy.linalg.sqrtm(rho)
            product = sqrt_rho @ sigma @ sqrt_rho
            sqrt_product = scipy.linalg.sqrtm(product)
        else:
            sqrt_rho = _matrix_exp(0.5 * np.log(rho + 1e-15 * np.eye(rho.shape[0])))
            product = sqrt_rho @ sigma @ sqrt_rho
            sqrt_product = _matrix_exp(0.5 * np.log(product + 1e-15 * np.eye(product.shape[0])))
        return float(np.real(np.trace(sqrt_product)) ** 2)

    def bell_state(which='phi+'):
        """Return a 2-qubit Bell state vector. which: 'phi+','phi-','psi+','psi-'."""
        s = 1/np.sqrt(2)
        states = {
            'phi+': s * (tensor(KET_0, KET_0) + tensor(KET_1, KET_1)),
            'phi-': s * (tensor(KET_0, KET_0) - tensor(KET_1, KET_1)),
            'psi+': s * (tensor(KET_0, KET_1) + tensor(KET_1, KET_0)),
            'psi-': s * (tensor(KET_0, KET_1) - tensor(KET_1, KET_0)),
        }
        return states[which]

    def chsh_value(rho, a1, a2, b1, b2):
        """
        Compute CHSH value <a1*b1> + <a1*b2> + <a2*b1> - <a2*b2>.
        a1, a2: 2x2 Hermitian operators (Alice's measurements).
        b1, b2: 2x2 Hermitian operators (Bob's measurements).
        rho: 4x4 density matrix of the 2-qubit state.
        """
        I2 = np.eye(2, dtype=complex)
        E = lambda A, B: expect(tensor(A, B), rho)
        return E(a1, b1) + E(a1, b2) + E(a2, b1) - E(a2, b2)

    def time_evolve(H, rho, t):
        """Time-evolve density matrix: rho(t) = U rho U†, U = exp(-iHt)."""
        U = scipy.linalg.expm(-1j * H * t) if scipy else _matrix_exp(-1j * H * t)
        return U @ rho @ U.conj().T

    def _matrix_exp(M):
        """Fallback matrix exponential via eigendecomposition."""
        eigvals, eigvecs = np.linalg.eig(M)
        return (eigvecs * np.exp(eigvals)) @ np.linalg.inv(eigvecs)

    def lindblad_rhs(rho, H, L_ops, gamma):
        """
        Right-hand side of Lindblad master equation:
        drho/dt = -i[H, rho] + sum_k gamma_k (L_k rho L_k† - 0.5 {L_k†L_k, rho})
        L_ops: list of Lindblad operators, gamma: list of rates.
        """
        drho = -1j * commutator(H, rho)
        for L, g in zip(L_ops, gamma):
            Ld = L.conj().T
            LdL = Ld @ L
            drho += g * (L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL))
        return drho

    # --- Quantum optics protocol builders ---

    def creation_op(n_dim):
        """Bosonic creation operator a† in Fock space truncated to n_dim."""
        a_dag = np.zeros((n_dim, n_dim), dtype=complex)
        for i in range(n_dim - 1):
            a_dag[i+1, i] = np.sqrt(i + 1)
        return a_dag

    def annihilation_op(n_dim):
        """Bosonic annihilation operator a in Fock space truncated to n_dim."""
        return creation_op(n_dim).conj().T

    def number_op(n_dim):
        """Number operator n = a†a in Fock space."""
        a_dag = creation_op(n_dim)
        return a_dag @ a_dag.conj().T

    def squeeze_operator(n_dim, r, phi=0):
        """Single-mode squeeze operator S(z) with z = r*exp(i*phi), truncated Fock space."""
        a = annihilation_op(n_dim)
        a_dag = creation_op(n_dim)
        z = r * np.exp(1j * phi)
        arg = 0.5 * (np.conj(z) * a @ a - z * a_dag @ a_dag)
        return scipy.linalg.expm(arg) if scipy else _matrix_exp(arg)

    def two_mode_squeeze(n_dim, r, phi=0):
        """
        Two-mode squeezed vacuum state (SPDC model).
        Returns density matrix in product Fock space (n_dim^2 x n_dim^2).
        """
        # |TMSV> = sum_n tanh(r)^n / cosh(r) |n,n>
        coeffs = np.array([np.tanh(r)**n / np.cosh(r) for n in range(n_dim)])
        psi = np.zeros(n_dim * n_dim, dtype=complex)
        for n in range(n_dim):
            psi[n * n_dim + n] = coeffs[n] * np.exp(1j * n * phi)
        psi = psi / np.linalg.norm(psi)
        return density_matrix(psi)

    def jaynes_cummings_H(omega_q, omega_c, g, n_dim):
        """
        Jaynes-Cummings Hamiltonian (circuit-QED / cavity-QED model).
        omega_q: qubit frequency, omega_c: cavity frequency, g: coupling strength.
        n_dim: Fock space truncation for cavity mode.
        Returns (2*n_dim) x (2*n_dim) Hamiltonian.
        """
        I_c = np.eye(n_dim, dtype=complex)
        I_q = np.eye(2, dtype=complex)
        a = annihilation_op(n_dim)
        a_dag = creation_op(n_dim)
        sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
        sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
        H = (omega_c * tensor(I_q, a_dag @ a) +
             omega_q / 2 * tensor(PAULI_Z, I_c) +
             g * (tensor(sigma_plus, a) + tensor(sigma_minus, a_dag)))
        return H

    def heisenberg_exchange_H(J, N_spins, periodic=False):
        """
        Heisenberg spin-exchange Hamiltonian for N spin-1/2 particles.
        H = J * sum_i (S_i · S_{i+1}) = J * sum_i (X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1}) / 4
        """
        dim = 2**N_spins
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(N_spins - 1 + (1 if periodic else 0)):
            j = (i + 1) % N_spins
            for P in [PAULI_X, PAULI_Y, PAULI_Z]:
                ops = [PAULI_I] * N_spins
                ops[i] = P
                ops[j] = P
                term = ops[0]
                for op in ops[1:]:
                    term = np.kron(term, op)
                H += J / 4 * term
        return H

    def concurrence(rho_2qubit):
        """Concurrence of a 2-qubit density matrix (entanglement measure)."""
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sy_sy = np.kron(sy, sy)
        rho_tilde = sy_sy @ rho_2qubit.conj() @ sy_sy
        product = rho_2qubit @ rho_tilde
        # Use eig (not eigvalsh) since product is not guaranteed Hermitian
        eigvals_raw = np.linalg.eigvals(product)
        eigvals = np.sort(np.real(np.sqrt(np.maximum(np.real(eigvals_raw), 0))))[::-1]
        return float(max(0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3]))

    def entanglement_of_formation(rho_2qubit):
        """Entanglement of formation for a 2-qubit state."""
        C = concurrence(rho_2qubit)
        if C < 1e-15:
            return 0.0
        x = (1 + np.sqrt(1 - C**2)) / 2
        return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))

# ─── Utility: relativistic kinematics (4-vectors, Lorentz, phase space) ──────
if np is not None:
    def four_vector(E, px, py=0, pz=0):
        """Create a 4-momentum vector (E, px, py, pz)."""
        return np.array([E, px, py, pz], dtype=float)

    def mass_from_4vec(p):
        """Invariant mass from 4-vector: m^2 = E^2 - |p|^2."""
        return np.sqrt(max(0, p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2))

    def minkowski_dot(p1, p2):
        """Minkowski inner product with (+,-,-,-) signature."""
        return p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3]

    def mandelstam_s(p1, p2):
        """Mandelstam s = (p1 + p2)^2."""
        total = p1 + p2
        return float(minkowski_dot(total, total))

    def mandelstam_t(p1, p3):
        """Mandelstam t = (p1 - p3)^2."""
        diff = p1 - p3
        return float(minkowski_dot(diff, diff))

    def mandelstam_u(p1, p4):
        """Mandelstam u = (p1 - p4)^2."""
        diff = p1 - p4
        return float(minkowski_dot(diff, diff))

    def lorentz_boost(beta_x=0, beta_y=0, beta_z=0):
        """
        General Lorentz boost matrix (4x4).
        beta_x, beta_y, beta_z: velocity components in units of c.
        """
        beta = np.array([beta_x, beta_y, beta_z])
        beta_sq = np.dot(beta, beta)
        if beta_sq < 1e-30:
            return np.eye(4)
        gamma = 1.0 / np.sqrt(1 - beta_sq)
        L = np.eye(4)
        L[0, 0] = gamma
        L[0, 1:] = -gamma * beta
        L[1:, 0] = -gamma * beta
        L[1:, 1:] += (gamma - 1) * np.outer(beta, beta) / beta_sq
        return L

    def boost_4vec(p, beta_x=0, beta_y=0, beta_z=0):
        """Apply Lorentz boost to a 4-vector."""
        L = lorentz_boost(beta_x, beta_y, beta_z)
        return L @ p

    def cm_frame_boost(p1, p2):
        """Compute the boost velocity to the center-of-mass frame of p1, p2."""
        total = p1 + p2
        E_total = total[0]
        if E_total < 1e-30:
            return np.zeros(3)
        return total[1:4] / E_total

    def two_body_phase_space(sqrt_s, m1, m2):
        """
        Two-body phase space: momentum magnitude in CM frame.
        Returns |p*| for a -> 1 + 2 decay/scattering with invariant mass sqrt_s.
        Returns 0 if below threshold.
        """
        s = sqrt_s**2
        if sqrt_s < m1 + m2:
            return 0.0
        return np.sqrt(max(0, (s - (m1+m2)**2) * (s - (m1-m2)**2))) / (2 * sqrt_s)

    def verify_lorentz_invariance(quantity_fn, p_list, n_boosts=20):
        """
        Test that a scalar quantity is Lorentz-invariant by computing it
        in the original frame and in n_boosts random boosted frames.
        quantity_fn(p_list): computes the scalar from a list of 4-vectors.
        Returns {original, boosted_values, max_deviation, is_invariant}.
        """
        original = quantity_fn(p_list)
        deviations = []
        for _ in range(n_boosts):
            beta = np.random.uniform(-0.9, 0.9, 3)
            beta *= np.random.uniform(0, 0.95) / (np.linalg.norm(beta) + 1e-15)
            L = lorentz_boost(*beta)
            boosted = [L @ p for p in p_list]
            val = quantity_fn(boosted)
            deviations.append(abs(val - original) / (abs(original) + 1e-15))
        max_dev = max(deviations)
        return {
            "original": float(original),
            "max_relative_deviation": float(max_dev),
            "is_invariant": max_dev < 1e-10,
        }

# ─── Utility: many-body physics (BdG, Green's functions, BCS, spectral) ─────
if np is not None:
    def lattice_2d(Nx, Ny, t_hop=1.0, mu=0.0, periodic=True):
        """
        Build 2D tight-binding Hamiltonian on a square lattice.
        Returns (N x N) Hamiltonian where N = Nx * Ny.
        """
        N = Nx * Ny
        H = np.zeros((N, N), dtype=complex)
        for ix in range(Nx):
            for iy in range(Ny):
                i = ix * Ny + iy
                H[i, i] = -mu
                # x-neighbor
                jx = ((ix + 1) % Nx) if periodic else (ix + 1)
                if jx < Nx:
                    j = jx * Ny + iy
                    H[i, j] -= t_hop
                    H[j, i] -= t_hop
                # y-neighbor
                jy = ((iy + 1) % Ny) if periodic else (iy + 1)
                if jy < Ny:
                    j = ix * Ny + jy
                    H[i, j] -= t_hop
                    H[j, i] -= t_hop
        return H

    def lattice_momentum_grid(Nx, Ny):
        """Generate 2D Brillouin zone momentum grid. Returns (kx_grid, ky_grid)."""
        kx = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
        ky = np.linspace(-np.pi, np.pi, Ny, endpoint=False)
        return np.meshgrid(kx, ky, indexing='ij')

    def bdg_hamiltonian(H_normal, Delta):
        """
        Build Bogoliubov-de Gennes (BdG) Hamiltonian from normal-state
        Hamiltonian and superconducting gap matrix.
        H_normal: N x N normal-state Hamiltonian.
        Delta: N x N gap matrix (or scalar for uniform s-wave).
        Returns 2N x 2N BdG Hamiltonian: [[H, Delta], [Delta†, -H*]].
        """
        N = H_normal.shape[0]
        if np.isscalar(Delta):
            Delta = Delta * np.eye(N, dtype=complex)
        Delta = np.asarray(Delta, dtype=complex)
        H_bdg = np.zeros((2*N, 2*N), dtype=complex)
        H_bdg[:N, :N] = H_normal
        H_bdg[:N, N:] = Delta
        H_bdg[N:, :N] = Delta.conj().T
        H_bdg[N:, N:] = -H_normal.conj()
        return H_bdg

    def bcs_gap_equation(H_k_fn, V, kgrid, temperature=0.0, max_iter=200, tol=1e-6):
        """
        Self-consistent BCS gap equation solver.
        H_k_fn(kx, ky): returns normal-state Hamiltonian H(k) for a single k-point.
        V: pairing interaction strength (negative = attractive).
        kgrid: (kx_array, ky_array) from lattice_momentum_grid.
        Returns {gap, converged, iterations, gap_history}.
        """
        kx_arr, ky_arr = kgrid
        Nk = kx_arr.size
        # Initial gap guess
        gap = 0.1
        gap_history = [gap]
        for iteration in range(max_iter):
            gap_sum = 0.0
            for idx in range(kx_arr.flatten().shape[0]):
                kx = kx_arr.flatten()[idx]
                ky = ky_arr.flatten()[idx]
                Hk = H_k_fn(kx, ky)
                n = Hk.shape[0]
                H_bdg = bdg_hamiltonian(Hk, gap * np.eye(n, dtype=complex))
                eigvals = np.linalg.eigvalsh(H_bdg)
                Ek = eigvals[eigvals > 0]  # Positive eigenvalues only
                for E in Ek:
                    if temperature > 1e-15:
                        fermi = 1.0 / (np.exp(E / temperature) + 1.0)
                        gap_sum += (1.0 - 2.0 * fermi) / (2.0 * E)
                    else:
                        gap_sum += 1.0 / (2.0 * E)
            new_gap = -V * gap_sum / Nk
            gap_history.append(float(abs(new_gap)))
            if abs(new_gap - gap) < tol:
                return {"gap": float(abs(new_gap)), "converged": True,
                        "iterations": iteration + 1, "gap_history": gap_history}
            gap = new_gap
        return {"gap": float(abs(gap)), "converged": False,
                "iterations": max_iter, "gap_history": gap_history}

    def greens_function_retarded(H, omega, eta=0.01):
        """
        Retarded Green's function G^R(omega) = (omega + i*eta - H)^{-1}.
        H: Hamiltonian matrix (N x N).
        omega: frequency (scalar or array).
        eta: broadening parameter.
        Returns G^R as N x N complex matrix (for scalar omega) or list of matrices.
        """
        N = H.shape[0]
        if np.isscalar(omega):
            return np.linalg.inv((omega + 1j * eta) * np.eye(N) - H)
        return [np.linalg.inv((w + 1j * eta) * np.eye(N) - H) for w in omega]

    def spectral_function(H, omega_array, eta=0.01):
        """
        Spectral function A(omega) = -1/pi * Im Tr G^R(omega).
        Returns array of A values for each omega.
        """
        A = np.zeros(len(omega_array))
        N = H.shape[0]
        I = np.eye(N)
        for i, w in enumerate(omega_array):
            G = np.linalg.inv((w + 1j * eta) * I - H)
            A[i] = -np.imag(np.trace(G)) / np.pi
        return A

    def spectral_function_k(H_k_fn, kx, ky, omega_array, eta=0.01):
        """
        Momentum-resolved spectral function A(k, omega).
        H_k_fn(kx, ky): returns Hamiltonian at momentum k.
        Returns 1D array of A values at the given (kx, ky) for each omega.
        """
        H = H_k_fn(kx, ky)
        return spectral_function(H, omega_array, eta)

    def density_of_states(H, omega_array, eta=0.01):
        """DOS = A(omega) / N. Normalized per site."""
        return spectral_function(H, omega_array, eta) / H.shape[0]

    def superfluid_weight(H_k_fn, Delta, kgrid, direction='x', temperature=0.0, dk=0.001):
        """
        Superfluid weight (Drude weight) D_s via current-current correlator.
        Computed from the second derivative of free energy w.r.t. vector potential.
        H_k_fn(kx, ky): normal-state Hamiltonian at k.
        Delta: scalar gap value.
        kgrid: (kx_array, ky_array).
        direction: 'x' or 'y'.
        Returns D_s (float). D_s > 0 indicates superconducting/superfluid response.
        """
        kx_arr, ky_arr = kgrid
        D_s = 0.0
        for idx in range(kx_arr.flatten().shape[0]):
            kx = kx_arr.flatten()[idx]
            ky = ky_arr.flatten()[idx]
            # Numerical second derivative of BdG spectrum w.r.t. A
            if direction == 'x':
                Hp = bdg_hamiltonian(H_k_fn(kx + dk, ky), Delta)
                Hm = bdg_hamiltonian(H_k_fn(kx - dk, ky), Delta)
                H0 = bdg_hamiltonian(H_k_fn(kx, ky), Delta)
            else:
                Hp = bdg_hamiltonian(H_k_fn(kx, ky + dk), Delta)
                Hm = bdg_hamiltonian(H_k_fn(kx, ky - dk), Delta)
                H0 = bdg_hamiltonian(H_k_fn(kx, ky), Delta)
            Ep = np.sort(np.linalg.eigvalsh(Hp))
            Em = np.sort(np.linalg.eigvalsh(Hm))
            E0 = np.sort(np.linalg.eigvalsh(H0))
            # Sum over negative eigenvalues (filled states)
            neg_mask = E0 < 0
            d2E = ((Ep[neg_mask] + Em[neg_mask] - 2 * E0[neg_mask]) / dk**2)
            if temperature > 1e-15:
                fermi = 1.0 / (np.exp(E0[neg_mask] / temperature) + 1.0)
                D_s += np.sum(d2E * fermi)
            else:
                D_s += np.sum(d2E)
        Nk = kx_arr.flatten().shape[0]
        return float(D_s / Nk)

    def self_energy(H_full, H_0, omega, eta=0.01):
        """
        Self-energy Sigma(omega) = G_0^{-1}(omega) - G^{-1}(omega).
        H_full: interacting Hamiltonian. H_0: non-interacting Hamiltonian.
        """
        N = H_full.shape[0]
        I = np.eye(N)
        G0_inv = (omega + 1j * eta) * I - H_0
        G_inv = (omega + 1j * eta) * I - H_full
        return G0_inv - G_inv

    def quasiparticle_weight(H_full, H_0, omega_0=0.0, eta=0.01, domega=0.001):
        """
        Quasiparticle weight Z = (1 - d Re Sigma / d omega)^{-1} at omega_0.
        Z < 1 indicates many-body renormalization; Z → 0 is a non-Fermi-liquid.
        """
        Sp = self_energy(H_full, H_0, omega_0 + domega, eta)
        Sm = self_energy(H_full, H_0, omega_0 - domega, eta)
        dSigma_domega = (Sp - Sm) / (2 * domega)
        # Take diagonal average
        dSigma_avg = np.mean(np.real(np.diag(dSigma_domega)))
        Z = 1.0 / (1.0 - dSigma_avg)
        return float(np.real(Z))

    def matsubara_greens(H, beta, n_matsubara=100):
        """
        Matsubara Green's function G(i*omega_n) for fermions.
        omega_n = (2n+1)*pi/beta.
        Returns (omega_n_array, G_list) where G_list[n] is N x N matrix.
        """
        eigvals, eigvecs = np.linalg.eigh(H)
        omega_n = np.array([(2*n + 1) * np.pi / beta for n in range(n_matsubara)])
        N = H.shape[0]
        G_list = []
        for wn in omega_n:
            G = np.zeros((N, N), dtype=complex)
            for a in range(N):
                G += np.outer(eigvecs[:, a], eigvecs[:, a].conj()) / (1j * wn - eigvals[a])
            G_list.append(G)
        return omega_n, G_list

# ─── Utility: RG flows, multi-subsystem coupling, protocol simulation ────────
if np is not None and scipy is not None:
    def polchinski_rg_flow(beta_fn, g_init, t_span=(0, 5), n_points=500):
        """
        Integrate a truncated Polchinski/Wetterstein RG flow equation.
        beta_fn(t, g): returns dg/dt where g is a vector of coupling constants
        and t is the RG 'time' (log scale).
        g_init: initial coupling constants at UV scale.
        Returns (t_array, g_array) — the flow of couplings from UV to IR.
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = scipy.integrate.solve_ivp(beta_fn, t_span, g_init,
                                         t_eval=t_eval, method='RK45')
        return sol.t, sol.y.T

    def rg_fixed_points(beta_fn, g_init_list, t_end=20):
        """
        Find RG fixed points by flowing multiple initial conditions to large t.
        beta_fn(t, g): RG beta function.
        g_init_list: list of initial coupling vectors.
        Returns list of {initial, final, converged} dicts.
        """
        results_list = []
        for g0 in g_init_list:
            g0 = np.asarray(g0, dtype=float)
            try:
                sol = scipy.integrate.solve_ivp(beta_fn, (0, t_end), g0, method='RK45',
                                                 rtol=1e-10, atol=1e-12)
                g_final = sol.y[:, -1]
                # Check if it converged (beta ~ 0)
                beta_final = np.asarray(beta_fn(t_end, g_final))
                converged = bool(np.linalg.norm(beta_final) < 1e-6)
                results_list.append({
                    "initial": g0.tolist(),
                    "final": g_final.tolist(),
                    "converged": converged,
                    "beta_norm": float(np.linalg.norm(beta_final)),
                })
            except Exception as e:
                results_list.append({"initial": g0.tolist(), "error": str(e)})
        return results_list

    def couple_subsystems(H_A, H_B, V_AB, coupling_strength=1.0):
        """
        Couple two subsystems: H_total = H_A ⊗ I_B + I_A ⊗ H_B + g * V_AB.
        H_A: (nA x nA), H_B: (nB x nB), V_AB: (nA*nB x nA*nB) interaction.
        coupling_strength: scalar multiplier on V_AB.
        """
        nA, nB = H_A.shape[0], H_B.shape[0]
        H = (np.kron(H_A, np.eye(nB, dtype=complex)) +
             np.kron(np.eye(nA, dtype=complex), H_B) +
             coupling_strength * np.asarray(V_AB, dtype=complex))
        return H

    def coupling_sweep(H_A, H_B, V_AB, observable_fn, g_values):
        """
        Sweep coupling strength and measure an observable.
        observable_fn(H_total): returns a scalar measurement from the coupled Hamiltonian.
        Returns list of {g, observable} dicts.
        """
        results_list = []
        for g in g_values:
            H = couple_subsystems(H_A, H_B, V_AB, g)
            try:
                obs = float(observable_fn(H))
                results_list.append({"g": float(g), "observable": obs})
            except Exception as e:
                results_list.append({"g": float(g), "error": str(e)})
        return results_list

    def time_ordered_evolution(H_list, t_list, rho0):
        """
        Time-ordered evolution through a sequence of Hamiltonians.
        H_list: [H1, H2, ...] — Hamiltonians applied in order.
        t_list: [t1, t2, ...] — duration of each stage.
        rho0: initial density matrix.
        Returns final density matrix after all stages.
        """
        rho = rho0.copy()
        for H, t in zip(H_list, t_list):
            U = scipy.linalg.expm(-1j * H * t)
            rho = U @ rho @ U.conj().T
        return rho

    def compare_orderings(H_list, t_list, rho0, observable_fn):
        """
        Test whether the ORDER of Hamiltonian stages matters.
        Runs all permutations of H_list and compares the observable.
        Returns {orderings: [{permutation, result}], order_matters: bool}.
        """
        import itertools as _it
        n = len(H_list)
        results_list = []
        for perm in _it.permutations(range(n)):
            H_perm = [H_list[i] for i in perm]
            t_perm = [t_list[i] for i in perm]
            rho_final = time_ordered_evolution(H_perm, t_perm, rho0)
            obs = float(observable_fn(rho_final))
            results_list.append({"permutation": list(perm), "result": obs})
        vals = [r["result"] for r in results_list]
        spread = max(vals) - min(vals)
        return {
            "orderings": results_list,
            "order_matters": spread > 1e-8,
            "spread": float(spread),
        }

    def chern_number_2d(H_k_fn, Nk=50):
        """
        Compute Chern number for a 2D Bloch Hamiltonian via discretized Berry curvature.
        H_k_fn(kx, ky): returns Hamiltonian matrix at (kx, ky).
        Sums Berry curvature over filled bands (below gap).
        Returns integer Chern number (rounded).
        """
        dk = 2 * np.pi / Nk
        chern = 0.0
        for ix in range(Nk):
            for iy in range(Nk):
                kx = -np.pi + ix * dk
                ky = -np.pi + iy * dk
                # Four corners of plaquette
                H00 = H_k_fn(kx, ky)
                H10 = H_k_fn(kx + dk, ky)
                H01 = H_k_fn(kx, ky + dk)
                H11 = H_k_fn(kx + dk, ky + dk)
                # Ground state at each corner
                n_bands = H00.shape[0]
                n_filled = n_bands // 2  # assume half-filling
                def ground_states(H):
                    _, vecs = np.linalg.eigh(H)
                    return vecs[:, :n_filled]
                u00 = ground_states(H00)
                u10 = ground_states(H10)
                u01 = ground_states(H01)
                u11 = ground_states(H11)
                # Berry phase around plaquette via link variables
                U1 = np.linalg.det(u00.conj().T @ u10)
                U2 = np.linalg.det(u10.conj().T @ u11)
                U3 = np.linalg.det(u11.conj().T @ u01)
                U4 = np.linalg.det(u01.conj().T @ u00)
                F = np.log(U1 * U2 * U3 * U4)
                chern += np.imag(F)
        return round(chern / (2 * np.pi))

    def compare_topological_invariants(H_k_fn_A, H_k_fn_B, Nk=50):
        """
        Compare Chern numbers of two 2D systems to test topological equivalence.
        Returns {chern_A, chern_B, same_class}.
        """
        cA = chern_number_2d(H_k_fn_A, Nk)
        cB = chern_number_2d(H_k_fn_B, Nk)
        return {"chern_A": cA, "chern_B": cB, "same_class": cA == cB}

    def randomized_benchmarking(n_qubits, gate_fn, noise_fn, sequence_lengths,
                                 n_sequences=50, n_shots=100):
        """
        Simulate randomized benchmarking.
        gate_fn(gate_id): returns unitary matrix for gate_id.
        noise_fn(U): returns noisy version of unitary U (e.g., with depolarization).
        sequence_lengths: list of integers (number of random gates per sequence).
        Returns {lengths, survival_probabilities, estimated_fidelity}.
        """
        dim = 2**n_qubits
        # Generate random Clifford-like gates (simplified: random unitaries)
        survivals = []
        for L in sequence_lengths:
            survival_sum = 0.0
            for _ in range(n_sequences):
                # Build random sequence
                U_ideal = np.eye(dim, dtype=complex)
                U_noisy = np.eye(dim, dtype=complex)
                for _ in range(L):
                    gate_id = np.random.randint(0, 24)  # simplified Clifford index
                    U_g = gate_fn(gate_id)
                    U_ideal = U_g @ U_ideal
                    U_noisy = noise_fn(U_g) @ U_noisy
                # Inversion gate
                U_inv = U_ideal.conj().T
                U_noisy_total = noise_fn(U_inv) @ U_noisy
                # Initial state |0...0>
                psi0 = np.zeros(dim, dtype=complex)
                psi0[0] = 1.0
                psi_final = U_noisy_total @ psi0
                survival_sum += abs(psi_final[0])**2
            survivals.append(float(survival_sum / n_sequences))
        # Fit exponential decay: p = A * r^L + B
        from scipy.optimize import curve_fit
        def rb_model(L, A, r, B):
            return A * r**L + B
        try:
            popt, _ = curve_fit(rb_model, sequence_lengths, survivals,
                                p0=[0.5, 0.99, 0.5], bounds=([0, 0, 0], [1, 1, 1]))
            avg_fidelity = (popt[1] * (dim - 1) + 1) / dim
        except:
            avg_fidelity = None
        return {
            "lengths": [int(L) for L in sequence_lengths],
            "survival_probabilities": survivals,
            "estimated_fidelity": float(avg_fidelity) if avg_fidelity else None,
        }

    def test_causal_chain(step_fns, initial_state, observable_fn):
        """
        Test a multi-step causal chain by running each link independently.
        step_fns: list of functions [f1, f2, ...] where f_i(state) → new_state.
        Tests: (1) does the full chain produce the claimed effect?
               (2) does each individual link's output feed the next link?
               (3) what happens if we skip each link?
        Returns {full_chain, skip_tests, link_outputs, chain_intact}.
        """
        # Full chain
        state = initial_state
        link_outputs = []
        for fn in step_fns:
            state = fn(state)
            link_outputs.append(float(observable_fn(state)))
        full_result = float(observable_fn(state))

        # Skip each link
        skip_tests = []
        for skip_idx in range(len(step_fns)):
            state = initial_state
            for i, fn in enumerate(step_fns):
                if i == skip_idx:
                    continue
                state = fn(state)
            skip_result = float(observable_fn(state))
            skip_tests.append({
                "skipped_step": skip_idx,
                "result": skip_result,
                "link_needed": abs(skip_result - full_result) > 1e-8,
            })

        all_needed = all(s["link_needed"] for s in skip_tests)
        return {
            "full_chain_result": full_result,
            "link_outputs": link_outputs,
            "skip_tests": skip_tests,
            "chain_intact": all_needed,
        }

    def with_and_without(H_base, H_extra, observable_fn):
        """
        Test whether adding H_extra to H_base changes an observable.
        Computes observable for H_base alone and H_base + H_extra.
        Returns {without, with_extra, difference, effect_present}.
        """
        obs_without = float(observable_fn(H_base))
        obs_with = float(observable_fn(H_base + H_extra))
        diff = abs(obs_with - obs_without)
        return {
            "without": obs_without,
            "with_extra": obs_with,
            "difference": float(diff),
            "effect_present": diff > 1e-8,
        }

    def verify_symplectic(omega_matrix):
        """
        Verify that a 2-form matrix is symplectic (closed, non-degenerate).
        omega_matrix: 2n x 2n antisymmetric matrix.
        Returns {antisymmetric, non_degenerate, det, valid_symplectic}.
        """
        antisymm = bool(np.allclose(omega_matrix, -omega_matrix.T, atol=1e-10))
        det_val = float(np.real(np.linalg.det(omega_matrix)))
        non_degen = abs(det_val) > 1e-10
        return {
            "antisymmetric": antisymm,
            "non_degenerate": non_degen,
            "determinant": det_val,
            "valid_symplectic": antisymm and non_degen,
        }

    def critical_exponents(observable_fn, control_param_values, critical_point, fit_range=0.1):
        """
        Extract critical exponent from power-law scaling near a phase transition.
        observable_fn(p): returns observable at control parameter p.
        critical_point: estimated location of transition.
        Fits observable ~ |p - p_c|^alpha near p_c.
        Returns {alpha, fit_quality, data}.
        """
        data = []
        for p in control_param_values:
            data.append({"p": float(p), "obs": float(observable_fn(p))})
        # Select points near critical point
        near = [(d["p"], d["obs"]) for d in data
                if abs(d["p"] - critical_point) < fit_range and abs(d["p"] - critical_point) > 1e-10]
        if len(near) < 3:
            return {"alpha": None, "fit_quality": 0, "data": data, "error": "too few points near critical point"}
        x = np.array([abs(p - critical_point) for p, _ in near])
        y = np.array([abs(o) for _, o in near])
        y = y[y > 1e-15]; x = x[:len(y)]
        if len(x) < 3:
            return {"alpha": None, "fit_quality": 0, "data": data, "error": "observable too small"}
        log_x, log_y = np.log(x), np.log(y)
        coeffs = np.polyfit(log_x, log_y, 1)
        alpha = coeffs[0]
        residuals = log_y - np.polyval(coeffs, log_x)
        r_sq = 1 - np.sum(residuals**2) / np.sum((log_y - np.mean(log_y))**2)
        return {"alpha": float(alpha), "fit_quality": float(r_sq), "data": data}

# ─── Utility: wave systems (band structure, transfer matrices) ───────────────
if np is not None:
    def tight_binding_1d(N, on_site, hopping, periodic=True):
        """
        Build 1D tight-binding Hamiltonian.
        on_site: scalar or array of length N (diagonal energies).
        hopping: scalar or array of length N-1 (nearest-neighbor hopping).
        Returns N x N Hamiltonian matrix.
        """
        H = np.zeros((N, N), dtype=complex)
        eps = np.full(N, on_site) if np.isscalar(on_site) else np.asarray(on_site, dtype=complex)
        t = np.full(N-1, hopping) if np.isscalar(hopping) else np.asarray(hopping, dtype=complex)
        np.fill_diagonal(H, eps)
        for i in range(N-1):
            H[i, i+1] = t[i]
            H[i+1, i] = np.conj(t[i])
        if periodic and N > 2:
            H[0, N-1] = np.conj(t[0]) if np.isscalar(hopping) else np.conj(hopping)
            H[N-1, 0] = t[0] if np.isscalar(hopping) else hopping
        return H

    def band_structure_1d(H_cell, H_hop, k_points=200):
        """
        Compute 1D band structure from unit cell Hamiltonian and hopping matrix.
        H_cell: intra-cell Hamiltonian (n x n).
        H_hop: inter-cell hopping matrix (n x n), couples cell i to cell i+1.
        k_points: number of k values in [-pi, pi].
        Returns (k_values, bands) where bands[i] are eigenvalues at k_values[i].
        """
        ks = np.linspace(-np.pi, np.pi, k_points)
        n = H_cell.shape[0]
        bands = np.zeros((k_points, n))
        for i, k in enumerate(ks):
            Hk = H_cell + H_hop * np.exp(1j * k) + H_hop.conj().T * np.exp(-1j * k)
            bands[i] = np.sort(np.real(np.linalg.eigvalsh(Hk)))
        return ks, bands

    def transfer_matrix(omega, layers):
        """
        Compute total transfer matrix for 1D wave propagation.
        layers: list of (n_i, d_i) tuples — refractive index and thickness.
        omega: angular frequency.
        Returns 2x2 transfer matrix.
        """
        M = np.eye(2, dtype=complex)
        for n_i, d_i in layers:
            k = omega * n_i
            phase = k * d_i
            layer_M = np.array([
                [np.cos(phase), 1j * np.sin(phase) / n_i],
                [1j * n_i * np.sin(phase), np.cos(phase)]
            ], dtype=complex)
            M = layer_M @ M
        return M

    def transmission_coefficient(M):
        """Transmission coefficient from a 2x2 transfer matrix."""
        return float(1.0 / abs(M[0, 0]) ** 2)

    def disorder_sweep(build_H_fn, N_sites, disorder_strengths, n_realizations=50):
        """
        Test robustness to disorder. For each disorder strength W, builds n_realizations
        of a disordered Hamiltonian and collects eigenvalue statistics.
        build_H_fn(N, disorder_array): returns Hamiltonian with on-site disorder.
        Returns list of {W, mean_gap, std_gap, mean_ipr} dicts.
        """
        results_list = []
        for W in disorder_strengths:
            gaps = []
            iprs = []
            for _ in range(n_realizations):
                disorder = np.random.uniform(-W/2, W/2, N_sites)
                H = np.asarray(build_H_fn(N_sites, disorder), dtype=complex)
                # Force Hermitian to avoid complex warnings
                H = (H + H.conj().T) / 2
                eigvals, eigvecs = np.linalg.eigh(H)
                sorted_e = np.sort(eigvals)
                mid = len(sorted_e) // 2
                gap = sorted_e[mid] - sorted_e[mid - 1] if mid > 0 else 0
                gaps.append(gap)
                # Inverse participation ratio of mid-gap state
                psi = eigvecs[:, mid]
                iprs.append(float(np.sum(np.abs(psi)**4)))
            results_list.append({
                "W": float(W),
                "mean_gap": float(np.mean(gaps)),
                "std_gap": float(np.std(gaps)),
                "mean_ipr": float(np.mean(iprs)),
            })
        return results_list

# ─── Utility: coupled dynamics (ODE integration, stability) ──────────────────
if np is not None and scipy is not None:
    def solve_ode(rhs, y0, t_span, n_points=1000, method='RK45', **kwargs):
        """
        Integrate dy/dt = rhs(t, y).
        y0: initial condition (array).
        t_span: (t_start, t_end).
        Returns (t_array, y_array) where y_array shape is (n_points, len(y0)).
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = scipy.integrate.solve_ivp(rhs, t_span, y0, method=method, t_eval=t_eval, **kwargs)
        return sol.t, sol.y.T

    def stability_eigenvalues(jacobian_fn, fixed_point):
        """
        Compute eigenvalues of the Jacobian at a fixed point.
        jacobian_fn(y): returns Jacobian matrix at state y.
        Returns sorted eigenvalues (complex).
        """
        J = np.asarray(jacobian_fn(fixed_point), dtype=complex)
        eigvals = np.linalg.eigvals(J)
        return sorted(eigvals.tolist(), key=lambda z: z.real)

    def find_threshold(param_fn, param_range, criterion_fn, n_points=200):
        """
        Binary-search style sweep to find critical parameter value.
        param_fn(p): set up system at parameter value p, return state/result.
        criterion_fn(state): return True if criterion met.
        param_range: (p_min, p_max).
        Returns estimated critical parameter value.
        """
        ps = np.linspace(param_range[0], param_range[1], n_points)
        below = param_range[0]
        above = param_range[1]
        for p in ps:
            state = param_fn(p)
            if criterion_fn(state):
                above = p
                break
            below = p
        # Refine with bisection
        for _ in range(30):
            mid = (below + above) / 2
            state = param_fn(mid)
            if criterion_fn(state):
                above = mid
            else:
                below = mid
        return (below + above) / 2

    def solve_master_equation(H, rho0, t_span, L_ops=None, gamma=None, n_points=500):
        """
        Integrate Lindblad master equation for density matrix evolution.
        H: Hamiltonian (n x n). rho0: initial density matrix (n x n).
        L_ops: list of Lindblad operators. gamma: list of decay rates.
        Returns (t_array, rho_list) where rho_list[i] is density matrix at t[i].
        """
        if L_ops is None:
            L_ops = []
        if gamma is None:
            gamma = [1.0] * len(L_ops)
        n = H.shape[0]
        # Vectorize: rho (n x n) -> vec (n^2,)
        def rhs(t, vec):
            rho = vec.reshape(n, n)
            drho = lindblad_rhs(rho, H, L_ops, gamma)
            return drho.flatten()
        y0 = rho0.flatten().astype(complex)
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = scipy.integrate.solve_ivp(rhs, t_span, y0, t_eval=t_eval, method='RK45',
                                         max_step=(t_span[1]-t_span[0])/100)
        rho_list = [sol.y[:, i].reshape(n, n) for i in range(sol.y.shape[1])]
        return sol.t, rho_list

    def wigner_function(rho, xvec, pvec):
        """
        Compute Wigner function W(x, p) for a single-mode state.
        rho: density matrix in Fock basis (truncated at dim n).
        xvec, pvec: 1D arrays of x and p values.
        Returns 2D array W[ix, ip].
        """
        n = rho.shape[0]
        W = np.zeros((len(xvec), len(pvec)))
        # Create annihilation operator
        a = np.zeros((n, n), dtype=complex)
        for i in range(n - 1):
            a[i, i+1] = np.sqrt(i + 1)
        x_op = (a + a.conj().T) / np.sqrt(2)
        p_op = 1j * (a.conj().T - a) / np.sqrt(2)
        for ix, x in enumerate(xvec):
            for ip, p in enumerate(pvec):
                # Displacement operator approach (truncated)
                alpha = (x + 1j * p) / np.sqrt(2)
                D = scipy.linalg.expm(alpha * a.conj().T - np.conj(alpha) * a)
                parity = np.diag([(-1)**k for k in range(n)]).astype(complex)
                W[ix, ip] = np.real(np.trace(rho @ D @ parity @ D.conj().T)) / np.pi
        return W

# === COMPUTATION (LLM-generated) ===
{{computation}}
# === END COMPUTATION ===
