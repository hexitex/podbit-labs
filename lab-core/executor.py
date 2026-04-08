#!/usr/bin/env python3
"""
Lab Sandbox Executor — used by every lab in podbit-labs.

Executes lab-generated Python code in a separate process. The code may be a
math-lab template with LLM-generated computation inserted, an nn-lab training
script, or any other Python the lab pipeline produces.

Security model:
  - Optional network kill switch (patches socket at the C level) — controlled by
    the second CLI arg, configured per-lab via sandbox.networkKillSwitch.
  - Process timeout enforced by the parent (sandbox.executionTimeoutMs).
  - Output cap enforced by the parent (sandbox.maxOutputBytes).

Code must set a variable called `result` (or define a `main()` that returns one).
Output is a single JSON object on stdout:
  {"result": ..., "error": null, "execution_time_ms": ...}
"""

import sys
import json
import time
import io

# =============================================================================
# CONFIGURATION
# =============================================================================

if len(sys.argv) < 2:
    print(json.dumps({"result": None, "error": "Usage: executor.py <code_path> [network_kill]", "execution_time_ms": 0}))
    sys.exit(1)

code_path = sys.argv[1]
network_kill_switch = sys.argv[2] == '1' if len(sys.argv) > 2 else True


# =============================================================================
# OUTPUT HELPER
# =============================================================================

_real_print = print

def output(result=None, error=None, execution_time_ms=0):
    """Write structured JSON to stdout and exit. On error, also mirror to stderr."""
    _real_print(json.dumps({
        "result": result,
        "error": error,
        "execution_time_ms": execution_time_ms,
    }))
    if error is not None:
        try:
            sys.stderr.write(f"EXECUTOR ERROR: {error}\n")
            sys.stderr.flush()
        except Exception:
            pass
    sys.exit(0 if error is None else 1)


# =============================================================================
# NETWORK KILL SWITCH — patches socket at C-module level
# =============================================================================

def _blocked(*a, **kw):
    raise PermissionError("Network access is blocked in the lab sandbox")

if network_kill_switch:
    try:
        import socket as _socket_mod
        _original_socket_init = _socket_mod.socket.__init__

        _socket_mod.socket.__init__ = lambda self, *a, **kw: (_ for _ in ()).throw(
            PermissionError("Network access is blocked in the lab sandbox"))
        _socket_mod.socket.connect = _blocked
        _socket_mod.socket.bind = _blocked
        _socket_mod.socket.listen = _blocked
        _socket_mod.socket.send = _blocked
        _socket_mod.socket.sendto = _blocked
        _socket_mod.socket.sendall = _blocked
        _socket_mod.create_connection = _blocked
        _socket_mod.create_server = _blocked

        try:
            import ssl as _ssl_mod
            _ssl_mod.wrap_socket = _blocked
            _ssl_mod.SSLContext.wrap_socket = _blocked
            _ssl_mod.create_default_context = _blocked
        except ImportError:
            pass
    except Exception:
        pass


# =============================================================================
# LOAD CODE
# =============================================================================

try:
    with open(code_path, 'r', encoding='utf-8') as f:
        code_str = f.read()
except Exception as e:
    output(error=f"Cannot read code file: {e}")

try:
    compiled = compile(code_str, '<lab-sandbox>', 'exec')
except SyntaxError as e:
    output(error=f"Syntax error: {e}")


# =============================================================================
# EXECUTE
# =============================================================================

# Use the real __main__ dict so pickle/numba can find functions defined in exec'd code
exec_globals = sys.modules['__main__'].__dict__

# Suppress print in user code (output via result variable only)
import builtins as _builtins
def _noop_print(*a, **kw): pass
_builtins.print = _noop_print

start = time.monotonic()
try:
    exec(compiled, exec_globals)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    if 'result' in exec_globals:
        output(result=exec_globals['result'], execution_time_ms=elapsed_ms)
    elif 'main' in exec_globals and callable(exec_globals['main']):
        ret = exec_globals['main']()
        if ret is not None:
            output(result=ret, execution_time_ms=elapsed_ms)
        elif 'result' in exec_globals:
            output(result=exec_globals['result'], execution_time_ms=elapsed_ms)
        else:
            output(error="Code executed main() but no 'result' variable was set and main() returned None", execution_time_ms=elapsed_ms)
    else:
        output(error="Code executed but no 'result' variable was set", execution_time_ms=elapsed_ms)
except Exception as e:
    import traceback as _tb
    elapsed_ms = int((time.monotonic() - start) * 1000)
    tb_str = _tb.format_exc()
    output(error=f"{type(e).__name__}: {e}\n{tb_str}", execution_time_ms=elapsed_ms)
