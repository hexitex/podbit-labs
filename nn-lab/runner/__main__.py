"""
NN-Lab Runner — direct invocation entry point.

Usage: python -m runner <spec.json> [--artifact-dir <dir>]
"""

import sys
import json
import os

from runner.experiment import run_experiment


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python -m runner <spec.json> [--artifact-dir <dir>]", "result": None}))
        sys.exit(1)

    spec_path = sys.argv[1]
    artifact_dir = "."

    for i, arg in enumerate(sys.argv):
        if arg == "--artifact-dir" and i + 1 < len(sys.argv):
            artifact_dir = sys.argv[i + 1]

    try:
        with open(spec_path, 'r') as f:
            spec = json.load(f)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load spec: {e}", "result": None}))
        sys.exit(1)

    result = run_experiment(spec, artifact_dir)
    print(json.dumps(result, default=_json_default))


def _json_default(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or (isinstance(obj, float) and (obj != obj)):
        return None
    return str(obj)


if __name__ == "__main__":
    main()
