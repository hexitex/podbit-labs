"""
Cross-condition comparison measurements.

Computed after all conditions have been trained.
"""

import numpy as np


def compute_comparisons(conditions: list[dict], requested: list[str]) -> dict:
    """
    Compute comparison measurements across conditions.

    Args:
        conditions: List of condition results, each with "runs" list
        requested: List of comparison measurement names

    Returns dict of comparison results, each with mean, std, per_run.
    """
    if len(conditions) < 2:
        return {}

    result = {}

    for name in requested:
        m_name = name if isinstance(name, str) else name.get("name", "")
        if m_name == "convergence_speed_ratio":
            result[m_name] = _convergence_speed_ratio(conditions)
        elif m_name == "final_accuracy_difference":
            result[m_name] = _final_accuracy_difference(conditions)
        elif m_name == "sparsity_distribution_divergence":
            result[m_name] = _sparsity_distribution_divergence(conditions)
        elif m_name == "gradient_stat_divergence":
            result[m_name] = _gradient_stat_divergence(conditions)

    return result


def _convergence_speed_ratio(conditions: list[dict]) -> dict:
    """
    Ratio of convergence epochs between condition 0 and condition 1.
    > 1 means condition 0 converges slower.
    """
    c0_runs = conditions[0]["runs"]
    c1_runs = conditions[1]["runs"]

    ratios = []
    for r0, r1 in zip(c0_runs, c1_runs):
        e0 = r0.get("convergence_epoch")
        e1 = r1.get("convergence_epoch")
        if e0 is not None and e1 is not None and e1 > 0:
            ratios.append(e0 / e1)

    if not ratios:
        # Fall back to comparing final val losses for convergence speed
        for r0, r1 in zip(c0_runs, c1_runs):
            lc0 = r0.get("loss_curve", {}).get("val", [])
            lc1 = r1.get("loss_curve", {}).get("val", [])
            if lc0 and lc1:
                # Find epoch where loss first drops below 50% of initial
                def _first_half(lc):
                    if not lc or lc[0] == 0:
                        return len(lc)
                    target = lc[0] * 0.5
                    for i, v in enumerate(lc):
                        if v < target:
                            return i + 1
                    return len(lc)
                e0 = _first_half(lc0)
                e1 = _first_half(lc1)
                if e1 > 0:
                    ratios.append(e0 / e1)

    if not ratios:
        return {"mean": None, "std": None, "per_run": []}

    return {
        "mean": float(np.mean(ratios)),
        "std": float(np.std(ratios)),
        "per_run": [float(r) for r in ratios],
    }


def _final_accuracy_difference(conditions: list[dict]) -> dict:
    """Difference in final accuracy: condition[0] - condition[1]."""
    c0_runs = conditions[0]["runs"]
    c1_runs = conditions[1]["runs"]

    diffs = []
    for r0, r1 in zip(c0_runs, c1_runs):
        a0 = r0.get("final_accuracy")
        a1 = r1.get("final_accuracy")
        if a0 is not None and a1 is not None:
            diffs.append(a0 - a1)

    if not diffs:
        return {"mean": None, "std": None, "per_run": []}

    return {
        "mean": float(np.mean(diffs)),
        "std": float(np.std(diffs)),
        "per_run": [float(d) for d in diffs],
    }


def _sparsity_distribution_divergence(conditions: list[dict]) -> dict:
    """
    Jensen-Shannon divergence of sparsity distributions between conditions.
    Treats per-layer sparsity as a discrete distribution.
    """
    c0_runs = conditions[0]["runs"]
    c1_runs = conditions[1]["runs"]

    divs = []
    for r0, r1 in zip(c0_runs, c1_runs):
        s0 = r0.get("sparsity_by_layer", [])
        s1 = r1.get("sparsity_by_layer", [])
        if s0 and s1:
            v0 = np.array([e["sparsity"] for e in s0])
            v1 = np.array([e["sparsity"] for e in s1])
            # Align lengths
            min_len = min(len(v0), len(v1))
            v0, v1 = v0[:min_len], v1[:min_len]
            # JS divergence
            v0_n = v0 / (v0.sum() + 1e-12)
            v1_n = v1 / (v1.sum() + 1e-12)
            m = (v0_n + v1_n) / 2
            kl0 = np.sum(v0_n * np.log((v0_n + 1e-12) / (m + 1e-12)))
            kl1 = np.sum(v1_n * np.log((v1_n + 1e-12) / (m + 1e-12)))
            divs.append((kl0 + kl1) / 2)

    if not divs:
        return {"mean": None, "std": None, "per_run": []}

    return {
        "mean": float(np.mean(divs)),
        "std": float(np.std(divs)),
        "per_run": [float(d) for d in divs],
    }


def _gradient_stat_divergence(conditions: list[dict]) -> dict:
    """
    Cosine distance between gradient stat trajectories.
    Compares the flattened gradient-mean vectors over epochs.
    """
    c0_runs = conditions[0]["runs"]
    c1_runs = conditions[1]["runs"]

    divs = []
    for r0, r1 in zip(c0_runs, c1_runs):
        g0 = r0.get("gradient_stats_by_layer", [])
        g1 = r1.get("gradient_stats_by_layer", [])
        if g0 and g1:
            # Flatten: [epoch][layer] mean values
            def _flatten(snapshots):
                vals = []
                for snap in snapshots:
                    for stat in snap.get("stats", []):
                        vals.append(stat.get("mean", 0))
                return np.array(vals)

            v0 = _flatten(g0)
            v1 = _flatten(g1)
            min_len = min(len(v0), len(v1))
            v0, v1 = v0[:min_len], v1[:min_len]

            norm0 = np.linalg.norm(v0)
            norm1 = np.linalg.norm(v1)
            if norm0 > 1e-12 and norm1 > 1e-12:
                cosine_sim = np.dot(v0, v1) / (norm0 * norm1)
                divs.append(1.0 - cosine_sim)  # cosine distance

    if not divs:
        return {"mean": None, "std": None, "per_run": []}

    return {
        "mean": float(np.mean(divs)),
        "std": float(np.std(divs)),
        "per_run": [float(d) for d in divs],
    }
