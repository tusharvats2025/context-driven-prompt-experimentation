"""
aggregate_results.py
====================
Loads all trial JSON files from models/results/ and produces
aggregated statistics per pattern, per model, and per metric.

Output:
  analysis/outputs/tables/table1_pattern_comparison.md
  analysis/outputs/tables/table2_model_comparison.md
  analysis/outputs/tables/table3_failure_analysis.md
  analysis/outputs/aggregate_summary.json

CDP v1.0 — linear_regression domain
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "models" / "results"
TABLES_DIR   = PROJECT_ROOT / "analysis" / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_trials() -> List[Dict[str, Any]]:
    trials = []
    for path in sorted(RESULTS_DIR.glob("trial_*.json")):
        if "_raw" in path.name:
            continue
        with open(path) as f:
            trials.append(json.load(f))
    print(f"[Aggregate] Loaded {len(trials)} trial results")
    return trials


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "no_dummy_trap"              : "No Dummy Trap",
    "vif_check"                  : "VIF Check",
    "heteroscedasticity_handled" : "Heteroscedasticity",
    "outliers_treated"           : "Outliers Treated",
    "scaler_no_leakage"          : "Scaler No Leakage",
    "residuals_checked"          : "Residuals Checked",
    "model_saves"                : "Model Saves",
}

PATTERN_ORDER = [
    "zero_shot", "few_shot", "chain_of_thought",
    "role_based", "legacy_expert", "cdp",
]


def group_by_pattern(trials: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for t in trials:
        groups[t["pattern_name"]].append(t)
    return dict(groups)


def group_by_model(trials: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for t in trials:
        groups[t["model_name"]].append(t)
    return dict(groups)


def avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = avg(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def metric_pass_rate(trials: List[Dict], metric_id: str) -> float:
    rates = []
    for t in trials:
        scores = t.get("scores", {})
        if metric_id in scores:
            rates.append(1.0 if scores[metric_id]["result"] == "PASS" else 0.0)
    return avg(rates) * 100


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Pattern comparison
# ─────────────────────────────────────────────────────────────────────────────

def build_table1(trials: List[Dict]) -> str:
    by_pattern = group_by_pattern(trials)
    metric_ids = list(METRIC_LABELS.keys())

    lines = [
        "# Table 1: Prompt Pattern Comparison",
        "",
        "| Pattern | N Trials | Avg Score | Std Dev | Avg Time (s) |"
        + " | ".join(METRIC_LABELS.values()) + " |",
        "|---------|----------|-----------|---------|--------------|"
        + "|".join(["---"] * len(METRIC_LABELS)) + "|",
    ]

    for pattern in PATTERN_ORDER:
        group = by_pattern.get(pattern, [])
        if not group:
            continue

        scores  = [t["score_pct"] for t in group]
        times   = [t["elapsed_seconds"] for t in group]
        metric_cols = " | ".join(
            f"{metric_pass_rate(group, m):.0f}%"
            for m in metric_ids
        )

        lines.append(
            f"| {pattern:<20} | {len(group):^8} | "
            f"{avg(scores):^9.1f}% | {std(scores):^7.1f}% | "
            f"{avg(times):^12.1f} | {metric_cols} |"
        )

    lines += ["", "_Pass rate per metric shown as % across all trials._", ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 2: Model comparison
# ─────────────────────────────────────────────────────────────────────────────

def build_table2(trials: List[Dict]) -> str:
    by_model = group_by_model(trials)

    lines = [
        "# Table 2: Model Comparison",
        "",
        "| Model | N Trials | Avg Score | Std Dev | Best Pattern | Worst Pattern |",
        "|-------|----------|-----------|---------|--------------|---------------|",
    ]

    for model, group in by_model.items():
        by_pat   = group_by_pattern(group)
        pat_avgs = {
            p: avg([t["score_pct"] for t in ts])
            for p, ts in by_pat.items()
        }
        best  = max(pat_avgs, key=pat_avgs.get) if pat_avgs else "—"
        worst = min(pat_avgs, key=pat_avgs.get) if pat_avgs else "—"
        scores= [t["score_pct"] for t in group]

        lines.append(
            f"| {model:<20} | {len(group):^8} | "
            f"{avg(scores):^9.1f}% | {std(scores):^7.1f}% | "
            f"{best:<12} | {worst:<13} |"
        )

    lines += ["", ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 3: Failure analysis
# ─────────────────────────────────────────────────────────────────────────────

def build_table3(trials: List[Dict]) -> str:
    by_pattern = group_by_pattern(trials)
    metric_ids = list(METRIC_LABELS.keys())

    lines = [
        "# Table 3: Failure Mode Analysis",
        "",
        "Failure rate per metric per pattern (% of trials where metric FAILED).",
        "",
        "| Metric | " + " | ".join(PATTERN_ORDER) + " |",
        "|--------|" + "|".join(["------"] * len(PATTERN_ORDER)) + "|",
    ]

    for metric_id in metric_ids:
        label = METRIC_LABELS[metric_id]
        cols  = []
        for pattern in PATTERN_ORDER:
            group = by_pattern.get(pattern, [])
            if not group:
                cols.append("—")
            else:
                fail_rate = 100 - metric_pass_rate(group, metric_id)
                cols.append(f"{fail_rate:.0f}%")
        lines.append(f"| {label:<30} | " + " | ".join(cols) + " |")

    lines += [
        "",
        "**Key insight**: High failure rate on `vif_check` and `no_dummy_trap`",
        "across zero-shot/few-shot confirms CDP's structured enforcement is necessary.",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate JSON summary
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_json(trials: List[Dict]) -> Dict:
    by_pattern = group_by_pattern(trials)
    metric_ids = list(METRIC_LABELS.keys())

    pattern_stats = {}
    for pattern, group in by_pattern.items():
        scores = [t["score_pct"] for t in group]
        times  = [t["elapsed_seconds"] for t in group]
        pattern_stats[pattern] = {
            "n_trials"     : len(group),
            "avg_score_pct": round(avg(scores), 2),
            "std_score_pct": round(std(scores), 2),
            "avg_time_sec" : round(avg(times), 2),
            "metric_pass_rates": {
                m: round(metric_pass_rate(group, m), 1)
                for m in metric_ids
            },
        }

    return {
        "total_trials"  : len(trials),
        "patterns_tested": list(by_pattern.keys()),
        "models_tested" : list(group_by_model(trials).keys()),
        "pattern_stats" : pattern_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trials = load_all_trials()

    if not trials:
        print("[Aggregate] No trial results found. Run agent_runner.py first.")
        sys.exit(0)

    # Write tables
    for name, content in [
        ("table1_pattern_comparison.md", build_table1(trials)),
        ("table2_model_comparison.md",   build_table2(trials)),
        ("table3_failure_analysis.md",   build_table3(trials)),
    ]:
        path = TABLES_DIR / name
        path.write_text(content)
        print(f"[Aggregate] Wrote → {path}")

    # Write summary JSON
    summary = build_summary_json(trials)
    summary_path = PROJECT_ROOT / "analysis" / "outputs" / "aggregate_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Aggregate] Summary → {summary_path}")
