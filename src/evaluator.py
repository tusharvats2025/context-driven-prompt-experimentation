"""
evaluator.py
============
Scores LLM-generated code/explanations against CDP checkpoints.

For each trial (one LLM output for one prompt pattern), the evaluator:
  1. Loads the ScoringCheckpoints extracted from linear_regression.cdp
  2. Scans the LLM output for pass/fail signals per metric
  3. Returns a TrialResult with binary scores for all 7 metrics

This is the scoring engine for the experiment.

CDP v1.0 — linear_regression domain
"""

from __future__ import annotations

import re
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

from cdp_checkpoint_extractor import ScoringCheckpoint
from cdp_types import EXPERIMENT_METRICS


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

class MetricResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"   # checkpoint not applicable to this pattern


@dataclass
class MetricScore:
    metric_id   : str
    result      : MetricResult
    evidence    : List[str]         # lines in LLM output that triggered PASS
    violations  : List[str]         # lines that triggered FAIL
    confidence  : float             # 0.0 – 1.0

    @property
    def passed(self) -> bool:
        return self.result == MetricResult.PASS


@dataclass
class TrialResult:
    """Complete result for one LLM trial."""
    trial_id        : str           # e.g. "trial_001"
    pattern_name    : str           # e.g. "zero_shot"
    model_name      : str           # e.g. "llama3.2"
    elapsed_seconds : float
    scores          : Dict[str, MetricScore]
    raw_output_path : str = ""

    @property
    def score_pct(self) -> float:
        passed = sum(1 for s in self.scores.values() if s.passed)
        total  = len(self.scores)
        return (passed / total * 100) if total > 0 else 0.0

    @property
    def passed_metrics(self) -> List[str]:
        return [k for k, v in self.scores.items() if v.passed]

    @property
    def failed_metrics(self) -> List[str]:
        return [k for k, v in self.scores.items() if not v.passed]

    def to_dict(self) -> dict:
        return {
            "trial_id"        : self.trial_id,
            "pattern_name"    : self.pattern_name,
            "model_name"      : self.model_name,
            "elapsed_seconds" : self.elapsed_seconds,
            "score_pct"       : round(self.score_pct, 2),
            "passed_count"    : len(self.passed_metrics),
            "failed_count"    : len(self.failed_metrics),
            "passed_metrics"  : self.passed_metrics,
            "failed_metrics"  : self.failed_metrics,
            "scores"          : {
                k: {
                    "result"    : v.result.value,
                    "confidence": round(v.confidence, 2),
                    "evidence"  : v.evidence[:3],       # top 3 only
                    "violations": v.violations[:3],
                }
                for k, v in self.scores.items()
            },
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class CDPEvaluator:
    """
    Scores LLM output against a list of ScoringCheckpoints.

    Usage:
        evaluator   = CDPEvaluator(checkpoints)
        trial_result = evaluator.score(
            llm_output   = "...generated code...",
            trial_id     = "trial_001",
            pattern_name = "zero_shot",
            model_name   = "llama3.2",
            elapsed      = 30.4,
        )
    """

    def __init__(self, checkpoints: List[ScoringCheckpoint]):
        self.checkpoints = {cp.metric_id: cp for cp in checkpoints}

    def score(
        self,
        llm_output   : str,
        trial_id     : str,
        pattern_name : str,
        model_name   : str,
        elapsed      : float,
    ) -> TrialResult:

        scores: Dict[str, MetricScore] = {}

        for metric_id, checkpoint in self.checkpoints.items():
            metric_score = self._score_metric(llm_output, checkpoint)
            scores[metric_id] = metric_score

        return TrialResult(
            trial_id        = trial_id,
            pattern_name    = pattern_name,
            model_name      = model_name,
            elapsed_seconds = elapsed,
            scores          = scores,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Per-metric scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _score_metric(
        self,
        llm_output : str,
        checkpoint : ScoringCheckpoint,
    ) -> MetricScore:

        evidence   : List[str] = []
        violations : List[str] = []

        # 1. Scan for explicit FAIL signals first (hard override)
        for fail_sig in checkpoint.fail_signals:
            for line in llm_output.splitlines():
                if self._signal_matches(fail_sig, line):
                    violations.append(line.strip())

        # 2. Scan for PASS signals
        matched_pass_signals: List[str] = []
        for pass_sig in checkpoint.pass_signals:
            for line in llm_output.splitlines():
                if self.  _signal_matches(pass_sig, line):
                    if line.strip() not in evidence:
                        evidence.append(line.strip())
                    if pass_sig not in matched_pass_signals:
                        matched_pass_signals.append(pass_sig)
                    break   # signal found — don't double-count

        # 3. Determine result
        has_violations  = len(violations) > 0
        has_pass        = len(matched_pass_signals) > 0

        if checkpoint.requires_both:
            # scaler_no_leakage: need BOTH fit_transform(train) AND transform(test)
            result = self._score_requires_both(
                checkpoint, matched_pass_signals, violations
            )
        elif has_violations and not has_pass:
            result = MetricResult.FAIL
        elif has_violations and has_pass:
            # Both found — violations override
            result = MetricResult.FAIL
        elif has_pass:
            result = MetricResult.PASS
        else:
            result = MetricResult.FAIL

        # 4. Confidence — based on signal strength
        confidence = self._compute_confidence(
            matched_pass_signals, violations, checkpoint
        )

        return MetricScore(
            metric_id  = checkpoint.metric_id,
            result     = result,
            evidence   = evidence[:5],
            violations = violations[:5],
            confidence = confidence,
        )

    def _score_requires_both(
        self,
        checkpoint           : ScoringCheckpoint,
        matched_pass_signals : List[str],
        violations           : List[str],
    ) -> MetricResult:
        """
        scaler_no_leakage requires two specific signals to both be present.
        """
        # Must see: fit_transform on train, AND transform on test
        train_signals = [s for s in matched_pass_signals
                        if "train" in s.lower() or "x_train" in s.lower()]
        test_signals  = [s for s in matched_pass_signals
                        if "test" in s.lower() or "x_test" in s.lower()]

        if violations:
            return MetricResult.FAIL
        if train_signals and test_signals:
            return MetricResult.PASS

        return MetricResult.FAIL

    def _signal_matches(self, signal: str, text: str) -> bool:
        """
        Check if a pass/fail signal matches a line of LLM output.
        Supports: plain substring, regex pattern, case-insensitive.
        """
        try:
            return bool(re.search(signal, text, re.IGNORECASE))
        except re.error:
            # Not a valid regex — fall back to substring
            return signal.lower() in text.lower()

    def _compute_confidence(
        self,
        matched_pass : List[str],
        violations   : List[str],
        checkpoint   : ScoringCheckpoint,
    ) -> float:
        """
        Confidence score 0.0–1.0 based on signal coverage.
        """
        if not checkpoint.pass_signals:
            return 0.5

        signal_coverage = len(matched_pass) / len(checkpoint.pass_signals)

        if violations:
            # Violations reduce confidence significantly
            return max(0.0, signal_coverage - 0.4)

        return min(1.0, signal_coverage)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_trial_result(result: TrialResult) -> None:
    width = 68
    print("\n" + "═" * width)
    print(f"  TRIAL: {result.trial_id}  |  Pattern: {result.pattern_name}  |  Model: {result.model_name}")
    print(f"  Score: {result.score_pct:.1f}%  |  Time: {result.elapsed_seconds:.1f}s")
    print("─" * width)

    metric_labels = {
        "no_dummy_trap"              : "No dummy trap (drop='first')",
        "vif_check"                  : "VIF multicollinearity check",
        "heteroscedasticity_handled" : "Heteroscedasticity detected",
        "outliers_treated"           : "Outliers Winsorized/capped",
        "scaler_no_leakage"          : "Scaler fit on train only",
        "residuals_checked"          : "Residuals checked",
        "model_saves"                : "Model + scaler saved",
    }

    for metric_id, score in result.scores.items():
        label  = metric_labels.get(metric_id, metric_id)
        status = "✅" if score.passed else "❌"
        conf   = f"({score.confidence:.0%})"
        print(f"  {status}  {label:<42} {conf}")

        if score.evidence and score.passed:
            preview = score.evidence[0][:55]
            print(f"       ↳ {preview}")

        if score.violations:
            preview = score.violations[0][:55]
            print(f"       ⚠ VIOLATION: {preview}")

    print("═" * width + "\n")
