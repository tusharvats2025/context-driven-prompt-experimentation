"""
cdp_checkpoint_extractor.py
============================
Extracts experiment-scoring checkpoints from a parsed CDPSchema.

This is the bridge between the CDP parser and the experiment evaluator.
It answers: "Given a .cdp schema, what exactly must appear in LLM output
to prove the agent followed this pipeline?"

Output: List[ScoringCheckpoint] — one per experiment metric.

CDP v1.0 — linear_regression domain
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from cdp_types import CDPSchema, EXPERIMENT_METRICS, MetricDefinition


@dataclass
class ScoringCheckpoint:
    """
    A single scoreable requirement extracted from the CDP schema.
    Used by evaluator.py to score LLM output as PASS / FAIL.
    """
    metric_id       : str
    description     : str
    source_step_id  : int
    source_step_name: str

    # What the CDP schema says about this metric
    cdp_action_lines: List[str]         # relevant action lines from that step
    cdp_comment     : str               # CDP's own comment/justification

    # Scoring signals (for evaluator.py)
    pass_signals    : List[str]         # regex/substring patterns → PASS
    fail_signals    : List[str]         # regex/substring patterns → FAIL
    requires_both   : bool = False      # if True, ALL pass_signals must match

    # Resolved config values (for richer evaluation)
    config_values   : Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class CDPCheckpointExtractor:
    """
    Extracts ScoringCheckpoints from a CDPSchema.

    Usage:
        extractor   = CDPCheckpointExtractor()
        checkpoints = extractor.extract(schema)
        # → List[ScoringCheckpoint], one per experiment metric
    """

    def extract(self, schema: CDPSchema) -> List[ScoringCheckpoint]:
        checkpoints = []

        for metric_id, metric_def in EXPERIMENT_METRICS.items():
            step = schema.get_step(metric_def.source_step)
            if step is None:
                continue

            checkpoint = self._build_checkpoint(
                schema     = schema,
                metric_def = metric_def,
                step_id    = metric_def.source_step,
            )
            checkpoints.append(checkpoint)

        return checkpoints

    def extract_to_json(self, schema: CDPSchema, output_path: str) -> None:
        """Serialize extracted checkpoints to JSON for the evaluator."""
        checkpoints = self.extract(schema)
        data = {
            "cdp_source"    : schema.source_path,
            "cdp_version"   : schema.pipeline.version,
            "model_class"   : schema.pipeline.model_class,
            "checkpoint_count": len(checkpoints),
            "checkpoints"   : [cp.to_dict() for cp in checkpoints],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[CDPCheckpointExtractor] Wrote {len(checkpoints)} checkpoints → {output_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_checkpoint(
        self,
        schema     : CDPSchema,
        metric_def : MetricDefinition,
        step_id    : int,
    ) -> ScoringCheckpoint:

        step = schema.get_step(step_id)
        step_name = step.name if step else "unknown"

        # Pull relevant action lines from the step
        relevant_lines = self._extract_relevant_action_lines(
            schema, step_id, metric_def
        )

        # Pull CDP comments that explain this metric
        cdp_comment = self._extract_metric_comment(schema, step_id, metric_def)

        # Resolve config values referenced in this metric
        config_vals = self._resolve_config_for_metric(schema, metric_def)

        # Special case: scaler_no_leakage requires BOTH fit_transform(train)
        # AND transform(test) — one without the other is still a bug
        requires_both = (metric_def.metric_id == "scaler_no_leakage")

        return ScoringCheckpoint(
            metric_id        = metric_def.metric_id,
            description      = metric_def.description,
            source_step_id   = step_id,
            source_step_name = step_name,
            cdp_action_lines = relevant_lines,
            cdp_comment      = cdp_comment,
            pass_signals     = metric_def.pass_signals,
            fail_signals     = metric_def.fail_signals,
            requires_both    = requires_both,
            config_values    = config_vals,
        )

    def _extract_relevant_action_lines(
        self,
        schema    : CDPSchema,
        step_id   : int,
        metric_def: MetricDefinition,
    ) -> List[str]:
        """
        Pull action lines from the step that are directly relevant
        to this metric, based on keyword proximity.
        """
        step = schema.get_step(step_id)
        if not step:
            return []

        relevant = []
        keywords = set(metric_def.pass_signals + metric_def.fail_signals)

        for line in step.action_lines:
            line_lower = line.lower()
            for kw in keywords:
                if kw.lower().replace("'", "").replace('"', "") in line_lower:
                    relevant.append(line)
                    break

        # If nothing specific found, return all action lines for that step
        return relevant if relevant else step.action_lines[:5]

    def _extract_metric_comment(
        self,
        schema    : CDPSchema,
        step_id   : int,
        metric_def: MetricDefinition,
    ) -> str:
        """
        Pull the CDP's own explanatory comments about this metric.
        These are the // notes inside the step that justify the requirement.
        """
        step = schema.get_step(step_id)
        if not step:
            return ""

        # Look for comments containing metric-related keywords
        metric_keywords = {
            "no_dummy_trap"              : ["dummy", "multicollinearity", "drop="],
            "vif_check"                  : ["VIF", "multicollinearity", "coefficient"],
            "heteroscedasticity_handled" : ["heteroscedasticity", "Breusch", "variance"],
            "outliers_treated"           : ["outlier", "Winsor", "MODEL_CLASS"],
            "scaler_no_leakage"          : ["leakage", "fit on", "transform only"],
            "residuals_checked"          : ["assumption", "residual", "normality"],
            "model_saves"                : ["save", "persist", "model_output_path"],
        }

        target_kws = metric_keywords.get(metric_def.metric_id, [])
        matched_notes = []

        for note in step.notes:
            note_lower = note.lower()
            for kw in target_kws:
                if kw.lower() in note_lower:
                    matched_notes.append(note.lstrip("/ ").strip())
                    break

        return " | ".join(matched_notes) if matched_notes else ""

    def _resolve_config_for_metric(
        self,
        schema    : CDPSchema,
        metric_def: MetricDefinition,
    ) -> Dict[str, str]:
        """
        Extract the config values most relevant to this metric.
        """
        metric_config_map: Dict[str, List[str]] = {
            "no_dummy_trap"              : ["OHE_CARDINALITY_LIMIT"],
            "vif_check"                  : ["VIF_THRESHOLD", "CORRELATION_CUTOFF"],
            "heteroscedasticity_handled" : ["RESIDUAL_NORMALITY_ALPHA"],
            "outliers_treated"           : ["OUTLIER_WINSOR_PERCENTILE", "OUTLIER_MEAN_SHIFT_MAX"],
            "scaler_no_leakage"          : ["TRAIN_SPLIT_RATIO"],
            "residuals_checked"          : ["RESIDUAL_NORMALITY_ALPHA"],
            "model_saves"                : [],
        }

        keys = metric_config_map.get(metric_def.metric_id, [])
        return {
            k: str(schema.config.get(k, "N/A"))
            for k in keys
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: print checkpoint summary
# ─────────────────────────────────────────────────────────────────────────────

def print_checkpoint_summary(checkpoints: List[ScoringCheckpoint]) -> None:
    print("\n" + "═" * 70)
    print("  CDP CHECKPOINT EXTRACTION SUMMARY")
    print("═" * 70)

    for i, cp in enumerate(checkpoints, 1):
        print(f"\n  [{i}] {cp.metric_id}")
        print(f"       Description  : {cp.description}")
        print(f"       Source       : Step {cp.source_step_id} | {cp.source_step_name}")
        print(f"       CDP Comment  : {cp.cdp_comment[:80] if cp.cdp_comment else '—'}")
        print(f"       Pass Signals : {cp.pass_signals[:3]}")
        print(f"       Fail Signals : {cp.fail_signals[:3] if cp.fail_signals else '—'}")
        if cp.config_values:
            print(f"       Config       : {cp.config_values}")
        if cp.requires_both:
            print(f"       ⚠ Requires ALL pass signals to match")

    print("\n" + "═" * 70 + "\n")
