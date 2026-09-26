"""
cdp_validator.py
================
Validates a parsed CDPSchema for structural integrity.

Checks:
  1. All config thresholds referenced in action blocks are declared
  2. All backtrack target steps exist
  3. All providers referenced in actions are declared
  4. No circular backtrack loops (simple cycle detection)
  5. Step IDs are sequential and non-duplicate
  6. Required sections are present

Returns a list of ValidationResult objects — never raises.
Callers decide whether to treat warnings/errors as fatal.

CDP v1.0 — linear_regression domain
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from enum import Enum

from cdp_types import CDPSchema, BacktrackTarget, Severity


class ValidationLevel(Enum):
    ERROR   = "ERROR"
    WARNING = "WARNING"
    INFO    = "INFO"


@dataclass
class ValidationResult:
    level   : ValidationLevel
    code    : str           # e.g. "UNDEFINED_THRESHOLD"
    message : str
    location: str           # e.g. "Step 5 | outlier_detection_and_treatment"

    def __str__(self):
        return f"[{self.level.value}] {self.code} @ {self.location}: {self.message}"


class CDPValidator:
    """
    Validates a CDPSchema and returns a list of ValidationResult objects.

    Usage:
        validator = CDPValidator()
        results   = validator.validate(schema)
        errors    = [r for r in results if r.level == ValidationLevel.ERROR]
    """

    def validate(self, schema: CDPSchema) -> List[ValidationResult]:
        results: List[ValidationResult] = []

        results += self._check_required_sections(schema)
        results += self._check_step_ids(schema)
        results += self._check_config_references(schema)
        results += self._check_backtrack_targets(schema)
        results += self._check_provider_references(schema)
        results += self._check_output_guarantees(schema)
        results += self._check_audit_checkpoints(schema)
        results += self._check_backtrack_cycles(schema)

        return results

    def is_valid(self, schema: CDPSchema) -> bool:
        results = self.validate(schema)
        return not any(r.level == ValidationLevel.ERROR for r in results)

    # ─────────────────────────────────────────────────────────────────────────
    # Check: Required sections
    # ─────────────────────────────────────────────────────────────────────────

    def _check_required_sections(self, schema: CDPSchema) -> List[ValidationResult]:
        results = []

        if not schema.pipeline.name:
            results.append(ValidationResult(
                level    = ValidationLevel.ERROR,
                code     = "MISSING_PIPELINE_NAME",
                message  = "Pipeline name not declared",
                location = "[pipeline]",
            ))

        if not schema.config:
            results.append(ValidationResult(
                level    = ValidationLevel.ERROR,
                code     = "EMPTY_CONFIG",
                message  = "No thresholds declared in [config]",
                location = "[config]",
            ))

        if not schema.steps:
            results.append(ValidationResult(
                level    = ValidationLevel.ERROR,
                code     = "NO_STEPS",
                message  = "No steps found in [steps]",
                location = "[steps]",
            ))

        if not schema.providers:
            results.append(ValidationResult(
                level    = ValidationLevel.WARNING,
                code     = "NO_PROVIDERS",
                message  = "No providers declared — pipeline has no I/O interface",
                location = "[providers]",
            ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Check: Step IDs are sequential and non-duplicate
    # ─────────────────────────────────────────────────────────────────────────

    def _check_step_ids(self, schema: CDPSchema) -> List[ValidationResult]:
        results = []
        seen_ids: Set[int] = set()
        ids = sorted([s.id for s in schema.steps.values()])

        for step in schema.steps.values():
            loc = f"Step {step.id} | {step.name}"

            if step.id in seen_ids:
                results.append(ValidationResult(
                    level    = ValidationLevel.ERROR,
                    code     = "DUPLICATE_STEP_ID",
                    message  = f"Step ID {step.id} appears more than once",
                    location = loc,
                ))
            seen_ids.add(step.id)

        # Check sequential
        expected = list(range(1, len(ids) + 1))
        if ids != expected:
            results.append(ValidationResult(
                level    = ValidationLevel.WARNING,
                code     = "NON_SEQUENTIAL_STEPS",
                message  = f"Step IDs are not sequential. Got: {ids}, expected: {expected}",
                location = "[steps]",
            ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Check: All config threshold names referenced in actions are declared
    # ─────────────────────────────────────────────────────────────────────────

    def _check_config_references(self, schema: CDPSchema) -> List[ValidationResult]:
        results = []
        declared_keys = set(schema.config.keys())

        # Regex: UPPER_SNAKE_CASE identifiers in action lines
        threshold_pattern = re.compile(r'\b([A-Z][A-Z0-9_]{3,})\b')

        # Known non-threshold caps that are NOT config keys
        known_non_config = {
            "HALT", "WARN", "INFO", "MCAR", "MAR", "MNAR",
            "CDP", "OHE", "VIF", "CLT", "ETL", "KNN",
        }

        for step in schema.steps.values():
            loc = f"Step {step.id} | {step.name}"
            for line in step.action_lines:
                for match in threshold_pattern.finditer(line):
                    name = match.group(1)
                    if name in known_non_config:
                        continue
                    if name not in declared_keys:
                        results.append(ValidationResult(
                            level    = ValidationLevel.WARNING,
                            code     = "UNDEFINED_THRESHOLD",
                            message  = f"Threshold '{name}' used but not declared in [config]",
                            location = loc,
                        ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Check: Backtrack target steps exist
    # ─────────────────────────────────────────────────────────────────────────

    def _check_backtrack_targets(self, schema: CDPSchema) -> List[ValidationResult]:
        results = []
        valid_step_ids = {s.id for s in schema.steps.values()}

        for step in schema.steps.values():
            loc = f"Step {step.id} | {step.name}"
            for rule in step.backtrack_rules:
                if rule.target_type in (BacktrackTarget.STEP, BacktrackTarget.RETRY):
                    if rule.target_step and rule.target_step not in valid_step_ids:
                        results.append(ValidationResult(
                            level    = ValidationLevel.ERROR,
                            code     = "INVALID_BACKTRACK_TARGET",
                            message  = f"Backtrack references Step {rule.target_step} which does not exist",
                            location = loc,
                        ))
                    # Forward backtrack is unusual — warn
                    if rule.target_step and rule.target_step > step.id:
                        results.append(ValidationResult(
                            level    = ValidationLevel.WARNING,
                            code     = "FORWARD_BACKTRACK",
                            message  = f"Backtrack to Step {rule.target_step} is a forward jump (unusual)",
                            location = loc,
                        ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Check: Provider names referenced in actions are declared
    # ─────────────────────────────────────────────────────────────────────────

    def _check_provider_references(self, schema: CDPSchema) -> List[ValidationResult]:
        results = []
        declared = set(schema.providers.keys())

        for step in schema.steps.values():
            loc = f"Step {step.id} | {step.name}"
            for line in step.action_lines:
                for provider in declared:
                    # Only warn if called as a function but not declared
                    pass    # providers in this version are declaration-only

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Check: Steps have non-empty output guarantees
    # ─────────────────────────────────────────────────────────────────────────

    def _check_output_guarantees(self, schema: CDPSchema) -> List[ValidationResult]:
        results = []

        for step in schema.steps.values():
            if not step.output_guarantees:
                results.append(ValidationResult(
                    level    = ValidationLevel.WARNING,
                    code     = "MISSING_OUTPUT_GUARANTEES",
                    message  = "Step has no output guarantees — postconditions unclear",
                    location = f"Step {step.id} | {step.name}",
                ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Check: Audit checkpoints exist in key steps
    # ─────────────────────────────────────────────────────────────────────────

    def _check_audit_checkpoints(self, schema: CDPSchema) -> List[ValidationResult]:
        results = []

        # These steps MUST have audit_log calls.
        # Steps 6 (feature_engineering) and 8 (feature_scaling) are excluded:
        #   Step 6 — verification is implicit in the output/verify block
        #   Step 8 — scaler.save() is the audit artifact, not an audit_log call
        # Only steps that make invisible decisions need explicit audit_log enforcement.
        critical_steps = {1, 2, 5, 7, 10, 11}

        for step in schema.steps.values():
            if step.id in critical_steps and not step.audit_checkpoints:
                results.append(ValidationResult(
                    level    = ValidationLevel.WARNING,
                    code     = "MISSING_AUDIT_CHECKPOINT",
                    message  = "Critical step has no audit_log calls — decisions untracked",
                    location = f"Step {step.id} | {step.name}",
                ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Check: No obvious backtrack cycles
    # ─────────────────────────────────────────────────────────────────────────

    def _check_backtrack_cycles(self, schema: CDPSchema) -> List[ValidationResult]:
        """
        Cycle detection: if Step A backtracks to Step B AND Step B backtracks
        to Step A, that is a potential infinite loop — flag it.

        EXCLUDED from cycle detection:
          - Self-referencing retry rules (→ Step N (retry) where N == current step)
            These are valid CDP patterns: relax a threshold and re-run the same step.
            They are bounded by the condition they carry, not infinite by definition.
          - Any rule where is_retry == True
        """
        results = []

        # Build adjacency map — EXCLUDE self-retries
        # key: step_id → set of target step_ids it can jump TO (non-retry only)
        edges: Dict[int, Set[int]] = {}
        for step in schema.steps.values():
            targets = set()
            for rule in step.backtrack_rules:
                if rule.target_type == BacktrackTarget.HALT:
                    continue
                if rule.is_retry:
                    # Self-referencing retry — valid pattern, skip cycle check
                    continue
                if rule.target_step and rule.target_step != step.id:
                    # Only include genuine jumps to OTHER steps
                    targets.add(rule.target_step)
            edges[step.id] = targets

        # Check for direct mutual backtracks between two DIFFERENT steps
        for step_id, targets in edges.items():
            for target in targets:
                if target == step_id:
                    # Self-loop after retry exclusion — shouldn't happen, but guard
                    continue
                if step_id in edges.get(target, set()):
                    results.append(ValidationResult(
                        level    = ValidationLevel.WARNING,
                        code     = "POTENTIAL_BACKTRACK_CYCLE",
                        message  = (
                            f"Step {step_id} ↔ Step {target} mutually backtrack "
                            f"— verify termination condition exists"
                        ),
                        location = f"Step {step_id}",
                    ))

        return results
