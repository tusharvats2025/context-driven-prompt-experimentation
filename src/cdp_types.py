"""
cdp_types.py
============
Core dataclasses for the CDP (Context-Driven Prompting) schema.
These are the AST nodes produced by CDPParser and consumed by
CDPInterpreter, CheckpointExtractor, and the Evaluator.

CDP v1.0 — linear_regression domain
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, OrderedDict as ODict
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Severity(Enum):
    HALT = "HALT"       # fatal — pipeline must stop
    WARN = "WARN"       # non-fatal — log and continue
    INFO = "INFO"       # informational only


class BacktrackTarget(Enum):
    HALT  = "HALT"
    RETRY = "RETRY"     # same step, retry with relaxed params
    STEP  = "STEP"      # jump to specific step number


# ─────────────────────────────────────────────────────────────────────────────
# Core AST Nodes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineMetadata:
    name        : str
    model_class : str
    version     : str
    iterative   : bool = True


@dataclass
class BacktrackRule:
    """
    A single conditional jump rule inside a step's [backtrack] block.

    Examples from linear_regression.cdp:
      → Step 3   if mean_shift > OUTLIER_MEAN_SHIFT_MAX
      HALT       if target_column not found after load
      → Step 6 (retry)  if OHE creates > 50 cols from single feature
    """
    raw_text       : str                        # original line from .cdp
    condition      : str                        # condition string (unparsed)
    target_type    : BacktrackTarget
    target_step    : Optional[int] = None       # set when target_type == STEP
    is_retry       : bool = False               # same step, relaxed params
    severity       : Severity = Severity.HALT
    resolved_config: Dict[str, Any] = field(default_factory=dict)  # threshold values


@dataclass
class AuditCheckpoint:
    """
    An audit_log() call extracted from a step's action block.
    These become the scoring signals in evaluator.py.
    """
    step_id      : int
    step_name    : str
    event_name   : str                          # e.g. "dropped_VIF", "MNAR_flag"
    context      : str                          # surrounding code line
    metric_id    : Optional[str] = None         # maps to one of 7 experiment metrics


@dataclass
class Step:
    id                  : int
    name                : str
    phase               : str                   # e.g. "Intake", "Cleaning", "Training"
    input_specs         : List[str]
    action_lines        : List[str]             # raw pseudocode lines
    backtrack_rules     : List[BacktrackRule]
    output_guarantees   : List[str]
    audit_checkpoints   : List[AuditCheckpoint]
    notes               : List[str]             # // comment lines inside step
    iterative_note      : Optional[str] = None  # "// iterative note:" lines


@dataclass
class CDPSchema:
    """
    Complete parsed representation of a .cdp file.
    Root node of the AST.
    """
    source_path  : str
    pipeline     : PipelineMetadata
    config       : Dict[str, Any]               # all threshold values
    providers    : Dict[str, str]               # name → signature
    guarantees   : Dict[str, bool]              # developer assertions
    steps        : ODict[str, Step]             # ordered: step_name → Step
    raw_content  : str                          # original file text

    def get_step(self, step_id: int) -> Optional[Step]:
        for step in self.steps.values():
            if step.id == step_id:
                return step
        return None

    def resolve_config(self, key: str) -> Any:
        """Resolve a threshold name to its declared value."""
        return self.config.get(key, None)

    def step_count(self) -> int:
        return len(self.steps)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Metrics — the 7 binary pass/fail criteria
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricDefinition:
    metric_id    : str
    description  : str
    source_step  : int                          # which CDP step governs this
    pass_signals : List[str]                    # strings that indicate PASS in LLM output
    fail_signals : List[str]                    # strings that indicate FAIL
    weight       : float = 1.0                  # future: weighted scoring


EXPERIMENT_METRICS: Dict[str, MetricDefinition] = {

    "no_dummy_trap": MetricDefinition(
        metric_id    = "no_dummy_trap",
        description  = "Used drop='first' or equivalent in OneHotEncoding",
        source_step  = 6,
        pass_signals = [
            "drop='first'",
            'drop="first"',
            "drop_first=True",
            "drop_first = True",
            "pd.get_dummies(.*drop_first=True)",
        ],
        fail_signals = [
            "pd.get_dummies(df)",
            "OneHotEncoder()",
            "get_dummies(X)",
        ],
    ),

    "vif_check": MetricDefinition(
        metric_id    = "vif_check",
        description  = "Checked multicollinearity via VIF",
        source_step  = 7,
        pass_signals = [
            "variance_inflation_factor",
            "VIF",
            "vif",
            "from statsmodels.stats.outliers_influence",
            "calc_vif",
            "multicollinearity",
        ],
        fail_signals = [],
    ),

    "heteroscedasticity_handled": MetricDefinition(
        metric_id    = "heteroscedasticity_handled",
        description  = "Detected and handled heteroscedasticity",
        source_step  = 11,
        pass_signals = [
            "Breusch_Pagan",
            "breuschpagan",
            "het_breuschpagan",
            "heteroscedasticity",
            "bp_p",
            "bp_stat",
            "White test",
            "white_test",
        ],
        fail_signals = [],
    ),

    "outliers_treated": MetricDefinition(
        metric_id    = "outliers_treated",
        description  = "Outliers Winsorized or capped",
        source_step  = 5,
        pass_signals = [
            "Winsor",
            "winsor",
            "clip(",
            "percentile",
            "IQR",
            "iqr",
            "zscore",
            "z_score",
            "OUTLIER_WINSOR_PERCENTILE",
        ],
        fail_signals = [
            "dropna",          # wrong treatment
            "fillna(0)",       # wrong treatment
        ],
    ),

    "scaler_no_leakage": MetricDefinition(
        metric_id    = "scaler_no_leakage",
        description  = "Scaler fit on train only, transform on test",
        source_step  = 8,
        pass_signals = [
            "fit_transform(X_train",
            "fit_transform(x_train",
            ".transform(X_test",
            ".transform(x_test",
            "fit on train",
        ],
        fail_signals = [
            "fit_transform(X_test",
            "fit_transform(x_test",
            "fit_transform(df)",
            "fit_transform(X)",
            "StandardScaler().fit_transform(X)",
        ],
    ),

    "residuals_checked": MetricDefinition(
        metric_id    = "residuals_checked",
        description  = "Residual normality or homoscedasticity test performed",
        source_step  = 11,
        pass_signals = [
            "Shapiro",
            "shapiro",
            "shapiro_wilk",
            "Durbin",
            "durbin",
            "durbin_watson",
            "residual",
            "normality",
            "Q-Q plot",
            "qqplot",
        ],
        fail_signals = [],
    ),

    "model_saves": MetricDefinition(
        metric_id    = "model_saves",
        description  = "Model and scaler persisted to disk",
        source_step  = 12,
        pass_signals = [
            "joblib.dump",
            "pickle.dump",
            "model.save",
            ".pkl",
            "save_model",
            "model_output_path",
            "scaler.pkl",
            "model.pkl",
        ],
        fail_signals = [],
    ),
}
