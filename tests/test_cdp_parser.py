"""
test_cdp_parser.py
==================
Unit tests for the CDP lexer, parser, validator, and checkpoint extractor.
Run with: pytest tests/test_cdp_parser.py -v

CDP v1.0 — linear_regression domain
"""

from __future__ import annotations
import sys
import json
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cdp_lexer               import CDPLexer, TokenType
from cdp_parser              import CDPParser
from cdp_validator           import CDPValidator, ValidationLevel
from cdp_checkpoint_extractor import CDPCheckpointExtractor
from cdp_types               import BacktrackTarget, Severity, EXPERIMENT_METRICS

CDP_FILE = PROJECT_ROOT / "prompts" / "patterns" / "06_cdp_pipeline.cdp"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_cdp() -> str:
    return CDP_FILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def schema(raw_cdp):
    parser = CDPParser()
    return parser.parse_string(raw_cdp, source_path=str(CDP_FILE))


@pytest.fixture(scope="module")
def checkpoints(schema):
    extractor = CDPCheckpointExtractor()
    return extractor.extract(schema)


# ─────────────────────────────────────────────────────────────────────────────
# Lexer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCDPLexer:

    def test_tokenizes_section_headers(self, raw_cdp):
        lexer  = CDPLexer()
        tokens = lexer.tokenize(raw_cdp)
        types  = [t.type for t in tokens]
        assert TokenType.SECTION_PIPELINE   in types
        assert TokenType.SECTION_CONFIG     in types
        assert TokenType.SECTION_PROVIDERS  in types
        assert TokenType.SECTION_GUARANTEES in types
        assert TokenType.SECTION_STEPS      in types

    def test_tokenizes_step_headers(self, raw_cdp):
        lexer  = CDPLexer()
        tokens = lexer.tokenize(raw_cdp)
        steps  = [t for t in tokens if t.type == TokenType.STEP_HEADER]
        assert len(steps) >= 10, f"Expected ≥10 steps, got {len(steps)}"

    def test_step_header_captures_id_and_name(self, raw_cdp):
        lexer  = CDPLexer()
        tokens = lexer.tokenize(raw_cdp)
        steps  = [t for t in tokens if t.type == TokenType.STEP_HEADER]
        first  = steps[0]
        assert first.groups[0] == "1"
        assert "data_intake" in first.groups[1]

    def test_tokenizes_backtrack_jumps(self, raw_cdp):
        lexer  = CDPLexer()
        tokens = lexer.tokenize(raw_cdp)
        jumps  = [t for t in tokens if t.type == TokenType.CTRL_BACKTRACK_JUMP]
        assert len(jumps) >= 3, f"Expected backtrack jumps, got {len(jumps)}"

    def test_tokenizes_halt(self, raw_cdp):
        lexer  = CDPLexer()
        tokens = lexer.tokenize(raw_cdp)
        halts  = [t for t in tokens if t.type == TokenType.CTRL_HALT]
        assert len(halts) >= 2

    def test_tokenizes_audit_log_calls(self, raw_cdp):
        lexer  = CDPLexer()
        tokens = lexer.tokenize(raw_cdp)
        audits = [t for t in tokens if t.type == TokenType.AUDIT_LOG]
        assert len(audits) >= 5

    def test_tokenizes_key_value_pairs(self, raw_cdp):
        lexer  = CDPLexer()
        tokens = lexer.tokenize(raw_cdp)
        kvs    = [t for t in tokens if t.type == TokenType.KEY_VALUE]
        assert len(kvs) >= 10


# ─────────────────────────────────────────────────────────────────────────────
# Parser tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCDPParser:

    def test_parses_pipeline_metadata(self, schema):
        assert schema.pipeline.name        == "linear_regression_pipeline"
        assert schema.pipeline.model_class == "linear"
        assert schema.pipeline.version     == "1.0"
        assert schema.pipeline.iterative   is True

    def test_parses_config_thresholds(self, schema):
        assert "EMPTY_COL_THRESHOLD"     in schema.config
        assert "VIF_THRESHOLD"           in schema.config
        assert "TRAIN_SPLIT_RATIO"       in schema.config
        assert "OUTLIER_WINSOR_PERCENTILE" in schema.config

    def test_config_value_types(self, schema):
        assert isinstance(schema.config["VIF_THRESHOLD"],          float)
        assert isinstance(schema.config["TRAIN_SPLIT_RATIO"],      float)
        assert isinstance(schema.config["OUTLIER_WINSOR_PERCENTILE"], list)
        assert schema.config["OUTLIER_WINSOR_PERCENTILE"] == [1, 99]

    def test_parses_guarantees(self, schema):
        assert schema.guarantees["no_null_values"]       is False
        assert schema.guarantees["data_is_pre_split"]    is False
        assert schema.guarantees["no_categorical_cols"]  is False
        assert schema.guarantees["outliers_pre_treated"] is False

    def test_parses_providers(self, schema):
        assert "csv_parser"        in schema.providers
        assert "audit_log"         in schema.providers
        assert "model_output_path" in schema.providers

    def test_parses_correct_step_count(self, schema):
        assert schema.step_count() == 12, \
            f"Expected 12 steps, got {schema.step_count()}"

    def test_step_ids_sequential(self, schema):
        ids = sorted([s.id for s in schema.steps.values()])
        assert ids == list(range(1, 13))

    def test_step_names_correct(self, schema):
        expected_names = [
            "data_intake", "data_quality_profile", "type_cast_and_parse",
            "missing_value_treatment", "outlier_detection_and_treatment",
            "feature_engineering", "multicollinearity_check", "feature_scaling",
            "feature_selection", "model_training", "assumption_validation",
            "evaluation_and_handoff",
        ]
        actual_names = list(schema.steps.keys())
        for name in expected_names:
            assert name in actual_names, f"Step '{name}' not found"

    def test_step_phases_assigned(self, schema):
        for step in schema.steps.values():
            assert step.phase, f"Step {step.id} has no phase"

    def test_step6_has_action_lines(self, schema):
        step6 = schema.get_step(6)
        assert step6 is not None
        assert len(step6.action_lines) > 0

    def test_step6_drop_first_in_actions(self, schema):
        step6 = schema.get_step(6)
        combined = " ".join(step6.action_lines + step6.notes)
        assert "drop='first'" in combined or "drop_first" in combined.lower(), \
            "Step 6 must enforce drop='first' for OHE"

    def test_step7_vif_in_actions(self, schema):
        step7 = schema.get_step(7)
        combined = " ".join(step7.action_lines)
        assert "VIF" in combined or "vif" in combined.lower()

    def test_step8_scaler_in_actions(self, schema):
        step8 = schema.get_step(8)
        combined = " ".join(step8.action_lines)
        assert "scaler" in combined.lower() or "StandardScaler" in combined

    def test_step11_breusch_pagan_in_actions(self, schema):
        step11 = schema.get_step(11)
        combined = " ".join(step11.action_lines)
        assert "Breusch_Pagan" in combined or "breusch" in combined.lower()

    def test_backtrack_rules_parsed(self, schema):
        """Every step with a backtrack block should have ≥1 rule."""
        steps_with_backtracks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for step_id in steps_with_backtracks:
            step = schema.get_step(step_id)
            assert step is not None
            assert len(step.backtrack_rules) >= 1, \
                f"Step {step_id} should have backtrack rules"

    def test_halt_rules_have_halt_target(self, schema):
        step1 = schema.get_step(1)
        halts = [r for r in step1.backtrack_rules
                 if r.target_type == BacktrackTarget.HALT]
        assert len(halts) >= 1

    def test_backtrack_jump_rules_have_target_step(self, schema):
        step2 = schema.get_step(2)
        jumps = [r for r in step2.backtrack_rules
                 if r.target_type == BacktrackTarget.STEP]
        if jumps:
            for rule in jumps:
                assert rule.target_step is not None
                assert rule.target_step >= 1

    def test_audit_checkpoints_extracted(self, schema):
        # Steps 1, 2, 5, 7, 10 must have audit checkpoints
        for step_id in [1, 2, 7, 10]:
            step = schema.get_step(step_id)
            assert step and len(step.audit_checkpoints) >= 1, \
                f"Step {step_id} should have audit checkpoints"

    def test_get_step_returns_none_for_invalid(self, schema):
        assert schema.get_step(99) is None
        assert schema.get_step(0)  is None

    def test_resolve_config_returns_value(self, schema):
        val = schema.resolve_config("VIF_THRESHOLD")
        assert val == 5.0

    def test_resolve_config_returns_none_for_unknown(self, schema):
        val = schema.resolve_config("NONEXISTENT_KEY")
        assert val is None


# ─────────────────────────────────────────────────────────────────────────────
# Validator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCDPValidator:

    def test_valid_schema_has_no_errors(self, schema):
        validator = CDPValidator()
        results   = validator.validate(schema)
        errors    = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0, \
            f"Valid schema should have no errors. Got: {[str(e) for e in errors]}"

    def test_is_valid_returns_true(self, schema):
        validator = CDPValidator()
        assert validator.is_valid(schema)

    def test_missing_pipeline_name_is_error(self):
        from cdp_types import CDPSchema, PipelineMetadata
        from collections import OrderedDict
        bad_schema = CDPSchema(
            source_path = "test",
            pipeline    = PipelineMetadata("", "linear", "1.0"),
            config      = {"KEY": 1},
            providers   = {},
            guarantees  = {},
            steps       = OrderedDict(),
            raw_content = "",
        )
        validator = CDPValidator()
        results   = validator.validate(bad_schema)
        codes     = [r.code for r in results]
        assert "MISSING_PIPELINE_NAME" in codes

    def test_sequential_step_ids_pass(self, schema):
        validator = CDPValidator()
        results   = validator.validate(schema)
        codes     = [r.code for r in results]
        assert "NON_SEQUENTIAL_STEPS" not in codes


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint extractor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCDPCheckpointExtractor:

    def test_extracts_all_7_metrics(self, checkpoints):
        metric_ids = {cp.metric_id for cp in checkpoints}
        for expected_id in EXPERIMENT_METRICS:
            assert expected_id in metric_ids, \
                f"Checkpoint for '{expected_id}' not extracted"

    def test_checkpoints_have_pass_signals(self, checkpoints):
        for cp in checkpoints:
            assert len(cp.pass_signals) > 0, \
                f"Checkpoint {cp.metric_id} has no pass signals"

    def test_scaler_checkpoint_requires_both(self, checkpoints):
        scaler_cp = next(cp for cp in checkpoints
                         if cp.metric_id == "scaler_no_leakage")
        assert scaler_cp.requires_both is True

    def test_vif_checkpoint_source_is_step7(self, checkpoints):
        vif_cp = next(cp for cp in checkpoints if cp.metric_id == "vif_check")
        assert vif_cp.source_step_id == 7

    def test_checkpoints_have_source_step_names(self, checkpoints):
        for cp in checkpoints:
            assert cp.source_step_name, \
                f"Checkpoint {cp.metric_id} has no source step name"

    def test_config_values_resolved(self, checkpoints):
        vif_cp = next(cp for cp in checkpoints if cp.metric_id == "vif_check")
        assert "VIF_THRESHOLD" in vif_cp.config_values
        assert vif_cp.config_values["VIF_THRESHOLD"] == "5.0"

    def test_to_dict_serializable(self, checkpoints):
        for cp in checkpoints:
            d = cp.to_dict()
            # Must be JSON-serializable
            json.dumps(d)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCDPEvaluator:

    def test_cdp_mock_output_scores_high(self, checkpoints):
        from evaluator import CDPEvaluator
        from src.agent_runner import MOCK_OUTPUTS

        evaluator = CDPEvaluator(checkpoints)
        result    = evaluator.score(
            llm_output   = MOCK_OUTPUTS["cdp"],
            trial_id     = "test_001",
            pattern_name = "cdp",
            model_name   = "test",
            elapsed      = 1.0,
        )
        # CDP mock output should pass at least 5/7 metrics
        assert result.score_pct >= 70, \
            f"CDP mock should score ≥70%, got {result.score_pct}%\n" \
            f"Failed: {result.failed_metrics}"

    def test_zero_shot_mock_output_scores_low(self, checkpoints):
        from evaluator import CDPEvaluator
        from src.agent_runner import MOCK_OUTPUTS

        evaluator = CDPEvaluator(checkpoints)
        result    = evaluator.score(
            llm_output   = MOCK_OUTPUTS["zero_shot"],
            trial_id     = "test_002",
            pattern_name = "zero_shot",
            model_name   = "test",
            elapsed      = 1.0,
        )
        # Zero-shot mock should fail most metrics
        assert result.score_pct <= 30, \
            f"Zero-shot mock should score ≤30%, got {result.score_pct}%"

    def test_scaler_leakage_detected(self, checkpoints):
        from evaluator import CDPEvaluator

        leaky_code = """
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # leakage — fit on full data
"""
        evaluator = CDPEvaluator(checkpoints)
        result    = evaluator.score(
            llm_output   = leaky_code,
            trial_id     = "test_003",
            pattern_name = "test",
            model_name   = "test",
            elapsed      = 1.0,
        )
        scaler_score = result.scores["scaler_no_leakage"]
        assert not scaler_score.passed, "Leaky scaler code should FAIL scaler metric"

    def test_trial_result_serializable(self, checkpoints):
        from evaluator import CDPEvaluator
        evaluator = CDPEvaluator(checkpoints)
        result    = evaluator.score("some code", "t1", "zero_shot", "llama", 1.0)
        d = result.to_dict()
        json.dumps(d)   # must not raise
