"""
cdp_parser.py
=============
Builds a CDPSchema AST from the token stream produced by CDPLexer.

Pipeline:
  CDPLexer → List[Token] → CDPParser → CDPSchema

The parser is a single-pass recursive descent parser.
It groups tokens by section → step → block, then delegates
each block to a dedicated handler.

CDP v1.0 — linear_regression domain
"""

from __future__ import annotations

import re
import ast
import json
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any

from cdp_lexer import CDPLexer, Token, TokenType
from cdp_types import (
    AuditCheckpoint, BacktrackRule, BacktrackTarget,
    CDPSchema, PipelineMetadata, Severity, Step,
)


# Phase label map — inferred from CDP file comments
_PHASE_LABELS: Dict[int, str] = {
    1:  "Intake & Validation",
    2:  "Cleaning",
    3:  "Feature Preparation",
    4:  "Training & Validation",
}

_STEP_PHASES: Dict[int, str] = {
    1: "Intake & Validation",
    2: "Intake & Validation",
    3: "Intake & Validation",
    4: "Cleaning",
    5: "Cleaning",
    6: "Feature Preparation",
    7: "Feature Preparation",
    8: "Feature Preparation",
    9: "Feature Preparation",
    10: "Training & Validation",
    11: "Training & Validation",
    12: "Training & Validation",
}


class CDPParseError(Exception):
    """Raised when the .cdp file has a structural error."""
    pass


class CDPParser:
    """
    Parses a .cdp file into a CDPSchema.

    Usage:
        parser = CDPParser()
        schema = parser.parse_file("prompts/patterns/06_cdp_pipeline.cdp")
        # or
        schema = parser.parse_string(raw_text, source_path="<inline>")
    """

    def __init__(self):
        self.lexer  = CDPLexer()
        self._tokens: List[Token] = []
        self._pos   : int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def parse_file(self, filepath: str) -> CDPSchema:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
        return self.parse_string(raw, source_path=filepath)

    def parse_string(self, content: str, source_path: str = "<string>") -> CDPSchema:
        self._tokens = self.lexer.tokenize(content)
        self._pos    = 0

        pipeline   = PipelineMetadata("", "", "1.0", True)
        config     : Dict[str, Any]  = {}
        providers  : Dict[str, str]  = {}
        guarantees : Dict[str, bool] = {}
        steps      : OrderedDict[str, Step] = OrderedDict()

        while self._pos < len(self._tokens):
            tok = self._peek()

            if tok.type == TokenType.SECTION_PIPELINE:
                self._advance()
                pipeline = self._parse_pipeline_section()

            elif tok.type == TokenType.SECTION_CONFIG:
                self._advance()
                config = self._parse_kv_section(cast_values=True)

            elif tok.type == TokenType.SECTION_PROVIDERS:
                self._advance()
                providers = self._parse_kv_section(cast_values=False)

            elif tok.type == TokenType.SECTION_GUARANTEES:
                self._advance()
                guarantees = self._parse_guarantees_section()

            elif tok.type == TokenType.SECTION_STEPS:
                self._advance()
                steps = self._parse_steps_section()

            else:
                self._advance()     # skip blanks, comments, separators

        return CDPSchema(
            source_path  = source_path,
            pipeline     = pipeline,
            config       = config,
            providers    = providers,
            guarantees   = guarantees,
            steps        = steps,
            raw_content  = content,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Section Parsers
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_pipeline_section(self) -> PipelineMetadata:
        kv = self._parse_kv_section(cast_values=False)
        return PipelineMetadata(
            name        = kv.get("name", ""),
            model_class = kv.get("model_class", ""),
            version     = kv.get("version", "1.0"),
            iterative   = str(kv.get("iterative", "true")).lower() == "true",
        )

    def _parse_kv_section(self, cast_values: bool) -> Dict[str, Any]:
        """
        Consumes KEY_VALUE tokens until the next section or step header.
        """
        result: Dict[str, Any] = {}

        while self._pos < len(self._tokens):
            tok = self._peek()

            if self._is_section_start(tok):
                break

            if tok.type == TokenType.KEY_VALUE and tok.groups:
                key   = tok.groups[0].strip()
                value = tok.groups[1].strip()
                # Strip trailing inline comments
                value = re.sub(r'\s*//.*$', '', value).strip()

                if cast_values:
                    result[key] = self._cast_value(value)
                else:
                    result[key] = value

            self._advance()

        return result

    def _parse_guarantees_section(self) -> Dict[str, bool]:
        kv = self._parse_kv_section(cast_values=False)
        return {
            k: str(v).lower() == "true"
            for k, v in kv.items()
        }

    def _parse_steps_section(self) -> OrderedDict[str, Step]:
        steps: OrderedDict[str, Step] = OrderedDict()

        while self._pos < len(self._tokens):
            tok = self._peek()

            if self._is_top_section(tok):
                break

            if tok.type == TokenType.STEP_HEADER and tok.groups:
                step = self._parse_single_step(tok)
                steps[step.name] = step
            else:
                self._advance()

        return steps

    # ─────────────────────────────────────────────────────────────────────────
    # Step Parser
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_single_step(self, header_tok: Token) -> Step:
        step_id   = int(header_tok.groups[0])
        step_name = header_tok.groups[1].strip()
        self._advance()     # consume the STEP_HEADER token

        input_specs       : List[str]             = []
        action_lines      : List[str]             = []
        backtrack_rules   : List[BacktrackRule]   = []
        output_guarantees : List[str]             = []
        audit_checkpoints : List[AuditCheckpoint] = []
        notes             : List[str]             = []
        iterative_note    : Optional[str]         = None

        current_block : Optional[str] = None

        while self._pos < len(self._tokens):
            tok = self._peek()

            # Stop at next step or top-level section
            if tok.type == TokenType.STEP_HEADER or self._is_top_section(tok):
                break

            # Block transitions
            if tok.type == TokenType.BLOCK_INPUT:
                current_block = "input"
                self._advance(); continue

            if tok.type == TokenType.BLOCK_ACTION:
                current_block = "action"
                self._advance(); continue

            if tok.type == TokenType.BLOCK_BACKTRACK:
                current_block = "backtrack"
                self._advance(); continue

            if tok.type == TokenType.BLOCK_OUTPUT:
                current_block = "output"
                self._advance(); continue

            if tok.type in (TokenType.BLOCK_VERIFY, TokenType.BLANK,
                            TokenType.PHASE_COMMENT):
                self._advance(); continue

            # Iterative note
            if tok.type == TokenType.ITERATIVE_NOTE:
                iterative_note = tok.text
                self._advance(); continue

            # Regular comment — store as note
            if tok.type == TokenType.COMMENT:
                notes.append(tok.text)
                self._advance(); continue

            # Audit log call — extract from action block
            if tok.type == TokenType.AUDIT_LOG:
                event = tok.groups[0] if tok.groups else tok.text
                cp = AuditCheckpoint(
                    step_id   = step_id,
                    step_name = step_name,
                    event_name= event,
                    context   = tok.text,
                )
                audit_checkpoints.append(cp)
                if current_block == "action":
                    action_lines.append(tok.text)
                self._advance(); continue

            # Control flow tokens inside backtrack block
            if current_block == "backtrack":
                rule = self._parse_backtrack_token(tok, step_id)
                if rule:
                    backtrack_rules.append(rule)
                self._advance(); continue

            # Content dispatch by current block
            if tok.type == TokenType.CONTENT:
                if current_block == "input":
                    input_specs.append(tok.text)
                elif current_block == "action":
                    action_lines.append(tok.text)
                elif current_block == "output":
                    output_guarantees.append(tok.text)
                else:
                    notes.append(tok.text)

            self._advance()

        return Step(
            id                 = step_id,
            name               = step_name,
            phase              = _STEP_PHASES.get(step_id, "Unknown"),
            input_specs        = input_specs,
            action_lines       = action_lines,
            backtrack_rules    = backtrack_rules,
            output_guarantees  = output_guarantees,
            audit_checkpoints  = audit_checkpoints,
            notes              = notes,
            iterative_note     = iterative_note,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Backtrack Rule Parser
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_backtrack_token(self, tok: Token, step_id: int) -> Optional[BacktrackRule]:

        if tok.type == TokenType.CTRL_HALT:
            condition = tok.groups[0].strip() if tok.groups and tok.groups[0] else "always"
            return BacktrackRule(
                raw_text     = tok.text,
                condition    = condition,
                target_type  = BacktrackTarget.HALT,
                target_step  = None,
                is_retry     = False,
                severity     = Severity.HALT,
            )

        if tok.type == TokenType.CTRL_WARN:
            condition = tok.groups[0].strip() if tok.groups and tok.groups[0] else "always"
            return BacktrackRule(
                raw_text     = tok.text,
                condition    = condition,
                target_type  = BacktrackTarget.HALT,
                target_step  = None,
                is_retry     = False,
                severity     = Severity.WARN,
            )

        if tok.type == TokenType.CTRL_BACKTRACK_RETRY:
            step_num  = int(tok.groups[0]) if tok.groups and tok.groups[0] else step_id
            condition = tok.groups[1].strip() if tok.groups and len(tok.groups) > 1 and tok.groups[1] else "always"
            return BacktrackRule(
                raw_text     = tok.text,
                condition    = condition,
                target_type  = BacktrackTarget.RETRY,
                target_step  = step_num,
                is_retry     = True,
                severity     = Severity.WARN,
            )

        if tok.type == TokenType.CTRL_BACKTRACK_JUMP:
            step_num  = int(tok.groups[0]) if tok.groups and tok.groups[0] else None
            condition = tok.groups[1].strip() if tok.groups and len(tok.groups) > 1 and tok.groups[1] else "always"
            return BacktrackRule(
                raw_text     = tok.text,
                condition    = condition,
                target_type  = BacktrackTarget.STEP,
                target_step  = step_num,
                is_retry     = False,
                severity     = Severity.WARN,
            )

        # Content line inside backtrack block — treat as condition description
        if tok.type == TokenType.CONTENT:
            # Try to infer from raw text
            raw = tok.text
            if 'HALT' in raw:
                cond = raw.split('if', 1)[-1].strip() if 'if' in raw else raw
                return BacktrackRule(
                    raw_text    = raw,
                    condition   = cond,
                    target_type = BacktrackTarget.HALT,
                    severity    = Severity.HALT,
                )
            m = re.search(r'→\s*Step\s*(\d+)', raw)
            if m:
                cond = raw.split('if', 1)[-1].strip() if 'if' in raw else "always"
                is_retry = "(retry)" in raw.lower()
                return BacktrackRule(
                    raw_text    = raw,
                    condition   = cond,
                    target_type = BacktrackTarget.RETRY if is_retry else BacktrackTarget.STEP,
                    target_step = int(m.group(1)),
                    is_retry    = is_retry,
                    severity    = Severity.WARN,
                )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _is_section_start(self, tok: Token) -> bool:
        return tok.type in (
            TokenType.SECTION_PIPELINE,
            TokenType.SECTION_CONFIG,
            TokenType.SECTION_PROVIDERS,
            TokenType.SECTION_GUARANTEES,
            TokenType.SECTION_STEPS,
            TokenType.STEP_HEADER,
        )

    def _is_top_section(self, tok: Token) -> bool:
        return tok.type in (
            TokenType.SECTION_PIPELINE,
            TokenType.SECTION_CONFIG,
            TokenType.SECTION_PROVIDERS,
            TokenType.SECTION_GUARANTEES,
            TokenType.SECTION_STEPS,
        )

    def _cast_value(self, raw: str) -> Any:
        """
        Cast config values to appropriate Python types.
        Examples:
          "0.95"       → 0.95
          "[1, 99]"    → [1, 99]
          "true"       → True
          "5.0"        → 5.0
          "linear"     → "linear"
        """
        # Boolean
        if raw.lower() == "true":  return True
        if raw.lower() == "false": return False

        # List literal
        if raw.startswith("["):
            try:
                return ast.literal_eval(raw)
            except Exception:
                return raw

        # Numeric
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            pass

        return raw
