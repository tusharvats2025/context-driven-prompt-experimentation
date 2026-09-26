"""
cdp_lexer.py
============
Tokenizes a .cdp file into a flat list of typed tokens.
This is the first stage of the two-stage parse pipeline:

  Raw text → CDPLexer → List[Token] → CDPParser → CDPSchema

Token types are kept minimal and non-overlapping so the parser
can work deterministically without lookahead.

CDP v1.0 — linear_regression domain
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum, auto


# ─────────────────────────────────────────────────────────────────────────────
# Token Types
# ─────────────────────────────────────────────────────────────────────────────

class TokenType(Enum):
    # Top-level section headers
    SECTION_PIPELINE    = auto()    # [pipeline]
    SECTION_CONFIG      = auto()    # [config]
    SECTION_PROVIDERS   = auto()    # [providers]
    SECTION_GUARANTEES  = auto()    # [guarantees]
    SECTION_STEPS       = auto()    # [steps]

    # Step declaration
    STEP_HEADER         = auto()    # [Step N | step_name]

    # Block keywords inside a step
    BLOCK_INPUT         = auto()    # input
    BLOCK_ACTION        = auto()    # action
    BLOCK_BACKTRACK     = auto()    # backtrack
    BLOCK_OUTPUT        = auto()    # output
    BLOCK_VERIFY        = auto()    # verify

    # Control flow
    CTRL_HALT           = auto()    # HALT
    CTRL_WARN           = auto()    # WARN
    CTRL_BACKTRACK_JUMP = auto()    # → Step N [if ...]
    CTRL_BACKTRACK_RETRY= auto()    # → Step N (retry) [if ...]

    # Key-value pairs (config, providers, guarantees, pipeline)
    KEY_VALUE           = auto()    # key = value

    # Audit calls
    AUDIT_LOG           = auto()    # audit_log("event", ...)

    # Phase separator comments
    PHASE_COMMENT       = auto()    # // ── Phase N: Name ──

    # Iterative notes
    ITERATIVE_NOTE      = auto()    # // iterative note: ...

    # Step-level inline comments (not phase, not iterative)
    COMMENT             = auto()    # // ...

    # Generic content line (action pseudocode, input specs, output guarantees)
    CONTENT             = auto()

    # Blank / separator lines
    BLANK               = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Token
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Token:
    type        : TokenType
    text        : str               # original line text (stripped)
    line_number : int
    groups      : tuple = ()        # regex capture groups if any

    def __repr__(self):
        preview = self.text[:60].replace('\n', '')
        return f"Token({self.type.name}, L{self.line_number}: '{preview}')"


# ─────────────────────────────────────────────────────────────────────────────
# Lexer Rules — ordered by priority (first match wins)
# ─────────────────────────────────────────────────────────────────────────────

# Each rule: (TokenType, compiled_regex, use_groups)
_RULES: List[tuple] = [

    # ── Section headers ──────────────────────────────────────────────────────
    (TokenType.SECTION_PIPELINE,
     re.compile(r'^\[pipeline\]\s*$', re.IGNORECASE), False),

    (TokenType.SECTION_CONFIG,
     re.compile(r'^\[config\]\s*$', re.IGNORECASE), False),

    (TokenType.SECTION_PROVIDERS,
     re.compile(r'^\[providers\]\s*$', re.IGNORECASE), False),

    (TokenType.SECTION_GUARANTEES,
     re.compile(r'^\[guarantees\]\s*$', re.IGNORECASE), False),

    (TokenType.SECTION_STEPS,
     re.compile(r'^\[steps\]\s*$', re.IGNORECASE), False),

    # ── Step header ──────────────────────────────────────────────────────────
    # Matches: [Step 7 | multicollinearity_check]
    (TokenType.STEP_HEADER,
     re.compile(r'^\[Step\s+(\d+)\s*\|\s*([a-z_]+)\]', re.IGNORECASE), True),

    # ── Block keywords ───────────────────────────────────────────────────────
    (TokenType.BLOCK_INPUT,
     re.compile(r'^\s*input\s*$'), False),

    (TokenType.BLOCK_ACTION,
     re.compile(r'^\s*action\s*$'), False),

    (TokenType.BLOCK_BACKTRACK,
     re.compile(r'^\s*backtrack\s*$'), False),

    (TokenType.BLOCK_OUTPUT,
     re.compile(r'^\s*output\s*$'), False),

    (TokenType.BLOCK_VERIFY,
     re.compile(r'^\s*verify\s*.*$'), False),

    # ── Control flow ─────────────────────────────────────────────────────────
    # Matches: → Step 6 (retry) if VIF loop removes > 50% of features
    (TokenType.CTRL_BACKTRACK_RETRY,
     re.compile(r'→\s*Step\s*(\d+)\s*\(retry\)\s*(?:if\s+(.+))?', re.IGNORECASE), True),

    # Matches: → Step 3   if mean_shift > OUTLIER_MEAN_SHIFT_MAX
    (TokenType.CTRL_BACKTRACK_JUMP,
     re.compile(r'→\s*Step\s*(\d+)\s*(?:if\s+(.+))?', re.IGNORECASE), True),

    # Matches: HALT   if ...
    (TokenType.CTRL_HALT,
     re.compile(r'^\s*HALT\s*(?:if\s+(.+))?$'), True),

    # Matches: WARN   if ...
    (TokenType.CTRL_WARN,
     re.compile(r'^\s*WARN\s*(?:if\s+(.+))?$'), True),

    # ── Special comment types ─────────────────────────────────────────────────
    # Phase separator: // ── Phase N: Name ──
    (TokenType.PHASE_COMMENT,
     re.compile(r'^\s*//\s*─+\s*Phase\s+\d+.*─+', re.IGNORECASE), False),

    # Iterative note: // iterative note: ...
    (TokenType.ITERATIVE_NOTE,
     re.compile(r'^\s*//\s*iterative note\s*:', re.IGNORECASE), False),

    # ── Audit log call ───────────────────────────────────────────────────────
    # Matches: audit_log("intake", df.shape + df.dtypes)
    (TokenType.AUDIT_LOG,
     re.compile(r'audit_log\s*\(\s*["\']([^"\']+)["\']', re.IGNORECASE), True),

    # ── Key-value pairs ──────────────────────────────────────────────────────
    # Matches: name = linear_regression_pipeline
    # Matches: EMPTY_COL_THRESHOLD = 0.95
    # Matches: no_null_values = false
    (TokenType.KEY_VALUE,
     re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)(?:\s*//.*)?$'), True),

    # ── Generic comment ──────────────────────────────────────────────────────
    (TokenType.COMMENT,
     re.compile(r'^\s*//'), False),

    # ── Blank line ───────────────────────────────────────────────────────────
    (TokenType.BLANK,
     re.compile(r'^\s*$'), False),

    # ── Everything else is content ───────────────────────────────────────────
    (TokenType.CONTENT,
     re.compile(r'.+'), False),
]


# ─────────────────────────────────────────────────────────────────────────────
# Lexer
# ─────────────────────────────────────────────────────────────────────────────

class CDPLexer:
    """
    Tokenizes a .cdp file into a flat list of Token objects.

    Usage:
        lexer  = CDPLexer()
        tokens = lexer.tokenize(raw_cdp_text)
    """

    def tokenize(self, content: str) -> List[Token]:
        tokens: List[Token] = []

        for line_num, raw_line in enumerate(content.splitlines(), start=1):
            # Strip trailing whitespace; preserve leading indent for context
            line = raw_line.rstrip()

            # Strip the file-level separator lines (────) but keep them as BLANK
            if re.match(r'^\s*//\s*─{10,}', line):
                tokens.append(Token(TokenType.BLANK, line, line_num))
                continue

            token = self._match_line(line, line_num)
            tokens.append(token)

        return tokens

    def _match_line(self, line: str, line_num: int) -> Token:
        stripped = line.strip()

        for token_type, pattern, use_groups in _RULES:
            m = pattern.search(stripped) if token_type in (
                TokenType.CTRL_BACKTRACK_JUMP,
                TokenType.CTRL_BACKTRACK_RETRY,
                TokenType.AUDIT_LOG,
            ) else pattern.match(stripped)

            if m:
                groups = m.groups() if use_groups else ()
                return Token(token_type, stripped, line_num, groups)

        # Fallback — should never reach here with CONTENT catch-all
        return Token(TokenType.CONTENT, stripped, line_num)

    def debug_print(self, tokens: List[Token]) -> None:
        """Pretty-print token stream for development."""
        for t in tokens:
            if t.type == TokenType.BLANK:
                continue
            grp = f"  groups={t.groups}" if t.groups else ""
            print(f"  L{t.line_number:03d}  {t.type.name:<30}{grp}")
            print(f"         '{t.text[:80]}'")
