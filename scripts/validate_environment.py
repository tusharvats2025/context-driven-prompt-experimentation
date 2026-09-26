"""
validate_environment.py
=======================
Pre-flight checks before running the experiment.
Validates Python version, dependencies, Ollama, and file structure.
Run this before anything else.
"""

from __future__ import annotations
import sys
import os
import importlib
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

REQUIRED_PACKAGES = [
    ("pandas",       "pandas"),
    ("numpy",        "numpy"),
    ("sklearn",      "scikit-learn"),
    ("scipy",        "scipy"),
    ("statsmodels",  "statsmodels"),
    ("matplotlib",   "matplotlib"),
    ("requests",     "requests"),
    ("joblib",       "joblib"),
]

REQUIRED_FILES = [
    "prompts/patterns/01_zero_shot.txt",
    "prompts/patterns/02_few_shot.txt",
    "prompts/patterns/03_chain_of_thought.txt",
    "prompts/patterns/04_role_based.txt",
    "prompts/patterns/05_legacy_expert.txt",
    "prompts/patterns/06_cdp_pipeline.cdp",
    "src/cdp_types.py",
    "src/cdp_lexer.py",
    "src/cdp_parser.py",
    "src/cdp_validator.py",
    "src/cdp_checkpoint_extractor.py",
    "src/evaluator.py",
    "src/agent_runner.py",
    "data/generate_dataset.py",
]

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "

errors   = []
warnings = []


def check(label: str, ok: bool, detail: str = "", fatal: bool = True) -> None:
    if ok:
        print(f"{PASS} {label}")
    else:
        status = FAIL if fatal else WARN
        print(f"{status} {label}" + (f": {detail}" if detail else ""))
        if fatal:
            errors.append(label)
        else:
            warnings.append(label)


def section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


# ── Python version ─────────────────────────────────────────────────────────
section("Python Version")
major, minor = sys.version_info[:2]
check(f"Python {major}.{minor} (need ≥3.9)", major == 3 and minor >= 9,
      f"Got {major}.{minor}")

# ── Required packages ──────────────────────────────────────────────────────
section("Python Packages")
for import_name, pip_name in REQUIRED_PACKAGES:
    try:
        importlib.import_module(import_name)
        check(f"{pip_name}", True)
    except ImportError:
        check(f"{pip_name}", False, f"pip install {pip_name}", fatal=True)

# ── Required files ─────────────────────────────────────────────────────────
section("Project Files")
for rel_path in REQUIRED_FILES:
    full = PROJECT_ROOT / rel_path
    check(rel_path, full.exists(), f"Missing: {full}")

# ── Dataset ────────────────────────────────────────────────────────────────
section("Dataset")
csv_path = PROJECT_ROOT / "data" / "dataset_500rows.csv"
if csv_path.exists():
    check("dataset_500rows.csv exists", True)
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        check(f"Dataset shape {df.shape}", df.shape[0] >= 100,
              f"Only {df.shape[0]} rows — need ≥100")
        check("Target column 'salary' present", "salary" in df.columns)
    except Exception as e:
        check("Dataset readable", False, str(e))
else:
    print(f"{WARN} dataset_500rows.csv not found — run: python data/generate_dataset.py")
    warnings.append("dataset missing")

# ── Ollama ─────────────────────────────────────────────────────────────────
section("Ollama (LLM Runtime)")
try:
    import requests
    resp = requests.get("http://localhost:11434/", timeout=3)
    check("Ollama server running", True)

    # Check models
    resp2 = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in resp2.json().get("models", [])]
    if models:
        check(f"Models available: {', '.join(models[:3])}", True)
    else:
        check("Models available", False,
              "No models pulled. Run: ollama pull llama3.2", fatal=False)
except Exception:
    check("Ollama server running", False,
          "Start with: ollama serve", fatal=False)
    warnings.append("Ollama not running (needed for live runs; dry-run still works)")

# ── Output dirs ────────────────────────────────────────────────────────────
section("Output Directories")
for d in ["models/results", "analysis/outputs/tables", "analysis/outputs/figures"]:
    path = PROJECT_ROOT / d
    path.mkdir(parents=True, exist_ok=True)
    check(f"{d}/", True)

# ── CDP parser smoke test ──────────────────────────────────────────────────
section("CDP Parser Smoke Test")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from cdp_parser import CDPParser
    from cdp_validator import CDPValidator, ValidationLevel

    cdp_path = PROJECT_ROOT / "prompts" / "patterns" / "06_cdp_pipeline.cdp"
    parser   = CDPParser()
    schema   = parser.parse_file(str(cdp_path))

    check(f"CDP parsed: {schema.step_count()} steps found",
          schema.step_count() >= 10)
    check(f"Config keys: {len(schema.config)}",
          len(schema.config) >= 5)
    check(f"Pipeline name: {schema.pipeline.name}",
          bool(schema.pipeline.name))

    validator = CDPValidator()
    issues    = validator.validate(schema)
    errs      = [i for i in issues if i.level == ValidationLevel.ERROR]
    warns     = [i for i in issues if i.level == ValidationLevel.WARNING]
    check(f"CDP validation: {len(errs)} errors, {len(warns)} warnings",
          len(errs) == 0,
          f"{len(errs)} errors found", fatal=True)

except Exception as e:
    check("CDP parser", False, str(e), fatal=True)

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'═' * 50}")
if errors:
    print(f"  RESULT: ❌ {len(errors)} error(s) — fix before running")
    for e in errors:
        print(f"    • {e}")
elif warnings:
    print(f"  RESULT: ⚠️  Ready with {len(warnings)} warning(s)")
    for w in warnings:
        print(f"    • {w}")
    print("\n  Dry-run mode available: python src/agent_runner.py --dry-run")
else:
    print("  RESULT: ✅ All checks passed — ready to run!")
    print("\n  Next: python src/agent_runner.py --dry-run")
print('═' * 50 + "\n")

sys.exit(1 if errors else 0)
