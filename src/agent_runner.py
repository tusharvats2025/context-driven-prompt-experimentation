"""
agent_runner.py
===============
Orchestrates the experiment:
  1. Loads all 6 prompt patterns
  2. Sends each to Ollama (configurable model)
  3. Captures raw output + timing
  4. Passes output to CDPEvaluator
  5. Saves TrialResult JSON per trial

Supports:
  - Multiple LLM models (llama3.2, mistral, etc.)
  - Configurable trial repetitions (default: 3 per pattern)
  - Resume from last saved trial (idempotent)
  - Dry-run mode (skips API calls, uses mock output)

CDP v1.0 — linear_regression domain
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from cdp_parser              import CDPParser
from cdp_validator           import CDPValidator, ValidationLevel
from cdp_checkpoint_extractor import CDPCheckpointExtractor, print_checkpoint_summary
from evaluator               import CDPEvaluator, TrialResult, print_trial_result
from cdp_types               import EXPERIMENT_METRICS


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "[%(levelname)s] %(asctime)s %(name)s: %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("agent_runner")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).parent.parent
PROMPTS_DIR    = PROJECT_ROOT / "prompts" / "patterns"
RESULTS_DIR    = PROJECT_ROOT / "models" / "results"
CDP_FILE       = PROMPTS_DIR / "06_cdp_pipeline.cdp"
DATA_CSV       = PROJECT_ROOT / "data" / "dataset_500rows.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PATTERN_FILES  = {
    "zero_shot"      : "01_zero_shot.txt",
    "few_shot"       : "02_few_shot.txt",
    "chain_of_thought": "03_chain_of_thought.txt",
    "role_based"     : "04_role_based.txt",
    "legacy_expert"  : "05_legacy_expert.txt",
    "cdp"            : "06_cdp_pipeline.cdp",
}

OLLAMA_URL     = "http://localhost:11434/api/generate"
DEFAULT_MODELS = ["llama3.2"]
DEFAULT_TRIALS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Ollama client
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL, timeout: int = 300):
        self.base_url = base_url
        self.timeout  = timeout

    def generate(self, model: str, prompt: str) -> Tuple[str, float]:
        """
        Send prompt to Ollama. Returns (output_text, elapsed_seconds).
        """
        start = time.time()
        try:
            resp = requests.post(
                self.base_url,
                json    = {
                    "model"  : model,
                    "prompt" : prompt,
                    "stream" : False,
                    "options": {
                        "temperature": 0.2,     # low temp for reproducibility
                        "num_predict": 2048,
                    },
                },
                timeout = self.timeout,
            )
            resp.raise_for_status()
            data    = resp.json()
            output  = data.get("response", "")
            elapsed = time.time() - start
            return output, elapsed

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama at localhost:11434. "
                "Start it with: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama request timed out after {self.timeout}s")

    def list_models(self) -> List[str]:
        try:
            resp = requests.get(
                self.base_url.replace("/api/generate", "/api/tags"),
                timeout=10,
            )
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def is_available(self) -> bool:
        try:
            requests.get(
                self.base_url.replace("/api/generate", "/"),
                timeout=3,
            )
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Prompt loader
# ─────────────────────────────────────────────────────────────────────────────

def load_prompt(pattern_name: str, csv_path: str) -> str:
    """
    Load a prompt file and inject the dataset path reference.
    For CDP pattern, the entire .cdp file IS the prompt (prepended with task).
    """
    filename = PATTERN_FILES.get(pattern_name)
    if not filename:
        raise ValueError(f"Unknown pattern: {pattern_name}")

    filepath = PROMPTS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Inject dataset path
    content = content.replace("{{DATA_CSV}}", csv_path)
    content = content.replace("{{TARGET_COLUMN}}", "salary")

    # CDP pattern: wrap with execution instruction
    if pattern_name == "cdp":
        content = (
            "You are a data science agent. Execute the following CDP pipeline "
            "exactly as specified. Generate complete, runnable Python code.\n\n"
            f"Dataset: {csv_path}\nTarget column: salary\n\n"
            f"{content}"
        )

    return content


# ─────────────────────────────────────────────────────────────────────────────
# Trial ID management
# ─────────────────────────────────────────────────────────────────────────────

def get_trial_id(model: str, pattern: str, trial_num: int) -> str:
    return f"trial_{trial_num:03d}_{model.replace(':', '_')}_{pattern}"


def trial_already_run(trial_id: str) -> bool:
    result_path = RESULTS_DIR / f"{trial_id}.json"
    return result_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Mock output (dry-run mode)
# ─────────────────────────────────────────────────────────────────────────────

MOCK_OUTPUTS: Dict[str, str] = {
    "zero_shot": """
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset_500rows.csv')
X = df.drop('salary', axis=1)
y = df['salary']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
""",
    "cdp": """
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import warnings

# Step 1: Data Intake
df = pd.read_csv('dataset_500rows.csv')
target_column = 'salary'
assert target_column in df.columns, "Target not found"

# Step 4: Missing value treatment
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)

# Step 5: Outlier treatment
for col in df.select_dtypes(include='number').columns:
    lower = np.percentile(df[col], 1)
    upper = np.percentile(df[col], 99)
    df[col] = df[col].clip(lower, upper)

# Step 6: Feature Engineering — OHE with drop='first' (avoid dummy trap)
df = pd.get_dummies(df, drop_first=True)

# Step 7: VIF multicollinearity check
X = df.drop('salary', axis=1)
y = df['salary']

def calc_vif(X):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

while True:
    vif = calc_vif(X)
    worst = vif.loc[vif['VIF'].idxmax()]
    if worst['VIF'] > 5.0:
        X = X.drop(columns=[worst['feature']])
    else:
        break

# Step 8: Feature scaling — fit on train ONLY to avoid leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Step 10: Training
model = LinearRegression()
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
residuals = y_test.values - y_pred

# Step 11: Assumption validation
# Heteroscedasticity — Breusch-Pagan test
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_test)
if bp_p < 0.05:
    print(f"WARNING: Heteroscedasticity detected (bp_p={bp_p:.4f})")

# Normality — Shapiro-Wilk
sw_stat, sw_p = stats.shapiro(residuals[:200])

# Step 12: Save model + scaler
joblib.dump(model, 'model_output/model.pkl')
joblib.dump(scaler, 'model_output/scaler.pkl')
print(f"CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRunner:

    def __init__(
        self,
        models    : List[str],
        patterns  : List[str],
        n_trials  : int,
        dry_run   : bool = False,
        resume    : bool = True,
    ):
        self.models   = models
        self.patterns = patterns
        self.n_trials = n_trials
        self.dry_run  = dry_run
        self.resume   = resume
        self.client   = OllamaClient()

        # Parse CDP schema and build evaluator
        log.info("Parsing CDP schema...")
        parser      = CDPParser()
        self.schema = parser.parse_file(str(CDP_FILE))

        validator = CDPValidator()
        issues    = validator.validate(self.schema)
        errors    = [i for i in issues if i.level == ValidationLevel.ERROR]
        warnings  = [i for i in issues if i.level == ValidationLevel.WARNING]

        if errors:
            for e in errors:
                log.error(str(e))
            raise RuntimeError("CDP schema has validation errors. Fix before running.")

        if warnings:
            for w in warnings:
                log.warning(str(w))

        log.info(f"CDP schema parsed: {self.schema.step_count()} steps, "
                 f"{len(self.schema.config)} config keys")

        extractor   = CDPCheckpointExtractor()
        checkpoints = extractor.extract(self.schema)
        print_checkpoint_summary(checkpoints)

        self.evaluator = CDPEvaluator(checkpoints)

    def run(self) -> List[TrialResult]:
        if not self.dry_run and not self.client.is_available():
            raise RuntimeError(
                "Ollama not running. Start with: ollama serve\n"
                "Then pull a model: ollama pull llama3.2"
            )

        all_results: List[TrialResult] = []

        for model in self.models:
            for pattern in self.patterns:
                for trial_num in range(1, self.n_trials + 1):
                    trial_id = get_trial_id(model, pattern, trial_num)

                    if self.resume and trial_already_run(trial_id):
                        log.info(f"SKIP (already done): {trial_id}")
                        # Load existing result
                        with open(RESULTS_DIR / f"{trial_id}.json") as f:
                            pass    # just skip
                        continue

                    log.info(f"Running trial: {trial_id}")
                    result = self._run_single_trial(
                        trial_id   = trial_id,
                        model      = model,
                        pattern    = pattern,
                        trial_num  = trial_num,
                    )

                    # Save immediately
                    result_path = RESULTS_DIR / f"{trial_id}.json"
                    result.raw_output_path = str(result_path)
                    result.save(str(result_path))
                    log.info(f"Saved: {result_path}")

                    print_trial_result(result)
                    all_results.append(result)

        return all_results

    def _run_single_trial(
        self,
        trial_id  : str,
        model     : str,
        pattern   : str,
        trial_num : int,
    ) -> TrialResult:

        if self.dry_run:
            output  = MOCK_OUTPUTS.get(pattern, MOCK_OUTPUTS["zero_shot"])
            elapsed = 0.5
        else:
            try:
                prompt  = load_prompt(pattern, str(DATA_CSV))
            except FileNotFoundError:
                log.warning(f"Prompt file not found for {pattern} — using placeholder")
                prompt  = f"Perform linear regression on {DATA_CSV} to predict salary."

            output, elapsed = self.client.generate(model, prompt)

        result = self.evaluator.score(
            llm_output   = output,
            trial_id     = trial_id,
            pattern_name = pattern,
            model_name   = model,
            elapsed      = elapsed,
        )

        # Also save raw output
        raw_path = RESULTS_DIR / f"{trial_id}_raw.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(output)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CDP Experiment Runner")
    p.add_argument("--models",   nargs="+", default=DEFAULT_MODELS,
                   help="Ollama model names to test")
    p.add_argument("--patterns", nargs="+", default=list(PATTERN_FILES.keys()),
                   help="Prompt patterns to run")
    p.add_argument("--trials",   type=int, default=DEFAULT_TRIALS,
                   help="Number of trials per pattern per model")
    p.add_argument("--dry-run",  action="store_true",
                   help="Use mock outputs instead of calling Ollama")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-run all trials even if results exist")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    runner = ExperimentRunner(
        models   = args.models,
        patterns = args.patterns,
        n_trials = args.trials,
        dry_run  = args.dry_run,
        resume   = not args.no_resume,
    )

    results = runner.run()

    log.info(f"\nExperiment complete. {len(results)} trials run.")
    log.info(f"Results saved to: {RESULTS_DIR}")
