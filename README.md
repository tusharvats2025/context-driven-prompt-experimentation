<div align="center">

<img src="docs/assets/cdp_banner.svg" alt="CDP Banner" width="100%"/>

# Context-Driven Prompting (CDP)

**A structured prompt schema language for ML pipeline generation**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2-black?style=flat)](https://ollama.com)
[![Tests](https://img.shields.io/badge/Tests-43%20passing-22C55E?style=flat&logo=pytest)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-F59E0B?style=flat)]()
[![Case Study](https://img.shields.io/badge/Case%20Study-PDF-1E40AF?style=flat)](docs/CDP_Case_Study_LaTeX.pdf)

<br/>

> *CDP scored **90.5%** vs **0–38.1%** for all baseline prompt patterns*
> *across 7 statistical validity metrics — empirically validated.*

<br/>

[Case Study PDF](docs/CDP_Case_Study_LaTeX.pdf) •
[Quick Start](#quick-start) •
[What is CDP?](#what-is-cdp) •
[Results](#results) •
[Roadmap](#roadmap)

</div>

---

## What is CDP?

Most prompt engineering approaches treat prompts as plain text instructions.
**CDP is different.** It defines pipeline logic in a typed `.cdp` schema file
with named sections, config-driven thresholds, and conditional backtrack rules —
making ML pipeline generation verifiable, auditable, and model-class aware.

```cdp
[pipeline]
  name         = linear_regression_pipeline
  model_class  = linear
  version      = 1.0
  iterative    = true

[config]
  VIF_THRESHOLD             = 5.0
  OUTLIER_WINSOR_PERCENTILE = [1, 99]
  TRAIN_SPLIT_RATIO         = 0.80
  RESIDUAL_NORMALITY_ALPHA  = 0.05

[steps]

  [Step 7 | multicollinearity_check]
    action
      while True:
          vif_scores = calc_VIF(df.features)
          worst      = max(vif_scores)
          if worst.vif > VIF_THRESHOLD:
              drop(worst.col)
              audit_log("dropped_VIF", worst.col + worst.vif)
          else:
              break
    backtrack
      → Step 6   if VIF loop removes > 50% of features
      WARN       if < 3 features remain after VIF loop
    output
      all remaining features have VIF < VIF_THRESHOLD
```

**No other prompt pattern has this.** Zero-shot, few-shot, CoT, role-based, and
legacy expert prompts are linear instruction sequences with no enforcement
mechanism, no conditional logic, and no backtracking when a step fails.

---

## Why CDP?

Three things that make CDP architecturally different from a long system prompt:

| Feature | Long Prompt | CDP |
|---------|-------------|-----|
| Threshold management | Hardcoded in prose | Named in `[config]`, change once |
| Conditional execution | Not possible | `backtrack → Step N if <condition>` |
| Deployment awareness | None | `[guarantees]` skips verified steps |
| Auditability | None | `audit_log()` at every decision point |
| Parseable / testable | No | Yes — lexer + parser + validator |

---

## Results

Controlled experiment: **6 prompt patterns × 3 trials each** using
Ollama llama3.2 (local, temperature=0.2) on a 500-row synthetic dataset
with 6 deliberately injected statistical issues.

### Score by Pattern

| Pattern | Trial 1 | Trial 2 | Trial 3 | **Avg** | Std Dev |
|---------|:-------:|:-------:|:-------:|:-------:|:-------:|
| Zero-shot | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% |
| Few-shot | 14.3% | 14.3% | 14.3% | **14.3%** | 0.0% |
| Chain-of-thought | 28.6% | 42.9% | 28.6% | **33.3%** | 8.3% |
| Role-based | 14.3% | 14.3% | 14.3% | **14.3%** | 0.0% |
| Legacy expert | 42.9% | 28.6% | 42.9% | **38.1%** | 8.3% |
| **CDP (ours)** | **85.7%** | **100%** | **85.7%** | **90.5%** | **8.3%** |

### Per-Metric Pass Rate

| Metric | Zero | Few | CoT | Role | Legacy | **CDP** |
|--------|:----:|:---:|:---:|:----:|:------:|:-------:|
| No dummy trap | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 100% |
| VIF check | ❌ | ❌ | ❌ | 33% | ❌ | ✅ 100% |
| Heteroscedasticity | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 100% |
| Outliers treated | ❌ | ❌ | ✅ | ❌ | 67% | ✅ 100% |
| Scaler no leakage | ❌ | ✅ | 33% | 67% | 67% | ⚠️ 33% |
| Residuals checked | ❌ | ❌ | ❌ | ❌ | 33% | ✅ 100% |
| Model saves | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ 100% |

> **Key finding:** No Dummy Trap, VIF Check, and Heteroscedasticity were
> triggered **zero times** across 15 non-CDP trials. CDP enforced all three
> in 3/3 trials by naming exact procedures in action blocks.

📄 **[Read the full case study →](docs/CDP_Case_Study_LaTeX.pdf)**

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/tushar/cdp-experiment
cd cdp-experiment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate dataset

```bash
python data/generate_dataset.py
```

### 3. Validate environment

```bash
PYTHONPATH=src python scripts/validate_environment.py
```

### 4. Run experiment (dry-run — no Ollama needed)

```bash
PYTHONPATH=src python src/agent_runner.py --dry-run
```

### 5. Run live experiment (requires Ollama)

```bash
# Install Ollama: https://ollama.com
ollama serve
ollama pull llama3.2

PYTHONPATH=src python src/agent_runner.py \
  --models llama3.2 \
  --patterns zero_shot few_shot chain_of_thought role_based legacy_expert cdp \
  --trials 3
```

### 6. Generate results

```bash
PYTHONPATH=src python analysis/aggregate_results.py
PYTHONPATH=src python analysis/generate_charts.py
```

---

## Project Structure

```
cdp-experiment/
│
├── prompts/patterns/
│   ├── 01_zero_shot.txt
│   ├── 02_few_shot.txt
│   ├── 03_chain_of_thought.txt
│   ├── 04_role_based.txt
│   ├── 05_legacy_expert.txt
│   └── 06_cdp_pipeline.cdp        ← the CDP schema
│
├── src/
│   ├── cdp_types.py               ← AST dataclasses + 7 metric definitions
│   ├── cdp_lexer.py               ← tokenizer (18 token types)
│   ├── cdp_parser.py              ← recursive descent parser → CDPSchema AST
│   ├── cdp_validator.py           ← 8 integrity checks
│   ├── cdp_checkpoint_extractor.py← CDP steps → scoring signals
│   ├── evaluator.py               ← binary pass/fail per metric per trial
│   └── agent_runner.py            ← orchestrator + Ollama client
│
├── data/
│   ├── generate_dataset.py        ← 500 rows, 6 injected issues
│   └── dataset_500rows.csv
│
├── analysis/
│   ├── aggregate_results.py       ← 3 markdown tables
│   └── generate_charts.py         ← 4 publication-ready figures
│
├── tests/
│   └── test_cdp_parser.py         ← 43 unit tests (all passing)
│
├── docs/
│   ├── CDP_Case_Study_LaTeX.pdf   ← academic paper
│   └── CDP_FORMAT_SPEC.md         ← formal .cdp specification
│
└── scripts/
    └── validate_environment.py    ← pre-flight checks
```

---

## CDP Parser Architecture

The `.cdp` format has a purpose-built two-stage parser:

```
.cdp file
    ↓
CDPLexer           18 typed token types
    ↓
CDPParser          recursive descent → CDPSchema AST
    ↓
CDPValidator       8 integrity checks
    ↓
CheckpointExtractor  maps steps → scoring signals
    ↓
CDPEvaluator       binary pass/fail per trial
```

```bash
# Run the 43 unit tests
PYTHONPATH=src python -m pytest tests/ -v
```

---

## CDP Format Specification

A `.cdp` file has five top-level sections:

```
[pipeline]     identity — name, model_class, version, iterative
[config]       all thresholds declared once, referenced by name
[providers]    external I/O interface declarations
[guarantees]   developer assertions — true value skips checks
[steps]        ordered execution units
```

Each step has four blocks:

```
input      preconditions — what must exist before this step runs
action     executable pseudocode — named procedures to implement
backtrack  conditional jumps — → Step N if <condition> | HALT | WARN
output     postconditions — what must be true after completion
```

📄 **[Full CDP Format Specification →](docs/CDP_FORMAT_SPEC.md)**

---

## Roadmap

### ✅ v1.0 — Linear Regression Case Study (Complete)
- [x] `linear_regression.cdp` — 12-step pipeline schema
- [x] CDPLexer + CDPParser + CDPValidator
- [x] CDPCheckpointExtractor + CDPEvaluator
- [x] Controlled experiment — 6 patterns × 3 trials × 7 metrics
- [x] Academic case study (LaTeX + PDF)
- [x] 43 unit tests

### 🔄 v1.1 — Grammar Refinements (In Progress)
- [ ] `retry_exempt` annotation for backtrack validator
- [ ] Executable constraints for Step 8 (scaler leakage fix)
- [ ] Audit checkpoints in Steps 6 and 8
- [ ] Logistic regression CDP schema
- [ ] Extend experiment to 10 trials per pattern

### 📋 v2.0 — CDP Directory Architecture (Planned)
- [ ] Multi-file pipeline (`pipeline.cdp` as orchestrator)
- [ ] Cross-file input/output contracts
- [ ] Global config inheritance
- [ ] Model-class-aware selective execution
- [ ] MCP server integration for agent runtimes
- [ ] ANTLR grammar (`cdp.g4`) → TypeScript + Java parsers
- [ ] VS Code syntax extension

---

## CDP v1.1 — Known Improvement

The experiment identified one boundary condition:

> **CDP v1.0 enforces structural compliance** (pipeline shape, statistical
> test selection) but **does not guarantee implementation fidelity** for
> steps described declaratively rather than with executable constraints.

The scaler leakage metric failed in 2/3 CDP trials because Step 8 used
declarative pseudocode. The v1.1 fix replaces it with an implementation
constraint + post-condition assertion:

```cdp
[Step 8 | feature_scaling]
  action
    // IMPLEMENTATION REQUIRED — not pseudocode:
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)    // never fit on test
    joblib.dump(scaler, model_output_path + "/scaler.pkl")
    // POST-CONDITION: scaler.n_samples_seen_ == len(X_train)
```

---

## Citation

If you use this work:

```bibtex
@misc{cdp2026,
  author  = {Tushar},
  title   = {Context-Driven Prompting (CDP): A Structured Prompt Schema
             Language for ML Pipeline Generation},
  year    = {2026},
  url     = {https://github.com/tusharvats2025/context-driven-prompt-experimentation},
  note    = {Independent research. Case study: linear regression, 16 trials,
             6 prompt patterns, 7 statistical validity metrics.}
}
```

---

## License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

**Built independently · No institution · No GPU cluster**
**One laptop · One idea · Real results**

*Tushar · India · 2026*

</div>
