# CDP Experiment — Linear Regression Case Study

> **Context-Driven Prompting (CDP) vs Baseline Prompt Patterns**
> An empirical evaluation of structured prompt schemas for ML pipeline generation.

---

## What This Is

This experiment compares 6 prompt engineering patterns against 7 binary
statistical-validity metrics when generating a linear regression pipeline.
The hypothesis: CDP's structured schema produces significantly more statistically
correct code than zero-shot, few-shot, CoT, role-based, and legacy expert prompts.

### The 6 Patterns Tested

| # | Pattern | Lines | Key Characteristic |
|---|---------|-------|--------------------|
| 1 | Zero-shot | 1 | Plain task description |
| 2 | Few-shot | ~10 | Input/output examples |
| 3 | Chain-of-thought | ~20 | Explicit step list |
| 4 | Role-based | ~5 | Persona injection |
| 5 | Legacy expert | ~15 | Kaggle-style instructions |
| 6 | **CDP** | **~250** | **Structured schema with backtrack rules** |

### The 7 Scoring Metrics (Binary Pass/Fail)

| Metric | Pass Condition | CDP Step |
|--------|---------------|----------|
| No dummy trap | `drop='first'` in OHE | Step 6 |
| VIF check | Multicollinearity checked | Step 7 |
| Heteroscedasticity | Breusch-Pagan test run | Step 11 |
| Outliers treated | Winsorized or clipped | Step 5 |
| Scaler no leakage | `fit` on train only | Step 8 |
| Residuals checked | Shapiro-Wilk or DW test | Step 11 |
| Model saves | `.pkl` files persisted | Step 12 |

---

## Project Structure

```
linear_regression_cdp_case_study/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── generate_dataset.py          # Synthetic 500-row dataset generator
│   ├── dataset_500rows.csv          # Generated dataset (7 columns, 6 injected issues)
│   └── dataset_schema.json          # Column descriptions + injected issues map
│
├── prompts/
│   └── patterns/
│       ├── 01_zero_shot.txt
│       ├── 02_few_shot.txt
│       ├── 03_chain_of_thought.txt
│       ├── 04_role_based.txt
│       ├── 05_legacy_expert.txt
│       └── 06_cdp_pipeline.cdp      ← CDP schema (the proposed format)
│
├── src/
│   ├── cdp_types.py                 # AST dataclasses + 7 metric definitions
│   ├── cdp_lexer.py                 # Tokenizer (18 token types)
│   ├── cdp_parser.py                # AST builder (recursive descent)
│   ├── cdp_validator.py             # Schema integrity checker
│   ├── cdp_checkpoint_extractor.py  # Maps schema → scoring checkpoints
│   ├── evaluator.py                 # Scores LLM output against checkpoints
│   └── agent_runner.py              # Orchestrates experiment trials
│
├── analysis/
│   ├── aggregate_results.py         # Combines trial JSONs → tables
│   ├── generate_charts.py           # Produces 4 figures
│   └── outputs/
│       ├── tables/                  # Markdown tables (auto-generated)
│       └── figures/                 # PNG charts (auto-generated)
│
├── models/
│   └── results/                     # One JSON file per trial
│
├── tests/
│   └── test_cdp_parser.py           # 43 unit tests
│
└── scripts/
    └── validate_environment.py      # Pre-flight checks
```

---

## Quick Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd linear_regression_cdp_case_study
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# or
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
python data/generate_dataset.py
```

Expected output:
```
[DataGen] Saved 500 rows → data/dataset_500rows.csv
[DataGen] Null counts: years_exp 37, bonus_amount 28
[DataGen] Schema → data/dataset_schema.json
```

### 5. Validate environment

```bash
PYTHONPATH=src python scripts/validate_environment.py
```

All checks should pass (green ✅). The only expected warning is Ollama
if you haven't installed it yet — dry-run still works without it.

### 6. Run tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Expected: **43 passed**.

---

## Running the Experiment

### Dry-run (no Ollama needed — uses mock outputs)

```bash
PYTHONPATH=src python src/agent_runner.py --dry-run
```

This confirms the full pipeline works end-to-end using pre-written
mock outputs for `zero_shot` and `cdp` patterns.

### Live run with Ollama

#### Install Ollama
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama
```

#### Start Ollama and pull a model
```bash
ollama serve                  # in one terminal
ollama pull llama3.2          # recommended (4GB RAM)
# or
ollama pull mistral           # alternative
```

#### Run full experiment
```bash
PYTHONPATH=src python src/agent_runner.py \
  --models llama3.2 \
  --patterns zero_shot few_shot chain_of_thought role_based legacy_expert cdp \
  --trials 3
```

#### Run with multiple models
```bash
PYTHONPATH=src python src/agent_runner.py \
  --models llama3.2 mistral \
  --trials 5
```

**Resume support**: if a run is interrupted, re-run the same command.
Completed trials are skipped automatically.

---

## Analyzing Results

### Generate tables (after experiment)

```bash
PYTHONPATH=src python analysis/aggregate_results.py
```

Outputs to `analysis/outputs/tables/`:
- `table1_pattern_comparison.md`
- `table2_model_comparison.md`
- `table3_failure_analysis.md`

### Generate charts

```bash
PYTHONPATH=src python analysis/generate_charts.py
```

Outputs to `analysis/outputs/figures/`:
- `figure1_score_by_pattern.png` — bar chart
- `figure2_time_vs_validity.png` — scatter
- `figure3_failure_heatmap.png` — metric × pattern heatmap
- `figure4_cdp_vs_rest.png` — CDP vs baseline per metric

---

## CDP Parser — Architecture

The CDP parser is a two-stage pipeline:

```
.cdp file
    ↓
CDPLexer          → List[Token]         (18 token types)
    ↓
CDPParser         → CDPSchema (AST)     (recursive descent)
    ↓
CDPValidator      → List[ValidationResult]
    ↓
CDPCheckpointExtractor → List[ScoringCheckpoint]
    ↓
CDPEvaluator      → TrialResult (7 binary scores)
```

**Key design decisions:**

- Lexer and parser are separated — lexer never makes semantic decisions
- Backtrack rules are first-class AST nodes with typed targets (HALT / STEP / RETRY)
- Validator runs before every experiment — no silently corrupt schemas
- Checkpoints are extracted once at startup, reused across all trials

---

## Expected Hypothesis Results

| Pattern | Expected Score | Std Dev | Expected Time |
|---------|---------------|---------|---------------|
| Zero-shot | 0–14% | high | ~30s |
| Few-shot | 14–28% | high | ~45s |
| Chain-of-thought | 28–42% | medium | ~60s |
| Role-based | 14–28% | high | ~40s |
| Legacy expert | 28–42% | medium | ~50s |
| **CDP** | **71–100%** | **low** | ~120s |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB |
| GPU VRAM | 4GB (llama3.2:3b) | 8GB (llama3.2:8b) |
| Disk | 5GB | 10GB |
| CPU | 4 cores | 8 cores |

> For 4GB GPU (AMD Ryzen 7 4800H): use `ollama pull llama3.2:3b`

---

## Citation

If you use this experiment in your research:

```
@misc{cdp_experiment_2026,
  author  = {Tushar},
  title   = {CDP: Context-Driven Prompting for ML Pipelines},
  year    = {2026},
  url     = {https://github.com/tusharvats2025/context-driven-prompt-experimentation}
}
```

---

## License

MIT
