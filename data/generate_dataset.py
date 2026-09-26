"""
generate_dataset.py
===================
Generates a synthetic 500-row dataset for the CDP experiment.

Intentionally injected issues (all 7 metrics must handle these):
  1. Categorical column with > 1 category  → triggers OHE / dummy trap test
  2. Outliers in numeric cols              → triggers outlier treatment
  3. Multicollinear feature pair           → triggers VIF check
  4. Heteroscedastic residual structure    → triggers assumption check
  5. Missing values in two columns         → triggers imputation
  6. Skewed feature                        → triggers log1p transform

Output:
  dataset_500rows.csv
  dataset_schema.json   ← column descriptions + injected issues map
"""

import numpy as np
import pandas as pd
import json
import os

SEED = 42
N    = 500

rng  = np.random.default_rng(SEED)


def generate() -> pd.DataFrame:
    # ── Base features ────────────────────────────────────────────────────────

    age            = rng.integers(22, 65, size=N).astype(float)
    years_exp      = np.clip(age - 22 + rng.normal(0, 1.5, N), 0, 40)
    # INJECTED: years_exp highly correlated with age → VIF trigger
    experience_dup = years_exp + rng.normal(0, 0.5, N)   # near-duplicate

    education      = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        p=[0.15, 0.45, 0.30, 0.10],
        size=N,
    )

    city           = rng.choice(
        ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"],
        p=[0.25, 0.25, 0.20, 0.15, 0.15],
        size=N,
    )

    # INJECTED: right-skewed feature → log1p trigger
    bonus_amount   = rng.exponential(scale=5000, size=N)

    # INJECTED: heteroscedasticity — variance of noise grows with salary
    base_salary    = (
          3000
        + 500  * years_exp
        + 800  * (education == "Bachelor").astype(int)
        + 1500 * (education == "Master").astype(int)
        + 3000 * (education == "PhD").astype(int)
        + 200  * age
        + rng.normal(0, 1, N) * (800 + 80 * years_exp)  # heteroscedastic noise
    )

    # ── Target ───────────────────────────────────────────────────────────────

    salary         = np.round(base_salary + rng.normal(0, 500, N), 2)

    # ── Inject missing values ────────────────────────────────────────────────

    df = pd.DataFrame({
        "age"           : age,
        "years_exp"     : years_exp,
        "experience_dup": experience_dup,   # multicollinearity pair
        "education"     : education,
        "city"          : city,
        "bonus_amount"  : bonus_amount,
        "salary"        : salary,
    })

    # ~8% missing in years_exp (MAR — correlated with city)
    miss_mask_exp = rng.random(N) < 0.08
    df.loc[miss_mask_exp, "years_exp"] = np.nan

    # ~5% missing in bonus_amount (MCAR)
    miss_mask_bonus = rng.random(N) < 0.05
    df.loc[miss_mask_bonus, "bonus_amount"] = np.nan

    # ── Inject outliers ──────────────────────────────────────────────────────

    # 5 extreme salary outliers
    outlier_idx = rng.choice(N, size=5, replace=False)
    df.loc[outlier_idx, "salary"] *= rng.uniform(3.5, 6.0, size=5)

    # 3 extreme age outliers
    outlier_idx_age = rng.choice(N, size=3, replace=False)
    df.loc[outlier_idx_age, "age"] = rng.choice([85, 90, 95], size=3)

    return df


def generate_schema(df: pd.DataFrame) -> dict:
    return {
        "version"         : "1.0",
        "rows"            : len(df),
        "target_column"   : "salary",
        "columns"         : {
            "age": {
                "dtype"      : "float",
                "description": "Employee age in years",
                "issues"     : ["outliers injected (3 extreme values: 85-95)"],
            },
            "years_exp": {
                "dtype"      : "float",
                "description": "Years of work experience",
                "issues"     : ["~8% missing (MAR)", "highly correlated with experience_dup (VIF trigger)"],
            },
            "experience_dup": {
                "dtype"      : "float",
                "description": "Near-duplicate of years_exp (injected multicollinearity)",
                "issues"     : ["INJECTED: correlation > 0.97 with years_exp — triggers VIF check"],
            },
            "education": {
                "dtype"      : "object",
                "description": "Highest education level",
                "cardinality": 4,
                "issues"     : ["requires OHE with drop='first' to avoid dummy trap"],
            },
            "city": {
                "dtype"      : "object",
                "description": "City of employment",
                "cardinality": 5,
                "issues"     : ["requires OHE with drop='first' to avoid dummy trap"],
            },
            "bonus_amount": {
                "dtype"      : "float",
                "description": "Annual bonus (exponential distribution — skewed)",
                "issues"     : ["~5% missing (MCAR)", "right-skewed — triggers log1p", "outliers from exponential tail"],
            },
            "salary": {
                "dtype"      : "float",
                "description": "TARGET: Annual salary in INR",
                "issues"     : ["heteroscedastic noise (variance grows with years_exp)", "5 extreme outliers injected"],
            },
        },
        "injected_issues_map": {
            "dummy_trap"            : ["education", "city"],
            "multicollinearity"     : ["years_exp", "experience_dup"],
            "heteroscedasticity"    : ["salary (noise ~ years_exp)"],
            "outliers"              : ["salary (5 extreme)", "age (3 extreme)"],
            "missing_values"        : ["years_exp (~8% MAR)", "bonus_amount (~5% MCAR)"],
            "skewed_feature"        : ["bonus_amount (exponential)"],
        },
        "expected_metric_triggers": {
            "no_dummy_trap"              : "education + city require OHE with drop='first'",
            "vif_check"                  : "experience_dup has VIF >> 5 with years_exp",
            "heteroscedasticity_handled" : "Breusch-Pagan should reject H0 (p < 0.05)",
            "outliers_treated"           : "salary and age both have extreme outliers",
            "scaler_no_leakage"          : "StandardScaler must be fit on train split only",
            "residuals_checked"          : "Shapiro-Wilk or Durbin-Watson on residuals",
            "model_saves"                : "model.pkl + scaler.pkl must be written",
        }
    }


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    df = generate()

    csv_path    = os.path.join(out_dir, "dataset_500rows.csv")
    schema_path = os.path.join(out_dir, "dataset_schema.json")

    df.to_csv(csv_path, index=False)
    print(f"[DataGen] Saved {len(df)} rows → {csv_path}")
    print(f"[DataGen] Null counts:\n{df.isnull().sum()}")
    print(f"[DataGen] Dtypes:\n{df.dtypes}")

    schema = generate_schema(df)
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"[DataGen] Schema → {schema_path}")
