# Table 3: Failure Mode Analysis

Failure rate per metric per pattern (% of trials where metric FAILED).

| Metric | zero_shot | few_shot | chain_of_thought | role_based | legacy_expert | cdp |
|--------|------|------|------|------|------|------|
| No Dummy Trap                  | 100% | 100% | 100% | 100% | 100% | 0% |
| VIF Check                      | 100% | 100% | 100% | 67% | 100% | 0% |
| Heteroscedasticity             | 100% | 100% | 100% | 100% | 100% | 0% |
| Outliers Treated               | 100% | 100% | 0% | 100% | 33% | 0% |
| Scaler No Leakage              | 100% | 0% | 67% | 33% | 33% | 67% |
| Residuals Checked              | 100% | 100% | 100% | 100% | 67% | 0% |
| Model Saves                    | 100% | 100% | 0% | 100% | 0% | 0% |

**Key insight**: High failure rate on `vif_check` and `no_dummy_trap`
across zero-shot/few-shot confirms CDP's structured enforcement is necessary.
