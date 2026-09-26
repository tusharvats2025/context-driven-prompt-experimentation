# Table 1: Prompt Pattern Comparison

| Pattern | N Trials | Avg Score | Std Dev | Avg Time (s) |No Dummy Trap | VIF Check | Heteroscedasticity | Outliers Treated | Scaler No Leakage | Residuals Checked | Model Saves |
|---------|----------|-----------|---------|--------------|---|---|---|---|---|---|---|
| zero_shot            |    3     |    0.0   % |   0.0  % |     28.4     | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| few_shot             |    3     |   14.3   % |   0.0  % |     20.8     | 0% | 0% | 0% | 0% | 100% | 0% | 0% |
| chain_of_thought     |    3     |   33.3   % |   8.3  % |     41.0     | 0% | 0% | 0% | 100% | 33% | 0% | 100% |
| role_based           |    3     |   14.3   % |   0.0  % |     32.2     | 0% | 33% | 0% | 0% | 67% | 0% | 0% |
| legacy_expert        |    3     |   38.1   % |   8.3  % |     44.8     | 0% | 0% | 0% | 67% | 67% | 33% | 100% |
| cdp                  |    3     |   90.5   % |   8.3  % |     72.5     | 100% | 100% | 100% | 100% | 33% | 100% | 100% |

_Pass rate per metric shown as % across all trials._
