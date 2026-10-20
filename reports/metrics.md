| model | test PR-AUC | ROC-AUC | precision @1% | recall @1% | $ recall @1% | valid PR-AUC | recall @1% FPR | Brier (calibrated) | new-account PR-AUC | test PR-AUC 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| Random (base rate) | 0.035 | 0.500 | 0.035 | 0.010 | 0.010 |  |  |  |  |  |
| LightGBM | 0.518 | 0.900 | 0.888 | 0.255 | 0.148 | 0.644 | 0.425 | 0.023 | 0.557 |  |
| LightGBM (scale_pos_weight) | 0.514 | 0.888 | 0.887 | 0.255 | 0.149 | 0.648 | 0.427 | 0.023 | 0.545 |  |
| LightGBM (Optuna-tuned) | 0.527 | 0.901 | 0.885 | 0.254 | 0.144 | 0.657 | 0.440 | 0.022 | 0.557 | 0.481-0.567 |
| Ablation: raw text strings | 0.522 | 0.899 | 0.886 | 0.255 | 0.145 | 0.647 | 0.435 | 0.023 | 0.559 |  |
| Ablation: no velocity/history/freq | 0.522 | 0.897 | 0.892 | 0.256 | 0.149 | 0.643 | 0.431 | 0.023 | 0.563 |  |
| Ablation: interpretable features only | 0.225 | 0.835 | 0.419 | 0.120 | 0.113 | 0.350 | 0.161 | 0.030 | 0.240 |  |
| Rules engine | 0.035 | 0.474 | 0.014 | 0.004 | 0.005 | 0.059 | 0.004 | 0.034 | 0.051 | 0.031-0.040 |
| Logistic regression | 0.098 | 0.773 | 0.049 | 0.014 | 0.014 | 0.160 | 0.014 | 0.033 | 0.151 | 0.081-0.131 |
| LightGBM (cost-weighted) | 0.508 | 0.887 | 0.881 | 0.253 | 0.147 | 0.638 | 0.426 | 0.023 | 0.542 |  |
