Costs: chargeback fee $15, friction $5 + 25% of the amount per false decline, review $3, review capacity 1.0% of daily volume.
| Policy | Fraud $ stopped | Fraud $ missed | Good customers declined | False-decline rate | Reviews | Total cost | Saving vs approve-all | Saving 95% CI | Saving vs top-1% |
|---|---|---|---|---|---|---|---|---|---|
| Approve everything | $0 | $469,609 | 0 | 0.00% | 0 | $515,854 | 0.0% |  | -20.3% |
| Rules engine (2+ rules decline, 1 rule review) | $3,441 | $466,167 | 0 | 0.00% | 2,320 | $518,517 | -0.5% | -1.6% to 0.3% | -20.9% |
| Model: decline top 1% | $78,901 | $390,707 | 163 | 0.19% | 0 | $428,738 | 16.9% | 13.5% to 20.7% | 0.0% |
| Model: best flat threshold | $218,043 | $251,565 | 2,099 | 2.46% | 0 | $381,818 | 26.0% | 20.3% to 31.2% | 10.9% |
| Model: amount-aware decline | $183,162 | $286,447 | 1,369 | 1.60% | 0 | $378,057 | 26.7% | 22.0% to 31.4% | 11.8% |
| Model: three-tier with review cap | $263,015 | $206,593 | 1,206 | 1.41% | 907 | $266,532 | 48.3% | 44.1% to 52.3% | 37.8% |

**Three-tier policy driven by each model**
| Model | Total cost | Saving vs approve-all | Fraud $ recall | False-decline rate | Review rate |
|---|---|---|---|---|---|
| LightGBM | $261,046 | 49.4% | 57.3% | 1.53% | 1.08% |
| Ablation: raw text strings | $263,926 | 48.8% | 56.1% | 1.32% | 1.08% |
| Ablation: no velocity/history/freq | $265,878 | 48.5% | 59.8% | 2.10% | 1.11% |
| LightGBM (Optuna-tuned) | $266,532 | 48.3% | 56.0% | 1.41% | 1.02% |
| LightGBM (scale_pos_weight) | $276,422 | 46.4% | 52.7% | 1.17% | 1.05% |
| LightGBM (cost-weighted) | $296,102 | 42.6% | 47.3% | 1.27% | 1.05% |
| Ablation: interpretable features only | $356,257 | 30.9% | 41.4% | 1.95% | 1.15% |
| Logistic regression | $460,535 | 10.7% | 24.3% | 2.34% | 1.15% |
| Rules engine | $468,578 | 9.2% | 10.6% | 0.00% | 1.03% |

**Break-even decline probability by amount**
| amount | decline if p > |
|---|---|
| $5 | 0.238 |
| $25 | 0.220 |
| $100 | 0.207 |
| $500 | 0.202 |
| $2,000 | 0.200 |

**Review queue (three-tier, test window):** mean 1.03% of daily volume, max 1.59%; 17 of 31 days above capacity (128 overflow cases).

Thresholds fit on calib: {"top_1pct_threshold": 0.62083, "cost_optimal_flat_threshold": 0.10352, "review_benefit_threshold": 56.86022, "rules_review_amount_bar": 0.0}
