# Model card: IEEE-CIS transaction fraud model + decision policy

## Intended use

Ranking card-not-present e-commerce transactions by fraud risk and choosing approve / review / decline under an
explicit cost model. Portfolio / research project: **not** validated for production decisions about real people.

## Model

- **Champion:** LightGBM binary classifier, the full-feature variant with the best validation PR-AUC among
  unweighted, `scale_pos_weight`, cost-weighted and Optuna-tuned variants. Exact choice and parameters are in
  `models/metadata.json`.
- **Calibration:** isotonic regression fit on a dedicated calibration window.
- **Decision layer:** expected-cost comparison per transaction, with a manual-review tier sized to capacity
  (`src/fraud/policy.py`, costs in `configs/costs.yaml`).
- **Baselines:** hand-written rules engine; logistic regression on 17 interpretable features.

## Data

IEEE-CIS Fraud Detection (Vesta, Kaggle): 590,540 transactions over ~6 months, ~3.5% labeled fraud (chargeback
based), left-joined to device/identity records available for ~24% of transactions. Most columns are masked by the
vendor. Time split: train 60% / valid 15% / calib 10% / test 15% of the time axis.

<!-- BEGIN:splits -->
| split | rows | fraud_rate | fraud_usd | first_day | last_day |
|---|---|---|---|---|---|
| train | 354324 | 0.0338 | 1,677,791 | 1 | 101 |
| valid | 88581 | 0.0404 | 615,011 | 101 | 130 |
| calib | 59054 | 0.0342 | 321,435 | 130 | 152 |
| test | 88581 | 0.0348 | 469,609 | 152 | 182 |
<!-- END:splits -->

## Performance (test window)

<!-- BEGIN:metrics -->
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
<!-- END:metrics -->

## Explainability

TreeSHAP attributions; reason codes restricted to interpretable features, with masked vendor features reported
as one aggregate line.

<!-- BEGIN:explain -->
Share of total |SHAP| by feature group: masked 60%, known_raw 15%, text_norm 6%, history 6%, frequency 5%, velocity 4%, simple 3%. Interpretable features carry 40% of attribution.

| feature | group | mean_abs_shap |
|---|---|---|
| C13 | masked | 0.1774 |
| card6 | known_raw | 0.1476 |
| TransactionAmt | known_raw | 0.1342 |
| V70 | masked | 0.1215 |
| freq_card1 | frequency | 0.1070 |
| C1 | masked | 0.1055 |
| D2 | masked | 0.1053 |
| p_email_provider | text_norm | 0.0914 |
| card_prior_n | history | 0.0880 |
| C14 | masked | 0.0880 |
| card1 | known_raw | 0.0815 |
| amt_cents | simple | 0.0814 |
| C11 | masked | 0.0727 |
| V91 | masked | 0.0699 |
| D15 | masked | 0.0670 |
<!-- END:explain -->

## Stability

<!-- BEGIN:drift -->
Last scored window vs training: features with PSI > 0.2: card_prior_n.

| feature | psi |
|---|---|
| card_prior_n | 0.359 |
| D15 | 0.110 |
| V30 | 0.063 |
| V91 | 0.059 |
| V70 | 0.056 |
| D2 | 0.044 |
| card1_n_addr_7d | 0.040 |
| C13 | 0.030 |
| C1 | 0.015 |
| model_score | 0.015 |
<!-- END:drift -->

## Limitations and risks

- Masked features limit the ability to explain individual decisions.
- Card / account identity is approximated from masked fields.
- Costs are assumptions; the recommended thresholds change with them.
- Labels are delayed and incomplete (undetected fraud is labeled legitimate).
- Single merchant ecosystem; no claim of transfer to other fraud types.

## Fairness and customer impact

The data contains no protected attributes, so group fairness cannot be measured directly. Proxies matter anyway:
false alarms concentrate in one product line (`ProductCD = C`: 47% of the top-scored legitimate transactions vs
9.5% of legitimate traffic) and in transactions with an identity/device record (61% vs 20%); see
`notebooks/02_error_analysis.ipynb`. If a product line or channel correlates with customer demographics, friction
does too. Any real deployment should monitor decline and complaint rates by product, channel and customer tenure.

## Maintenance

Monitor PSI of top features and the score monthly (`make drift`), the daily review queue against capacity, and
calibration. Re-fit thresholds when costs or capacity change; retrain when backtest performance degrades.
