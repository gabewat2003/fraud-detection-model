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
_Not generated yet: run `make all` on the Kaggle data, then `make writeup`._
<!-- END:splits -->

## Performance (test window)

<!-- BEGIN:metrics -->
_Not generated yet: run `make all` on the Kaggle data, then `make writeup`._
<!-- END:metrics -->

## Explainability

TreeSHAP attributions; reason codes restricted to interpretable features, with masked vendor features reported
as one aggregate line.

<!-- BEGIN:explain -->
_Not generated yet: run `make all` on the Kaggle data, then `make writeup`._
<!-- END:explain -->

## Stability

<!-- BEGIN:drift -->
_Not generated yet: run `make all` on the Kaggle data, then `make writeup`._
<!-- END:drift -->

## Limitations and risks

- Masked features limit the ability to explain individual decisions.
- Card / account identity is approximated from masked fields.
- Costs are assumptions; the recommended thresholds change with them.
- Labels are delayed and incomplete (undetected fraud is labeled legitimate).
- Single merchant ecosystem; no claim of transfer to other fraud types.

## Fairness and customer impact

The data contains no protected attributes, so group fairness cannot be measured directly. Proxies matter anyway:
false declines are expected to concentrate on new accounts and transactions without a device record (check in
`notebooks/02_error_analysis.ipynb`), which can correlate with newer, younger or lower-income customers. Any real
deployment should monitor decline and complaint rates by customer tenure and channel.

## Maintenance

Monitor PSI of top features and the score monthly (`make drift`), the daily review queue against capacity, and
calibration. Re-fit thresholds when costs or capacity change; retrain when backtest performance degrades.
