# Transaction Fraud Detection with a Cost-Based Decision Policy

> Detects fraudulent e-commerce transactions in a heavily imbalanced dataset (IEEE-CIS, ~590K transactions,
> ~3.5% fraud), evaluates with precision-recall and dollars instead of accuracy, and turns model scores into an
> **approve / review / decline** policy that minimizes fraud losses plus customer friction under an explicit,
> adjustable cost model.

**Stack:** Python · SQL (DuckDB) · pandas · LightGBM · scikit-learn · Optuna · MLflow · Altair · FastAPI · Streamlit · Docker · pytest · GitHub Actions

<!-- DRAFT: rewrite the paragraph below in your own voice -->
I built this as a portfolio project to practice the part of fraud work that most course projects skip: what
to actually *do* with a fraud score. A model that ranks transactions well is only half the job. The other half
is deciding where the cutoffs go, given what a missed fraud and a wrongly declined customer each cost, and how
many cases a review team can handle in a day.

## Results at a glance

<!-- BEGIN:headline -->
- **48.3% lower fraud + friction cost** than approving everything on the held-out test window (95% CI 44.1% to 52.3%, day-block bootstrap), and 48.6% lower than a hand-written rules engine.
- Stops **56.0% of fraud dollars** while declining **1.41% of legitimate customers** and sending 1.02% of traffic to manual review (cap 1.0%).
- 37.8% cheaper than the common 'decline the top 1% of scores' rule: declines follow expected cost, and the most uncertain high-value cases go to manual review.
- Model: LightGBM (Optuna-tuned), test PR-AUC **0.527** (95% CI 0.481-0.567) against a 0.035 base rate.
<!-- END:headline -->

![Total cost by policy](reports/figures/policy_costs.png)

Every number in this README is generated from pipeline outputs in [`reports/`](reports/) by `make writeup`;
the blocks between `<!-- BEGIN -->` / `<!-- END -->` markers are overwritten on each run.

---

## Contents

1. [The business question](#1-the-business-question)
2. [Data](#2-data)
3. [How the pipeline works](#3-how-the-pipeline-works)
4. [What the data says (SQL analysis)](#4-what-the-data-says-sql-analysis)
5. [Features, and keeping them leak-free](#5-features-and-keeping-them-leak-free)
6. [Normalizing device, browser and email strings](#6-normalizing-device-browser-and-email-strings)
7. [Models](#7-models)
8. [From scores to decisions](#8-from-scores-to-decisions)
9. [Error analysis and explainability](#9-error-analysis-and-explainability)
10. [Does it hold up over time?](#10-does-it-hold-up-over-time)
11. [Serving: API and dashboard](#11-serving-api-and-dashboard)
12. [Limitations](#12-limitations)
13. [How this was built (AI-assisted development)](#13-how-this-was-built-ai-assisted-development)
14. [Reproduce](#14-reproduce)

---

## 1. The business question

> Given a fixed review capacity and a cost for every false alarm, which transactions should we approve, send to
> manual review, or decline, and how much fraud loss does that prevent versus the friction it creates?

A fraud team balances two failures:

| Error | What it costs |
|---|---|
| **Missed fraud** (approve a fraudulent transaction) | the chargeback: roughly the amount, plus a fee |
| **False alarm** (decline or challenge a good customer) | lost margin, a support contact, a customer less likely to come back |
| **Manual review** | analyst time, whether or not the case turns out to be fraud |

<!-- DRAFT: rewrite in your own words -->
**Why accuracy is useless here.** About 96.5% of transactions are legitimate, so a "model" that approves
everything is 96.5% accurate and catches nothing. Accuracy also treats every error as equally bad, and they
are not: missing a $2,000 fraud costs about 400 times more than missing a $5 one, while wrongly declining a
customer costs roughly the same whatever they were buying. So the right question is never "how often is the
model right?" but "how much money do its mistakes cost?", and the answer changes from one transaction to the
next.

## 2. Data

| Item | Detail |
|---|---|
| Source | [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Vesta Corporation, Kaggle) |
| Files | `train_transaction.csv` (590,540 rows) LEFT JOIN `train_identity.csv` (144,233 rows) on `TransactionID` |
| Target | `isFraud`, about 3.5% positive. Labels come from chargebacks |
| Time | `TransactionDT`: seconds from an **unknown** reference date, about 6 months. Usable for ordering and relative hour, not for calendar dates |
| Access | Competition rules forbid redistribution: [`data/README.md`](data/README.md) explains the download. Nothing row-level is committed |

Most of the ~430 columns are masked by the vendor (`V1-V339`, `C1-C14`, `D1-D15`, `M1-M9`, `id_01-id_38`):
their meaning is not published. The project uses them (they carry real signal) but leans on features whose
meaning is known when it explains decisions, and says so plainly when it can't.

**Why a LEFT join.** Only about a quarter of transactions have an identity row. An inner join would discard the
rest and change the population the model learns from:

<!-- BEGIN:sql_identity -->
| population | transactions | share_of_rows | fraud_rate | fraud_usd | share_of_fraud_usd | median_amount |
|---|---|---|---|---|---|---|
| kept by INNER join (has identity) | 144233 | 0.2442 | 0.0785 | 1005155.0 | 0.3259 | 50.0 |
| dropped by INNER join (no identity) | 446307 | 0.7558 | 0.0209 | 2078689.0 | 0.6741 | 77.95 |
<!-- END:sql_identity -->

**The time split.** Fraud patterns shift as attackers adapt, and a random split would put the same card in
train and test. Splits follow time, with a separate window for every decision that is tuned:

<!-- BEGIN:splits -->
| split | rows | fraud_rate | fraud_usd | first_day | last_day |
|---|---|---|---|---|---|
| train | 354324 | 0.0338 | 1,677,791 | 1 | 101 |
| valid | 88581 | 0.0404 | 615,011 | 101 | 130 |
| calib | 59054 | 0.0342 | 321,435 | 130 | 152 |
| test | 88581 | 0.0348 | 469,609 | 152 | 182 |
<!-- END:splits -->

| Split | Used for |
|---|---|
| train (first 60% of time) | fitting |
| valid (60-75%) | early stopping, Optuna, rule thresholds |
| calib (75-85%) | isotonic calibration, every policy threshold |
| test (last 15%) | final numbers only, touched once |

## 3. How the pipeline works

```mermaid
flowchart LR
    A[CSV files] -->|DuckDB| B[base<br/>LEFT JOIN identity]
    B --> C[text normalization<br/>lookup tables]
    C --> D[keys + velocity +<br/>spend history SQL]
    D --> E[features.parquet]
    E --> F[time split<br/>train / valid / calib / test]
    F --> G[rules engine<br/>logistic regression<br/>LightGBM variants]
    G --> H[isotonic calibration]
    H --> I[decision policy<br/>approve / review / decline]
    I --> J[reports, memo,<br/>dashboard, API]
```

| Stage | Code | Output |
|---|---|---|
| Ingest + join | [`sql/features/01_base_join.sql`](sql/features/01_base_join.sql), [`src/fraud/io.py`](src/fraud/io.py) | DuckDB `base` table |
| Text normalization | [`src/fraud/text_norm.py`](src/fraud/text_norm.py) | lookup tables joined in SQL |
| Features | [`sql/features/02-06`](sql/features/) | `features.parquet` |
| Split | [`src/fraud/split.py`](src/fraud/split.py) | split label per row |
| Models | [`src/fraud/train.py`](src/fraud/train.py), [`rules.py`](src/fraud/rules.py), [`tune.py`](src/fraud/tune.py) | models + calibrated scores |
| Evaluation | [`src/fraud/evaluate.py`](src/fraud/evaluate.py), [`report.py`](src/fraud/report.py) | `reports/metrics.*` |
| Decisions | [`src/fraud/policy.py`](src/fraud/policy.py) | `reports/policy.*` |
| Robustness | [`backtest.py`](src/fraud/backtest.py), [`drift.py`](src/fraud/drift.py) | backtest + PSI reports |
| Serving | [`src/fraud/serving.py`](src/fraud/serving.py), [`api/`](api/), [`app/`](app/) | FastAPI service, Streamlit app |

Each stage is a `make` target that calls one CLI command (`python -m fraud.cli <stage>`), and every
experiment run is logged to a local MLflow store (`make mlflow-ui`).

## 4. What the data says (SQL analysis)

The analyst queries live in [`sql/analysis/`](sql/analysis/), one business question per file, and run against
the DuckDB feature table with `make sql`. Results land in [`reports/sql/`](reports/sql/).

**Fraud rate by cohort** ([`01_fraud_rate_by_cohort.sql`](sql/analysis/01_fraud_rate_by_cohort.sql)):

<!-- BEGIN:sql_cohorts -->
| dimension | value | transactions | fraud_rate | lift | fraud_usd |
|---|---|---|---|---|---|
| DeviceType | mobile | 55645 | 0.1017 | 2.91 | 546427.0 |
| DeviceType | desktop | 85165 | 0.0652 | 1.86 | 449879.0 |
| DeviceType | (no identity) | 449730 | 0.021 | 0.6 | 2087538.0 |
| ProductCD | C | 68519 | 0.1169 | 3.34 | 391421.0 |
| ProductCD | S | 11628 | 0.059 | 1.69 | 43416.0 |
| ProductCD | H | 33024 | 0.0477 | 1.36 | 246632.0 |
| ProductCD | R | 37699 | 0.0378 | 1.08 | 348050.0 |
| ProductCD | W | 439670 | 0.0204 | 0.58 | 2054325.0 |
| card4 | discover | 6651 | 0.0773 | 2.21 | 182000.0 |
| card4 | visa | 384767 | 0.0348 | 0.99 | 1997706.0 |
| card4 | mastercard | 189217 | 0.0343 | 0.98 | 855739.0 |
| card4 | american express | 8328 | 0.0287 | 0.82 | 42798.0 |
| card4 | (missing) | 1577 | 0.026 | 0.74 | 5601.0 |
| card6 | credit | 148986 | 0.0668 | 1.91 | 1656751.0 |
| card6 | (missing) | 1571 | 0.0248 | 0.71 | 5051.0 |
| card6 | debit | 439938 | 0.0243 | 0.69 | 1422042.0 |
| has_identity | 1 | 144233 | 0.0785 | 2.24 | 1005155.0 |
| has_identity | 0 | 446307 | 0.0209 | 0.6 | 2078689.0 |
<!-- END:sql_cohorts -->

**Card testing** ([`05_card_testing.sql`](sql/analysis/05_card_testing.sql)): fraud rate by amount band and by
how often the card was used in the previous hour:

<!-- BEGIN:sql_card_testing -->
| amount_band | card_velocity_1h | transactions | fraud_rate |
|---|---|---|---|
| < $10 | 0 prior in 1h | 3429 | 0.0782 |
| < $10 | 3+ prior in 1h | 1310 | 0.0855 |
| < $10 | 1-2 prior in 1h | 2056 | 0.0905 |
| $10-50 | 0 prior in 1h | 131311 | 0.0289 |
| $10-50 | 1-2 prior in 1h | 37323 | 0.0592 |
| $10-50 | 3+ prior in 1h | 12086 | 0.0747 |
| $50-200 | 0 prior in 1h | 232586 | 0.0239 |
| $50-200 | 1-2 prior in 1h | 56645 | 0.0423 |
| $50-200 | 3+ prior in 1h | 13324 | 0.0499 |
| $200+ | 0 prior in 1h | 77475 | 0.041 |
| $200+ | 1-2 prior in 1h | 19465 | 0.0595 |
| $200+ | 3+ prior in 1h | 3530 | 0.0649 |
<!-- END:sql_card_testing -->

<!-- DRAFT: rewrite in your own voice (source: notebooks/01_eda.ipynb) -->
What stood out in the exploration ([`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)):

- **Missing data is a signal, not just a nuisance.** Transactions with an identity record are fraud 7.8% of the
  time vs 2.1% without, and for several masked columns (`D7`, `D12`-`D14`, `id_03`-`id_10`) the fraud rate is 4-5x
  higher when the value is *present*. Imputing those away would throw signal out.
- **One product line carries a disproportionate share.** `ProductCD = C` is 12% of transactions but has a 11.7%
  fraud rate (3.3x the average); mobile devices run at 10%.
- **Cents tell a story.** Retail-style prices ending in .95-.99 are fraud only 0.9% of the time, whole-dollar
  amounts 3.6%, and "odd" cents (.01-.95) 6.6-9.9%, plausibly currency-converted orders.
- **Card testing is visible but small.** Under $10 the fraud rate is 8-9% whatever the velocity; above $10, a card
  used 3+ times in the previous hour has a 1.6-2.6x higher fraud rate. But sub-$10 transactions are only ~1% of volume.

Other queries: daily KPI table with a trailing 7-day fraud rate ([`03_daily_kpis.sql`](sql/analysis/03_daily_kpis.sql)),
highest-risk product x email-provider x device segments ([`04_top_risk_segments.sql`](sql/analysis/04_top_risk_segments.sql)),
and decision-by-outcome under the recommended policy ([`06_policy_outcomes.sql`](sql/analysis/06_policy_outcomes.sql)).

## 5. Features, and keeping them leak-free

The data has no card or account ID, so two approximations stand in for one:

| Key | Built from | Notes |
|---|---|---|
| `card_key` | `card1 \| card2 \| addr1` | masked card attributes + billing region |
| `uid` | `card1 \| addr1 \| (day - D1)` | `D1` counts days since the card was first seen, so `day - D1` is roughly constant per card: an inferred card start date. This idea comes from the Kaggle IEEE-CIS community |

Both are proxies: collisions group near-identical cards. Features computed on them:

| Family | Examples | Why it matters |
|---|---|---|
| Velocity | earlier transactions on this card in the last 1h / 24h / 7d; seconds since the previous one; distinct regions / emails / devices per card number in 7 days | card testing and burst spending |
| Spend history | count, mean, std, max of the account's earlier amounts; amount / account mean; z-score; first-transaction flag | out-of-pattern spend |
| Frequency | how common the card number, region, email provider, device family were **in the training window** | rare values are riskier |
| Simple | `log(amount)`, cents part, email-domain match, has-identity flag, relative hour | odd amounts, third-party use |

The window logic is SQL, for example ([`03_velocity.sql`](sql/features/03_velocity.sql)):

```sql
COUNT(*) OVER (PARTITION BY card_key ORDER BY TransactionDT
               RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING) AS card_tx_1h
```

**Leakage is tested, not assumed.** The `1 PRECEDING` frame end means only transactions at least one second
earlier are visible. [`tests/test_no_leakage.py`](tests/test_no_leakage.py) checks that the first transaction of
every account has zero prior count and a missing prior mean, that same-second transactions can't see each other,
and that changing or adding *future* transactions never changes an earlier row's features.
[`tests/test_sql_pandas_parity.py`](tests/test_sql_pandas_parity.py) recomputes the window features with an
independent NumPy/pandas implementation and requires both to agree exactly. Frequency encodings are counted on
the training window only, and account-level keys are deliberately not frequency-encoded, because inside the
training window a row's own future transactions would be counted.

## 6. Normalizing device, browser and email strings

<!-- DRAFT: this is your linguistics angle; make the framing yours -->
The identity table stores user-agent-like strings, and the same thing is spelled many ways: one phone shows up as
`SM-G930V Build/NRD90M`, `SAMSUNG SM-G930V` and `SM-G930V`; one webmail provider as `hotmail.com`, `hotmail.es`,
`outlook.com` and `live.com.mx`. Left raw, every spelling is a separate, rarer category, which splits the signal
and breaks on spellings the model has never seen. That's the same problem as normalizing word forms in text: map
surface variants onto a canonical form (a "lemma") while keeping the informative parts as separate features.

[`src/fraud/text_norm.py`](src/fraud/text_norm.py) does this with ordered, testable rules:

| Raw field | Normalized features | Example |
|---|---|---|
| `DeviceInfo` | `device_brand`, `device_family` | `SAMSUNG SM-G930V Build/NRD90M` -> samsung / samsung g930 |
| `id_30` (OS) | `os_family`, `os_version` | `Mac OS X 10_13_2` -> macos / 10.13 |
| `id_31` (browser) | `browser_family`, `browser_major`, `browser_is_mobile`, versions behind the newest seen | `chrome 62.0 for android` -> chrome / 62 / mobile |
| `P_/R_emaildomain` | `email_provider`, `email_tld`, `is_free_mail`, `email_region`, provider match | `live.com.mx` -> microsoft / com.mx / free / mx |

Missing stays missing (absence of a device record is itself a signal). The rules are pure functions over one
string, so the batch pipeline (lookup tables over a few thousand distinct values) and the API (one transaction)
share them. [`tests/test_text_norm.py`](tests/test_text_norm.py) holds ~50 table-driven cases.

<!-- BEGIN:textnorm -->
| mapping | raw levels | normalized levels | rows mapped |
|---|---|---|---|
| DeviceInfo -> device_family | 1786 | 382 | 98.2% |
| DeviceInfo -> device_brand | 1786 | 19 | 98.2% |
| id_30 -> os_family | 75 | 6 | 100.0% |
| id_31 -> browser_family | 130 | 14 | 99.7% |
| emaildomain -> email_provider | 60 | 25 | 100.0% |
<!-- END:textnorm -->

**Did it help the model?** Ablations retrain the same LightGBM with one change each:

<!-- BEGIN:ablation -->
| variant | features | test PR-AUC | three-tier cost |
|---|---|---|---|
| LightGBM | 476 | 0.5182 | $261,046 |
| Ablation: raw text strings | 463 | 0.5223 | $263,926 |
| Ablation: no velocity/history/freq | 450 | 0.5222 | $265,878 |
| Ablation: interpretable features only | 63 | 0.2251 | $356,257 |
<!-- END:ablation -->

<!-- DRAFT: rewrite in your own voice -->
**Honest result: normalization did not improve the score.** The raw-string model scored slightly higher
(0.522 vs 0.518 test PR-AUC), a gap well inside the bootstrap noise. Gradient boosting can learn useful splits on
raw categories by itself, and the masked vendor columns already carry most of the device signal. What
normalization does buy is engineering value rather than accuracy: 1,786 device strings become 19 brands, so a
spelling never seen in training still maps to a known brand in production, and reason codes can say "device
brand: huawei" instead of "DeviceInfo: CAM-L03 Build/HUAWEICAM-L03". The same pattern holds for the velocity and
history features: removing them doesn't hurt (0.522), because the masked `C*`/`D*` columns appear to encode the
same behavior. Removing every masked column does hurt (0.225): the explainable features alone recover less than
half of the model's performance. Details in [`notebooks/03_text_normalization.ipynb`](notebooks/03_text_normalization.ipynb).

## 7. Models

| Model | What it is |
|---|---|
| Rules engine | five hand-written rules ("card used 5+ times in the last hour", "amount 5x the account average", ...), thresholds fit on valid. The "what an ops team runs today" baseline ([`rules.py`](src/fraud/rules.py)) |
| Logistic regression | 17 explainable features, median-imputed and standardized |
| LightGBM | all features; native missing-value and categorical handling; early stopping on valid PR-AUC |
| LightGBM, `scale_pos_weight` | up-weights fraud 5x; shows how class weighting distorts probabilities |
| LightGBM, cost-weighted | each row weighted by what a mistake on it costs (amount + fee for fraud, friction for legit) |
| LightGBM, Optuna-tuned | 30-trial TPE search on validation PR-AUC ([`tune.py`](src/fraud/tune.py)) |

No SMOTE or resampling: it distorts probabilities, and the decision layer multiplies probabilities by dollars.
All models are calibrated with isotonic regression on the calib split; the champion is the full-feature
LightGBM variant with the best **validation** PR-AUC.

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

PR-AUC is the headline metric because a random model scores about the base rate on it, while ROC-AUC looks
impressive on imbalanced data almost regardless. "@1%" columns answer the review-capacity question: if analysts
can look at the top 1% of scores, how much fraud, and how many fraud **dollars**, is in there? The new-account
column scores accounts never seen before the test window (cold start). Confidence intervals come from a
bootstrap that resamples whole days, since transactions on the same day are correlated.

![Precision-recall curves](reports/figures/pr_curves.png)

**Calibration.** The cost rule multiplies the probability by dollars, so a "30%" must mean about 30%:

![Calibration](reports/figures/calibration.png)

## 8. From scores to decisions

Cost assumptions ([`configs/costs.yaml`](configs/costs.yaml)) are explicit and varied below: chargeback fee $15;
a wrongly declined customer costs $5 of friction **plus 25% of the order amount** (the margin on the lost sale);
$3 per manual review; review capacity 1% of daily volume.

For a transaction with calibrated fraud probability *p* and amount *a*:

```
E[cost | approve] = p * (a + fee)
E[cost | decline] = (1 - p) * (friction + margin * a)
E[cost | review]  = review_cost            (reviewer assumed to resolve correctly)
```

Declining beats approving when `p > (friction + margin*a) / (friction + margin*a + a + fee)`: about 0.24 for a $5
order, falling to 0.20 for large ones.

<!-- DRAFT: this is a good story to tell in your own words -->
**The lost-margin term was a correction, not part of the first design.** The first version charged a flat $5 for
every false decline. Declining a $500 order then looked as cheap as declining a $5 one, so the "optimal" policy
declined **15.5% of legitimate customers**: cheap on paper, unacceptable in practice. A declined order also loses
the sale, so the decline cost has to grow with the amount. With that fixed, false declines dropped to about 1.4%,
and it turned out that most of the value comes from the **review tier**: sending the most uncertain high-value
cases to a person, within capacity.

Policies compared (all thresholds fit on calib, results on test):

| Policy | Rule |
|---|---|
| Approve everything | the do-nothing baseline |
| Rules engine | 2+ rules -> decline, 1 rule and large amount -> review (within capacity) |
| Model: decline top 1% | fixed share of traffic, the common default |
| Model: best flat threshold | the single score cutoff with the lowest cost on calib |
| Model: amount-aware decline | decline exactly when expected decline cost < expected approve cost |
| **Model: three-tier with review cap** | as above, plus review where it saves the most, with the review bar set so volume fits capacity |

<!-- BEGIN:policy -->
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
<!-- END:policy -->

![Cost curve](reports/figures/cost_curve.png)

**Sensitivity.** The costs are assumptions, so the policy is re-fit for 27 combinations of fixed
friction ($2/$5/$10), lost margin (10/25/50% of the amount) and review capacity (0.5/1/2%):

![Sensitivity](reports/figures/sensitivity_three_tier.png)

<!-- DRAFT: rewrite in your own voice -->
Across all 27 combinations the three-tier policy saves 36-63% against approving everything, so the conclusion
holds up. The assumption that matters most is not a cost but **review capacity**: going from 0.5% to 2% of daily
volume adds roughly 15-20 points of savings, more than any change in the friction or margin assumptions. For a
fraud team, that means the strongest lever is analyst headcount, and it's worth pricing what one more reviewer
saves.

**Operations view.** The review bar is fit on the calibration window and then applied unchanged, the way a
real-time system would run. That means daily queue volume varies:

![Review queue](reports/figures/review_queue.png)

A one-page summary for a non-technical fraud lead is in [`reports/memo.md`](reports/memo.md).

## 9. Error analysis and explainability

Attributions use TreeSHAP (LightGBM's native `pred_contrib`, the same algorithm as `shap.TreeExplainer`) on a
test sample:

![Attribution by feature group](reports/figures/shap_by_group.png)

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

Reason codes name only interpretable features; masked vendor columns are folded into one "masked vendor
signals" line, because a decline can't be explained to a customer or an auditor with "V258 was high". The API
returns the top three with every score.

<!-- DRAFT: rewrite in your own voice (source: notebooks/02_error_analysis.ipynb) -->
**Where the model goes wrong** ([`notebooks/02_error_analysis.ipynb`](notebooks/02_error_analysis.ipynb)):

- **False alarms concentrate in product C and identity-present traffic, not on new accounts** (which I expected).
  Among the top 1% highest-scored legitimate transactions, 47% are `ProductCD = C` (vs 9.5% of all legitimate
  traffic) and 61% have an identity record (vs 20%). Friction from this model falls on one product line.
- **Missed fraud is mostly ordinary-looking product W purchases.** 66% of the fraud the policy approved is product W,
  mostly $50-200. That's not card testing (only 2.6% of missed fraud is under $10); it's fraud that resembles normal
  spending.
- **Surprise: cold start isn't the weak spot.** 55% of test transactions come from accounts never seen before, and
  the model scores *better* on them (PR-AUC 0.557 vs 0.527 overall, at a higher 4.8% fraud rate). One likely reason:
  the account proxy is approximate, so some "new" accounts are probably existing cards whose `D1` shifted.

## 10. Does it hold up over time?

**Rolling backtest:** train on every earlier 30-day window, score the next, same parameters:

<!-- BEGIN:backtest -->
| test_window | train_rows | test_rows | fraud_rate | pr_auc | recall_at_1pct | dollar_recall_at_1pct |
|---|---|---|---|---|---|---|
| 2 | 223738 | 92189 | 0.0404 | 0.638 | 0.240 | 0.184 |
| 3 | 315927 | 98615 | 0.0395 | 0.599 | 0.239 | 0.143 |
| 4 | 414542 | 83571 | 0.0341 | 0.593 | 0.271 | 0.186 |
| 5 | 498113 | 86934 | 0.0342 | 0.597 | 0.267 | 0.173 |
<!-- END:backtest -->

![Backtest](reports/figures/backtest.png)

Performance is stable across windows (PR-AUC 0.59-0.64), and every fold scores **higher than the main test
number (0.53)**. The difference is recency: backtest folds train right up to the window they score, while the main
model stops at day 101 and is tested from day 152, because validation and calibration windows sit in between.
That gap is the price of a clean evaluation; a deployed model should be retrained on all labeled data first.

**Drift (PSI)** of the top features and of the score against the training window:

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

![Drift](reports/figures/drift_psi.png)

The model score itself is stable (PSI about 0.015 in every window). The one large shift, `card_prior_n` (count of
earlier transactions on the card), is **built into the feature**: the history starts on day 1 of the dataset, so
counts can only grow over time, and later windows look different from training by construction. A production
version would use a rolling count (e.g. last 90 days) instead of an all-time one.

## 11. Serving: API and dashboard

**Scoring API** ([`api/`](api/), [`src/fraud/serving.py`](src/fraud/serving.py)). `POST /score` takes raw
transaction fields and returns the calibrated probability, decision, expected cost of each action, break-even
probability and reason codes. Costs can be overridden per request.

```bash
make api          # http://localhost:8000/docs
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d @api/example_request.json
```

Online features are computed by **the same SQL files** as the batch pipeline: the incoming transaction is
written into empty raw-shaped tables, joined, normalized, and windowed against that card's stored history.
[`tests/test_api.py`](tests/test_api.py) replays test transactions through the API path and requires the
features to equal the batch features exactly (no train/serve skew). It's a demo service (~300 ms per call on a
laptop, read-only history snapshot). A production system would keep running aggregates in a feature store.
`make docker` packages it; CI builds the image and smoke-tests it.

**Policy simulator** ([`app/dashboard.py`](app/dashboard.py), `make dashboard`): sliders for fee, friction,
review cost and capacity; thresholds are re-fit live and every policy re-scored on the test window. Tabs for
model quality, drift, and per-transaction explanations.

<!-- TODO: add a screenshot or GIF of the dashboard: reports/figures/dashboard.png -->

## 12. Limitations

- **Masked features.** Most predictive columns (`V*`, `C*`, `D*`) are opaque. That limits explainability, which
  matters where decisions must be justified. The interpretable-only ablation shows what that costs.
- **No calendar.** `TransactionDT` has no anchor: no seasonality, no day-of-week claims, only ~6 months of history.
- **Card and account are proxies** built from masked fields; collisions blur velocity and history features.
- **Costs are assumptions**, not observed business numbers. Dollar results are illustrative; the sensitivity grid
  shows how much they move.
- **Labels lag and miss fraud.** Chargeback labels arrive late and never cover undetected fraud, so "legitimate"
  contains some unlabeled fraud, and the review queue is assumed to resolve cases perfectly.
- **Reviewers are assumed perfect.** Most of the three-tier saving comes from manual review, which is modeled as
  a $3 decision that is always right. Real reviewers make mistakes and add delay, so treat that part as an upper bound.
- **The rules baseline is weak by construction.** Five quick rules, fit on one window, decayed to near-random on
  test (the velocity rule's precision fell from 14% to 2.5%). A real ops team's rules would be stronger and
  maintained, so the "vs rules engine" comparison flatters the model.
- **Thresholds are fit once.** The review bar is set on the calibration window; real traffic would need it
  monitored and refreshed (the queue chart shows days over capacity).
- **One merchant ecosystem.** It may not transfer to other fraud types (account takeover, synthetic identity,
  first-party/BNPL fraud).
- **Demo serving.** The API recomputes features from a static history snapshot; it does not update state.

## 13. How this was built (AI-assisted development)

I used Claude Code to help design and write much of this codebase. I treated
its output as untrusted until checked, and the checks are part of the repo:_

- *Backward-looking features are proven backward-looking by tests, including a "change the future, the past must
  not move" test, and by an independent pandas reimplementation that must match the SQL exactly.*
- *No `TransactionID` appears in two splits; split boundaries are strictly time-ordered.*
- *PR-AUC is recomputed by hand from the precision-recall curve and must match scikit-learn.*
- *The cost rule is checked on three hand-calculated transactions (the "spreadsheet check"), and the cost curve's
  minimum must equal the cost of the selected threshold.*
- *The API's online features must equal the batch features exactly.*
- *The whole pipeline runs end-to-end in CI on synthetic data.*

## 14. Reproduce

Requires Python 3.11+ and GNU make (on Windows, Git Bash or `choco install make`).

```bash
make setup        # .venv + dependencies
make data-check   # after downloading the CSVs into data/raw/ (see data/README.md)
make all          # ingest -> features -> tune -> train -> evaluate -> policy -> explain -> backtest -> drift -> sql -> writeup
make test         # 100+ tests on synthetic data
```

No Kaggle account? `make sample` runs everything on a generated dataset with the same schema (results land in
`reports/sample/` and mean nothing; it exists to prove the pipeline runs).

| Command | What it does |
|---|---|
| `make dashboard` | Streamlit policy simulator |
| `make api` / `make docker` | scoring API locally / as a container |
| `make mlflow-ui` | browse experiment runs |
| `make notebooks` | execute the notebooks in place |

<details>
<summary>Repo layout</summary>

```
.
├── api/                 FastAPI app + request/response schemas
├── app/                 Streamlit policy simulator
├── configs/             cost assumptions, split fractions, model params (+ tuned params)
├── data/                README with download steps; raw/processed are gitignored
├── docs/                model card, decision log
├── notebooks/           EDA, error analysis, text normalization (thin; logic lives in src/)
├── reports/             generated charts, tables, SQL results, stakeholder memo
├── sql/
│   ├── features/        01-06: join, keys, velocity, history, frequency, assemble
│   └── analysis/        one business question per query
├── src/fraud/           pipeline package (CLI: python -m fraud.cli)
└── tests/               leakage, parity, policy math, metrics, API, end-to-end smoke
```

</details>

See also: [model card](docs/model_card.md) · [design decisions](docs/decisions.md)
