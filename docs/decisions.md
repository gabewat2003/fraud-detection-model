# Design decisions

Short records of choices that shaped the project, with the reason and the alternative considered.

### 1. Feature engineering in SQL (DuckDB), checked against pandas
**Why:** window functions (`RANGE ... PRECEDING`) express "only look backward" more clearly than loops, and SQL is
what analyst teams read. DuckDB runs in-process on a 590K x 430 table with no server.
**Check:** an independent pandas implementation must match the SQL exactly (`tests/test_sql_pandas_parity.py`).
**Alternative:** Postgres in Docker. More setup for anyone cloning the repo, same SQL.

### 2. Four time-ordered splits instead of three
**Why:** early stopping, calibration and policy thresholds are three separate fits. Re-using the early-stopping
window for calibration would make the probabilities look better calibrated than they are, and tuning thresholds
on test would make the dollar results optimistic.
**Cost:** less data for training (60% of the time axis).

### 3. No SMOTE / resampling
**Why:** gradient boosting handles 3.5% positives fine, and resampling distorts probabilities, which the cost rule
multiplies by dollars. Class weighting (`scale_pos_weight`) is tried as a variant and then calibrated.

### 4. Account proxy `uid = card1 | addr1 | (day - D1)`
**Why:** there is no card ID. `D1` behaves like "days since the card was first seen", so `day - D1` is roughly
constant per card. The idea comes from the Kaggle IEEE-CIS community; it is an approximation and is labeled as one.

### 5. Frequency encodings counted on the training window only; no frequency encoding of account keys
**Why:** counting over all data lets future traffic leak into early rows. For account keys, even training-window
counts include a row's own future transactions, and test rows (often new accounts) would get systematically
lower counts than training rows. Backward-looking history features cover that signal without the leak.

### 6. Isotonic calibration
**Why:** non-parametric, monotone, fit on its own window. Platt scaling assumes a sigmoid shape that boosted
scores often don't have. **Side effect:** isotonic output is a step function with ties, so rank-based rules
(top 1%, flat cutoffs) use raw scores, and cost-based rules use calibrated probabilities.

### 7. Review capacity as a threshold fit on calib, not a per-day top-k
**Why:** a real-time system decides per transaction and cannot see the rest of the day. The bar is set so review
volume matches capacity on the calibration window, then applied unchanged; the queue chart shows the days it
overflows. **Fix for the overflow:** a hard daily cap, first come first served, with overflow decided
automatically by expected cost (`policy.enforce_daily_cap`). It needs no look-ahead and costs about 2 points of
saving on test.

### 11. Reviewers are charged for their mistakes
**Why:** a flat "$3, always right" review cost made the review tier look like most of the value. Each reviewed
case now carries `(1 - accuracy) x` the cost of the wrong automatic action, with accuracy 90% in the base case and
80/90/100% x $3/$8/$15 in a sensitivity grid. With accuracy 100% the formula reduces to the old flat cost.
**Limitation:** mistakes are assumed independent of the case; real reviewers are likely worse on hard cases.

### 12. Champion chosen on validation, not test
**Why:** plain and tuned LightGBM are within noise on test, and which one gives the cheaper policy flips with
the cost assumptions. Picking the test winner would make the test window a second validation set.

### 8. Text normalization as rules, not learned embeddings
**Why:** a few thousand distinct strings, mostly structured (brand prefixes, version numbers, TLDs). Ordered regex
rules are transparent, unit-testable and identical online and offline. Coverage is reported, including the
top unmapped strings.

### 9. Serving reuses the batch SQL
**Why:** train/serve skew is a classic production bug. The API writes the incoming transaction into empty
raw-shaped tables and runs the same SQL files against that card's history; a test requires identical features.
**Cost:** ~300 ms per call. Production would maintain running aggregates in a feature store.

### 10. TreeSHAP via LightGBM `pred_contrib`
**Why:** the same algorithm as `shap.TreeExplainer`, with no extra dependency in the serving image.
