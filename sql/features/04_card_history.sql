-- 04 · Spend history: how does this amount compare with the account's earlier spending?
--
-- Frames are ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING, ordered by
-- (TransactionDT, TransactionID): every earlier transaction, never the current one.
-- An account's first transaction therefore gets n = 0 and a NULL mean, which
-- tests/test_no_leakage.py asserts.

CREATE OR REPLACE TABLE feat_history AS
SELECT
    TransactionID,
    CASE WHEN uid IS NULL THEN NULL ELSE COUNT(*)                    OVER uid_prior END AS uid_prior_n,
    CASE WHEN uid IS NULL THEN NULL ELSE AVG(TransactionAmt)         OVER uid_prior END AS uid_prior_mean_amt,
    CASE WHEN uid IS NULL THEN NULL ELSE STDDEV_SAMP(TransactionAmt) OVER uid_prior END AS uid_prior_std_amt,
    CASE WHEN uid IS NULL THEN NULL ELSE MAX(TransactionAmt)         OVER uid_prior END AS uid_prior_max_amt,
    COUNT(*)            OVER card_prior AS card_prior_n,
    AVG(TransactionAmt) OVER card_prior AS card_prior_mean_amt
FROM window_input
WINDOW
    uid_prior  AS (PARTITION BY uid      ORDER BY TransactionDT, TransactionID
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    card_prior AS (PARTITION BY card_key ORDER BY TransactionDT, TransactionID
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);
