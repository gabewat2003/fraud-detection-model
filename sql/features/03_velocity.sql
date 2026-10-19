-- 03 · Velocity features: how busy was this card just before this transaction?
--
-- Reads `window_input`, a narrow projection of `keyed`. The scoring API swaps in a table
-- holding one card's history plus the incoming transaction, so the same SQL runs offline
-- and online.
--
-- Leakage rule: every time frame ends at "1 PRECEDING" on the TransactionDT axis, so only
-- transactions at least one second EARLIER are counted. The current row, and anything at
-- the same second or later, is never visible. tests/test_no_leakage.py checks this.

CREATE OR REPLACE TABLE feat_velocity AS
SELECT
    TransactionID,
    COUNT(*)                        OVER card_1h   AS card_tx_1h,
    COUNT(*)                        OVER card_24h  AS card_tx_24h,
    COUNT(*)                        OVER card_7d   AS card_tx_7d,
    coalesce(SUM(TransactionAmt)    OVER card_24h, 0) AS card_amt_24h,
    CASE WHEN uid IS NULL THEN NULL ELSE COUNT(*) OVER uid_1h  END AS uid_tx_1h,
    CASE WHEN uid IS NULL THEN NULL ELSE COUNT(*) OVER uid_24h END AS uid_tx_24h,
    CASE WHEN uid IS NULL THEN NULL ELSE COUNT(*) OVER uid_7d  END AS uid_tx_7d,
    COUNT(DISTINCT addr1)            OVER card1_7d AS card1_n_addr_7d,
    COUNT(DISTINCT p_email_provider) OVER card1_7d AS card1_n_email_7d,
    COUNT(DISTINCT device_family)    OVER card1_7d AS card1_n_device_7d,
    TransactionDT - LAG(TransactionDT) OVER card_seq AS card_secs_since_prev,
    CASE WHEN uid IS NULL THEN NULL
         ELSE TransactionDT - LAG(TransactionDT) OVER uid_seq END AS uid_secs_since_prev
FROM window_input
WINDOW
    card_1h  AS (PARTITION BY card_key ORDER BY TransactionDT RANGE BETWEEN 3600   PRECEDING AND 1 PRECEDING),
    card_24h AS (PARTITION BY card_key ORDER BY TransactionDT RANGE BETWEEN 86400  PRECEDING AND 1 PRECEDING),
    card_7d  AS (PARTITION BY card_key ORDER BY TransactionDT RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING),
    uid_1h   AS (PARTITION BY uid      ORDER BY TransactionDT RANGE BETWEEN 3600   PRECEDING AND 1 PRECEDING),
    uid_24h  AS (PARTITION BY uid      ORDER BY TransactionDT RANGE BETWEEN 86400  PRECEDING AND 1 PRECEDING),
    uid_7d   AS (PARTITION BY uid      ORDER BY TransactionDT RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING),
    card1_7d AS (PARTITION BY card1    ORDER BY TransactionDT RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING),
    card_seq AS (PARTITION BY card_key ORDER BY TransactionDT, TransactionID),
    uid_seq  AS (PARTITION BY uid      ORDER BY TransactionDT, TransactionID);
