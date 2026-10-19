-- 01 · Base table: transactions LEFT JOIN identity.
--
-- Only ~24% of transactions have an identity row. An INNER join would silently drop the
-- other ~76% and bias every rate computed afterwards, so this is a LEFT join and the
-- presence of an identity row becomes a feature of its own (has_identity).
--
-- TransactionDT is seconds since an unknown reference date. tx_day / hour / dow are only
-- cohort labels: dow = 0 does not mean Monday.

CREATE OR REPLACE TABLE base AS
SELECT
    t.*,
    i.* EXCLUDE (TransactionID),
    CAST(i.TransactionID IS NOT NULL AS INTEGER)          AS has_identity,
    CAST(floor(t.TransactionDT / 86400) AS INTEGER)       AS tx_day,
    CAST(floor(t.TransactionDT / 3600) AS INTEGER) % 24   AS hour,
    CAST(floor(t.TransactionDT / 86400) AS INTEGER) % 7   AS dow
FROM transactions AS t
LEFT JOIN identity AS i
    ON t.TransactionID = i.TransactionID;
