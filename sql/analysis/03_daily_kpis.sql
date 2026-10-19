-- Q: The daily KPI table a fraud-ops dashboard would sit on: volume, fraud rate, fraud
--    dollars, and a trailing 7-day fraud rate (window function) to smooth daily noise.
--    tx_day is relative: the dataset's reference date is unknown.

WITH daily AS (
    SELECT
        tx_day,
        count(*)                                        AS transactions,
        sum(isFraud)                                    AS frauds,
        round(sum(TransactionAmt))                      AS volume_usd,
        round(sum(TransactionAmt) FILTER (WHERE isFraud = 1)) AS fraud_usd
    FROM features
    GROUP BY tx_day
)
SELECT
    tx_day,
    transactions,
    frauds,
    round(frauds / transactions, 4) AS fraud_rate,
    round(sum(frauds) OVER w7 / sum(transactions) OVER w7, 4) AS fraud_rate_7d,
    volume_usd,
    fraud_usd,
    round(fraud_usd / volume_usd, 4) AS fraud_usd_share
FROM daily
WINDOW w7 AS (ORDER BY tx_day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
ORDER BY tx_day;
