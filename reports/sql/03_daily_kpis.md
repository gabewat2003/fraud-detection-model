### 03_daily_kpis

Q: The daily KPI table a fraud-ops dashboard would sit on: volume, fraud rate, fraud dollars, and a trailing 7-day fraud rate (window function) to smooth daily noise. tx_day is relative: the dataset's reference date is unknown.

```sql
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
```

| tx_day | transactions | frauds | fraud_rate | fraud_rate_7d | volume_usd | fraud_usd | fraud_usd_share |
|---|---|---|---|---|---|---|---|
| 1 | 5122 | 112.0 | 0.0219 | 0.0219 | 742187.0 | 20337.0 | 0.0274 |
| 2 | 3730 | 123.0 | 0.033 | 0.0265 | 453400.0 | 15360.0 | 0.0339 |
| 3 | 3241 | 92.0 | 0.0284 | 0.027 | 348517.0 | 14254.0 | 0.0409 |
| 4 | 4036 | 115.0 | 0.0285 | 0.0274 | 515969.0 | 14120.0 | 0.0274 |
| 5 | 3964 | 127.0 | 0.032 | 0.0283 | 543141.0 | 16693.0 | 0.0307 |
| 6 | 3717 | 99.0 | 0.0266 | 0.0281 | 476861.0 | 12869.0 | 0.027 |
| 7 | 3786 | 136.0 | 0.0359 | 0.0291 | 534372.0 | 21670.0 | 0.0406 |
| 8 | 3999 | 93.0 | 0.0233 | 0.0297 | 546578.0 | 13508.0 | 0.0247 |
| 9 | 3907 | 119.0 | 0.0305 | 0.0293 | 456673.0 | 13107.0 | 0.0287 |
| 10 | 3445 | 90.0 | 0.0261 | 0.029 | 355377.0 | 12710.0 | 0.0358 |
| 11 | 4319 | 123.0 | 0.0285 | 0.029 | 579747.0 | 21726.0 | 0.0375 |
| 12 | 4164 | 84.0 | 0.0202 | 0.0272 | 518738.0 | 14711.0 | 0.0284 |
| 13 | 4183 | 82.0 | 0.0196 | 0.0261 | 559196.0 | 10975.0 | 0.0196 |
| 14 | 4446 | 126.0 | 0.0283 | 0.0252 | 605742.0 | 15782.0 | 0.0261 |
| 15 | 5034 | 143.0 | 0.0284 | 0.026 | 697885.0 | 16439.0 | 0.0236 |
| 16 | 4325 | 70.0 | 0.0162 | 0.024 | 503903.0 | 7280.0 | 0.0144 |
| 17 | 3805 | 126.0 | 0.0331 | 0.0249 | 402041.0 | 14338.0 | 0.0357 |
| 18 | 5585 | 161.0 | 0.0288 | 0.0251 | 749350.0 | 19746.0 | 0.0264 |
| 19 | 5526 | 118.0 | 0.0214 | 0.0251 | 775397.0 | 13648.0 | 0.0176 |
| 20 | 5749 | 135.0 | 0.0235 | 0.0255 | 795645.0 | 20878.0 | 0.0262 |

_182 rows; first 20 shown, full result in `03_daily_kpis.csv`._
