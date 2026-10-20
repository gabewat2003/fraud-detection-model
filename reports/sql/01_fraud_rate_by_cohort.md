### 01_fraud_rate_by_cohort

Q: Where is fraud concentrated? Fraud rate and fraud dollars by product, device type, identity presence, card network and card type, with lift over the overall rate.

```sql
-- Q: Where is fraud concentrated? Fraud rate and fraud dollars by product, device type,
--    identity presence, card network and card type, with lift over the overall rate.

WITH overall AS (SELECT avg(isFraud) AS rate FROM features),
cohorts AS (
    SELECT 'ProductCD' AS dimension, ProductCD AS value, isFraud, TransactionAmt FROM features
    UNION ALL SELECT 'DeviceType',   coalesce(DeviceType, '(no identity)'), isFraud, TransactionAmt FROM features
    UNION ALL SELECT 'has_identity', CAST(has_identity AS VARCHAR), isFraud, TransactionAmt FROM features
    UNION ALL SELECT 'card4',        coalesce(card4, '(missing)'), isFraud, TransactionAmt FROM features
    UNION ALL SELECT 'card6',        coalesce(card6, '(missing)'), isFraud, TransactionAmt FROM features
)
SELECT
    dimension,
    value,
    count(*)                                              AS transactions,
    round(avg(isFraud), 4)                                AS fraud_rate,
    round(avg(isFraud) / (SELECT rate FROM overall), 2)   AS lift,
    round(sum(TransactionAmt) FILTER (WHERE isFraud = 1)) AS fraud_usd
FROM cohorts
GROUP BY dimension, value
HAVING count(*) >= 500
ORDER BY dimension, fraud_rate DESC;
```

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
