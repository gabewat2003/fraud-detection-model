### 02_identity_missingness

Q: What would an INNER join to the identity table have done to the analysis? Compares the full (LEFT-joined) population with the subset an INNER join keeps.

```sql
-- Q: What would an INNER join to the identity table have done to the analysis?
--    Compares the full (LEFT-joined) population with the subset an INNER join keeps.

SELECT
    CASE WHEN has_identity = 1 THEN 'kept by INNER join (has identity)'
         ELSE 'dropped by INNER join (no identity)' END        AS population,
    count(*)                                                     AS transactions,
    round(count(*) / sum(count(*)) OVER (), 4)                   AS share_of_rows,
    round(avg(isFraud), 4)                                       AS fraud_rate,
    round(sum(TransactionAmt) FILTER (WHERE isFraud = 1))        AS fraud_usd,
    round(sum(TransactionAmt) FILTER (WHERE isFraud = 1)
          / sum(sum(TransactionAmt) FILTER (WHERE isFraud = 1)) OVER (), 4) AS share_of_fraud_usd,
    round(median(TransactionAmt), 2)                             AS median_amount
FROM features
GROUP BY 1
ORDER BY 1 DESC;
```

| population | transactions | share_of_rows | fraud_rate | fraud_usd | share_of_fraud_usd | median_amount |
|---|---|---|---|---|---|---|
| kept by INNER join (has identity) | 144233 | 0.2442 | 0.0785 | 1005155.0 | 0.3259 | 50.0 |
| dropped by INNER join (no identity) | 446307 | 0.7558 | 0.0209 | 2078689.0 | 0.6741 | 77.95 |
