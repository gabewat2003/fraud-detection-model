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
