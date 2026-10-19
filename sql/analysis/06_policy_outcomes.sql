-- Q: Under the recommended three-tier policy, what happens to each product line in the
--    test window? Decision x outcome, with dollars. `decisions` is written by `fraud policy`.

SELECT
    s.ProductCD,
    d.decision,
    count(*)                                                AS transactions,
    sum(s.isFraud)                                          AS frauds,
    round(avg(s.isFraud), 4)                                AS fraud_rate,
    round(sum(s.TransactionAmt) FILTER (WHERE s.isFraud = 1)) AS fraud_usd,
    count(*) FILTER (WHERE s.isFraud = 0 AND d.decision = 'decline') AS good_customers_declined
FROM scored AS s
JOIN decisions AS d USING (TransactionID)
WHERE s.split = 'test'
GROUP BY ALL
ORDER BY s.ProductCD, d.decision;
