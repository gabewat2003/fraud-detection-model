### 06_policy_outcomes

Q: Under the recommended three-tier policy, what happens to each product line in the test window? Decision x outcome, with dollars. `decisions` is written by `fraud policy`.

```sql
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
```

| ProductCD | decision | transactions | frauds | fraud_rate | fraud_usd | good_customers_declined |
|---|---|---|---|---|---|---|
| C | approve | 7858 | 308.0 | 0.0392 | 13323.0 | 0 |
| C | decline | 1529 | 962.0 | 0.6292 | 47645.0 | 567 |
| C | review | 1 | 0.0 | 0.0 |  | 0 |
| H | approve | 2356 | 65.0 | 0.0276 | 6850.0 | 0 |
| H | decline | 185 | 90.0 | 0.4865 | 13310.0 | 95 |
| H | review | 21 | 9.0 | 0.4286 | 3600.0 | 0 |
| R | approve | 3937 | 42.0 | 0.0107 | 6300.0 | 0 |
| R | decline | 223 | 157.0 | 0.704 | 40975.0 | 66 |
| R | review | 30 | 13.0 | 0.4333 | 7225.0 | 0 |
| S | approve | 2867 | 82.0 | 0.0286 | 3900.0 | 0 |
| S | decline | 106 | 62.0 | 0.5849 | 2395.0 | 44 |
| W | approve | 67989 | 963.0 | 0.0142 | 177684.0 | 0 |
| W | decline | 605 | 175.0 | 0.2893 | 24456.0 | 430 |
| W | review | 874 | 155.0 | 0.1773 | 121945.0 | 0 |
