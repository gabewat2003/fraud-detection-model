### 05_card_testing

Q: Is there card-testing behaviour (bursts of tiny charges)? Fraud rate by amount band and by how many times the card was used in the previous hour.

```sql
-- Q: Is there card-testing behaviour (bursts of tiny charges)? Fraud rate by amount band
--    and by how many times the card was used in the previous hour.

SELECT
    CASE WHEN TransactionAmt < 10 THEN '< $10'
         WHEN TransactionAmt < 50 THEN '$10-50'
         WHEN TransactionAmt < 200 THEN '$50-200'
         ELSE '$200+' END                              AS amount_band,
    CASE WHEN card_tx_1h = 0 THEN '0 prior in 1h'
         WHEN card_tx_1h < 3 THEN '1-2 prior in 1h'
         ELSE '3+ prior in 1h' END                     AS card_velocity_1h,
    count(*)                                           AS transactions,
    round(avg(isFraud), 4)                             AS fraud_rate
FROM features
GROUP BY ALL
HAVING count(*) >= 500
ORDER BY min(TransactionAmt), card_velocity_1h;
```

| amount_band | card_velocity_1h | transactions | fraud_rate |
|---|---|---|---|
| < $10 | 0 prior in 1h | 3429 | 0.0782 |
| < $10 | 3+ prior in 1h | 1310 | 0.0855 |
| < $10 | 1-2 prior in 1h | 2056 | 0.0905 |
| $10-50 | 0 prior in 1h | 131311 | 0.0289 |
| $10-50 | 1-2 prior in 1h | 37323 | 0.0592 |
| $10-50 | 3+ prior in 1h | 12086 | 0.0747 |
| $50-200 | 0 prior in 1h | 232586 | 0.0239 |
| $50-200 | 1-2 prior in 1h | 56645 | 0.0423 |
| $50-200 | 3+ prior in 1h | 13324 | 0.0499 |
| $200+ | 0 prior in 1h | 77475 | 0.041 |
| $200+ | 1-2 prior in 1h | 19465 | 0.0595 |
| $200+ | 3+ prior in 1h | 3530 | 0.0649 |
