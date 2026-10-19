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
HAVING count(*) >= $min_support
ORDER BY min(TransactionAmt), card_velocity_1h;
