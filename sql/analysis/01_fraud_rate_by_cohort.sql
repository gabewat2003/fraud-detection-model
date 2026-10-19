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
HAVING count(*) >= $min_support
ORDER BY dimension, fraud_rate DESC;
