-- Q: Which combinations of product, email provider and device brand carry the highest
--    fraud rate, among segments big enough to act on? (Uses the normalized text fields.)

WITH overall AS (SELECT avg(isFraud) AS rate FROM features),
seg AS (
    SELECT
        ProductCD,
        coalesce(p_email_provider, '(none)') AS email_provider,
        coalesce(device_brand, '(none)')     AS device_brand,
        count(*)      AS transactions,
        avg(isFraud)  AS fraud_rate,
        sum(TransactionAmt) FILTER (WHERE isFraud = 1) AS fraud_usd
    FROM features
    GROUP BY ALL
    HAVING count(*) >= $min_support
)
SELECT
    ProductCD, email_provider, device_brand, transactions,
    round(fraud_rate, 4)                            AS fraud_rate,
    round(fraud_rate / (SELECT rate FROM overall), 1) AS lift,
    round(fraud_usd)                                AS fraud_usd,
    rank() OVER (ORDER BY fraud_rate DESC)          AS risk_rank
FROM seg
ORDER BY fraud_rate DESC
LIMIT 15;
