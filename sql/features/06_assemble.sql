-- 06 · Assemble the modeling table: raw + normalized columns, simple transforms,
-- velocity, history and frequency features. One row per TransactionID.

CREATE OR REPLACE TABLE features AS
SELECT
    k.*,
    ln(1 + k.TransactionAmt)                                   AS amt_log,
    round(k.TransactionAmt - floor(k.TransactionAmt), 3)       AS amt_cents,
    CASE WHEN k.P_emaildomain IS NULL OR k.R_emaildomain IS NULL THEN NULL
         WHEN k.P_emaildomain = k.R_emaildomain THEN 1 ELSE 0 END             AS email_match,
    CASE WHEN k.p_email_provider IS NULL OR k.r_email_provider IS NULL THEN NULL
         WHEN k.p_email_provider = k.r_email_provider THEN 1 ELSE 0 END       AS email_provider_match,
    v.* EXCLUDE (TransactionID),
    h.* EXCLUDE (TransactionID),
    CASE WHEN h.uid_prior_n = 0 THEN 1 WHEN h.uid_prior_n > 0 THEN 0 END       AS is_first_txn_uid,
    k.TransactionAmt / NULLIF(h.uid_prior_mean_amt, 0)                        AS amt_vs_uid_mean,
    (k.TransactionAmt - h.uid_prior_mean_amt) / NULLIF(h.uid_prior_std_amt, 0) AS amt_zscore_uid,
    k.TransactionAmt / NULLIF(h.uid_prior_max_amt, 0)                         AS amt_vs_uid_max,
    k.TransactionAmt / NULLIF(h.card_prior_mean_amt, 0)                       AS amt_vs_card_mean,
    coalesce(f1.n, 0)  AS freq_card1,
    coalesce(f2.n, 0)  AS freq_card2,
    coalesce(fa.n, 0)  AS freq_addr1,
    coalesce(fe.n, 0)  AS freq_p_email,
    coalesce(fd.n, 0)  AS freq_device,
    -- DuckDB's greatest() skips NULLs, so guard explicitly: unknown version stays NULL
    CASE WHEN k.browser_major IS NULL OR bl.max_major IS NULL THEN NULL
         ELSE greatest(bl.max_major - k.browser_major, 0) END                 AS browser_versions_behind
FROM keyed AS k
JOIN feat_velocity AS v ON k.TransactionID = v.TransactionID
JOIN feat_history  AS h ON k.TransactionID = h.TransactionID
LEFT JOIN freq_card1     AS f1 ON k.card1 = f1.key
LEFT JOIN freq_card2     AS f2 ON k.card2 = f2.key
LEFT JOIN freq_addr1     AS fa ON k.addr1 = fa.key
LEFT JOIN freq_p_email   AS fe ON k.p_email_provider = fe.key
LEFT JOIN freq_device    AS fd ON k.device_family = fd.key
LEFT JOIN browser_latest AS bl ON k.browser_family = bl.browser_family;
