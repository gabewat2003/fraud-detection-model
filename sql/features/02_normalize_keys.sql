-- 02 · Attach normalized text fields and build the two account proxies.
--
-- map_device / map_os / map_browser / map_email are lookup tables built in Python by
-- fraud.text_norm from the distinct raw strings (a few thousand values, not 590K rows).
--
-- The data has no card or account ID, so two approximations are used:
--   card_key : card1 | card2 | addr1             (masked card attributes + billing region)
--   uid      : card1 | addr1 | (tx_day - D1)     D1 counts days since the card was first
--              seen, so tx_day - D1 stays roughly constant for one card: an inferred
--              "card start day". The idea comes from the Kaggle IEEE-CIS community.
-- Both are proxies: collisions group near-identical cards together.

CREATE OR REPLACE TABLE keyed AS
SELECT
    b.*,
    md.device_brand,
    md.device_family,
    mo.os_family,
    mo.os_version,
    mb.browser_family,
    mb.browser_major,
    mb.browser_is_mobile,
    pe.email_provider  AS p_email_provider,
    pe.email_tld       AS p_email_tld,
    pe.is_free_mail    AS p_is_free_mail,
    pe.email_region    AS p_email_region,
    re.email_provider  AS r_email_provider,
    re.email_tld       AS r_email_tld,
    re.is_free_mail    AS r_is_free_mail,
    concat_ws('_',
        CAST(b.card1 AS VARCHAR),
        coalesce(CAST(CAST(b.card2 AS INTEGER) AS VARCHAR), 'NA'),
        coalesce(CAST(CAST(b.addr1 AS INTEGER) AS VARCHAR), 'NA')
    ) AS card_key,
    CASE WHEN b.D1 IS NULL THEN NULL ELSE concat_ws('_',
        CAST(b.card1 AS VARCHAR),
        coalesce(CAST(CAST(b.addr1 AS INTEGER) AS VARCHAR), 'NA'),
        CAST(CAST(b.tx_day - floor(b.D1) AS INTEGER) AS VARCHAR)
    ) END AS uid
FROM base AS b
LEFT JOIN map_device  AS md ON b.DeviceInfo    = md.raw
LEFT JOIN map_os      AS mo ON b.id_30         = mo.raw
LEFT JOIN map_browser AS mb ON b.id_31         = mb.raw
LEFT JOIN map_email   AS pe ON b.P_emaildomain = pe.raw
LEFT JOIN map_email   AS re ON b.R_emaildomain = re.raw;
