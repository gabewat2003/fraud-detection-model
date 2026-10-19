-- 05 · Frequency lookups, computed on the TRAINING window only.
--
-- "How common is this card1 / region / email provider / device?" Rare values are riskier.
-- Counting over the whole dataset would let future traffic leak into early rows, so the
-- counts come from transactions with TransactionDT <= $train_end and are joined onto
-- every split. Values never seen in training get 0.
--
-- Account-level keys (uid, card_key) are deliberately NOT frequency-encoded: inside the
-- training window a row's own future transactions would be counted. The backward-looking
-- history features in 04 cover that signal safely.

CREATE OR REPLACE TABLE freq_card1 AS
    SELECT card1 AS key, COUNT(*) AS n FROM keyed WHERE TransactionDT <= $train_end GROUP BY 1;
CREATE OR REPLACE TABLE freq_card2 AS
    SELECT card2 AS key, COUNT(*) AS n FROM keyed WHERE TransactionDT <= $train_end GROUP BY 1;
CREATE OR REPLACE TABLE freq_addr1 AS
    SELECT addr1 AS key, COUNT(*) AS n FROM keyed WHERE TransactionDT <= $train_end GROUP BY 1;
CREATE OR REPLACE TABLE freq_p_email AS
    SELECT p_email_provider AS key, COUNT(*) AS n FROM keyed WHERE TransactionDT <= $train_end GROUP BY 1;
CREATE OR REPLACE TABLE freq_device AS
    SELECT device_family AS key, COUNT(*) AS n FROM keyed WHERE TransactionDT <= $train_end GROUP BY 1;

-- Newest browser major version seen per family during training. Used to measure how many
-- versions behind a transaction's browser is (outdated browsers skew toward automation).
CREATE OR REPLACE TABLE browser_latest AS
    SELECT browser_family, MAX(browser_major) AS max_major
    FROM keyed
    WHERE TransactionDT <= $train_end AND browser_major IS NOT NULL
    GROUP BY 1;
