-- Credit Card Fraud & Risk Scoring Engine: Validation & Feature Engineering (PostgreSQL)

CREATE TABLE IF NOT EXISTS raw_transactions (
    txn_id          BIGSERIAL PRIMARY KEY,
    account_id      VARCHAR(32) NOT NULL,
    txn_timestamp   TIMESTAMP NOT NULL,
    amount          NUMERIC(12, 2) NOT NULL,
    merchant_cat    VARCHAR(64),
    is_fraud        SMALLINT NOT NULL CHECK (is_fraud IN (0, 1))
);

-- 1. Automated Data-Quality Validation Audit
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE account_id IS NULL OR txn_timestamp IS NULL) AS missing_keys,
    COUNT(*) FILTER (WHERE amount <= 0) AS non_positive_amounts,
    COUNT(*) - COUNT(DISTINCT (account_id, txn_timestamp, amount)) AS potential_duplicates
FROM raw_transactions;

-- 2. Behavioral Velocity & Rolling Spend Feature Engineering
WITH cleaned_txns AS (
    SELECT DISTINCT ON (account_id, txn_timestamp, amount)
        txn_id, account_id, txn_timestamp, amount, merchant_cat, is_fraud
    FROM raw_transactions
    WHERE amount > 0 AND account_id IS NOT NULL
),
behavioral_features AS (
    SELECT
        txn_id,
        account_id,
        txn_timestamp,
        amount,
        is_fraud,
        COUNT(*) OVER (
            PARTITION BY account_id
            ORDER BY txn_timestamp
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
        ) AS txn_count_1h,
        COUNT(*) OVER (
            PARTITION BY account_id
            ORDER BY txn_timestamp
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        ) AS txn_count_24h,
        SUM(amount) OVER (
            PARTITION BY account_id
            ORDER BY txn_timestamp
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        ) AS rolling_spend_24h,
        AVG(amount) OVER (
            PARTITION BY account_id
            ORDER BY txn_timestamp
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS trailing_30_avg_amount
    FROM cleaned_txns
)
SELECT
    *,
    ROUND(amount / NULLIF(trailing_30_avg_amount, 0), 3) AS amount_to_baseline_ratio
FROM behavioral_features;
