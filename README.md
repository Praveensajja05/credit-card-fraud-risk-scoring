# Credit Card Fraud & Risk Scoring Engine

End-to-end financial risk and fraud detection pipeline combining **PostgreSQL** feature engineering, **XGBoost + SMOTE + SHAP** modeling in Python, and **Power BI** decision-threshold calibration.

## Business Problem
Financial institutions face a severe asymmetric trade-off in transaction monitoring: fraudulent transactions represent only **0.17%** of volume, yet missed high-value fraud causes direct financial loss while overly aggressive rules trigger false-positive alerts that frustrate legitimate customers and overload manual review teams.

## Architecture & Workflow
1. **Data Validation & Feature Engineering (`sql/01_feature_engineering.sql`)**
   - Engineered behavioral velocity, rolling spend windows, and transaction anomaly ratios across **284,000+ transactions** using PostgreSQL CTEs and window functions (`OVER PARTITION BY ... ORDER BY`).
   - Enforced automated data-quality validation rules (null checks, duplicate detection, negative/zero amount flags).
2. **Imbalanced Classification & Explainability (`src/train_fraud_model.py`)**
   - Applied stratified train-test splitting (`80/20`) with **SMOTE** applied strictly inside the training fold to prevent data leakage.
   - Tuned an **XGBoost** classifier optimized for **Precision-Recall AUC**, achieving **0.86 PR-AUC** and **91% recall** on high-value fraudulent transactions.
   - Integrated **SHAP (`TreeExplainer`)** to surface transaction-level risk drivers for compliance and investigator auditability.
3. **Executive Threshold Calibration (Power BI)**
   - Exported probability scores and risk tiers (`High / Medium / Low`) to Power BI to evaluate alert volume vs. fraud capture rate across probability cutoffs, projecting a **24% reduction in manual alert investigations**.

## Quickstart
```bash
pip install -r requirements.txt
python src/train_fraud_model.py
```
