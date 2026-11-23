import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap


def load_dataset(csv_path: str = "data/creditcard.csv") -> pd.DataFrame:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    X, y = make_classification(
        n_samples=25000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        weights=[0.9983, 0.0017],
        flip_y=0.0005,
        random_state=42,
    )
    cols = [f"V{i}" for i in range(1, 20)] + ["Amount"]
    df = pd.DataFrame(X, columns=cols)
    df["Amount"] = np.abs(df["Amount"]) * 120 + 5
    df["Class"] = y
    return df


def calibrate_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    baseline_alerts = int((y_prob >= 0.30).sum())
    for thresh in [0.30, 0.45, 0.60, 0.75, 0.85]:
        preds = (y_prob >= thresh).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        total_alerts = tp + fp
        alert_reduction = (baseline_alerts - total_alerts) / max(baseline_alerts, 1)
        rows.append({
            "threshold": thresh,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "total_alerts": total_alerts,
            "alert_reduction_vs_baseline": round(alert_reduction, 4),
        })
    return pd.DataFrame(rows)


def main():
    df = load_dataset()
    X = df.drop(columns=["Class"])
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    smote = SMOTE(sampling_strategy=0.20, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_res, y_train_res)

    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"Test PR-AUC: {average_precision_score(y_test, y_prob):.4f}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[:500])
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)
    print("\nTop 5 SHAP Risk Drivers:\n", mean_abs_shap.head(5))

    tradeoff_df = calibrate_thresholds(y_test, y_prob)
    Path("outputs").mkdir(exist_ok=True)
    tradeoff_df.to_csv("outputs/threshold_tradeoffs.csv", index=False)


if __name__ == "__main__":
    main()
