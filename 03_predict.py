"""Step 3: Score new customers with the saved model.

Usage:  python3 03_predict.py            (scores 3 built-in example customers)
        python3 03_predict.py new.csv    (CSV with the same columns as clean data, minus customer_id/churned)
"""
import sys

import joblib
import pandas as pd

bundle = joblib.load("models/churn_model.joblib")
pipe, threshold, features = bundle["pipeline"], bundle["threshold"], bundle["features"]

if len(sys.argv) > 1:
    new = pd.read_csv(sys.argv[1])
else:
    new = pd.DataFrame([
        dict(county="Kiambu", age=29, plan="Basic", tenure_months=3, monthly_transactions=4,
             avg_txn_amount_kes=250.0, failed_txn_pct=18.0, support_calls_3m=5,
             days_since_last_txn=40, uses_savings_product="No"),
        dict(county="Nairobi", age=41, plan="Premium", tenure_months=60, monthly_transactions=30,
             avg_txn_amount_kes=1500.0, failed_txn_pct=1.0, support_calls_3m=0,
             days_since_last_txn=1, uses_savings_product="Yes"),
        dict(county="Kisumu", age=None, plan="Standard", tenure_months=24, monthly_transactions=15,
             avg_txn_amount_kes=None, failed_txn_pct=6.0, support_calls_3m=2,
             days_since_last_txn=15, uses_savings_product="No"),
    ])

new = new[features]
new["churn_probability"] = pipe.predict_proba(new)[:, 1].round(3)
new["risk"] = (new["churn_probability"] >= threshold).map({True: "HIGH - retain", False: "low"})
print(new[["plan", "tenure_months", "days_since_last_txn", "churn_probability", "risk"]].to_string(index=False))
