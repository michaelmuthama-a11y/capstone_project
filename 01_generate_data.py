"""Step 1: Generate a SYNTHETIC mobile-money customer dataset (with realistic mess).

The data is simulated, NOT real customer data. Churn probability is driven by a
hidden logistic formula so that models have genuine signal to learn.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

counties = ["Nairobi", "Kiambu", "Mombasa", "Kisumu", "Nakuru", "Machakos", "Uasin Gishu", "Nyeri"]
plans = ["Basic", "Standard", "Premium"]

df = pd.DataFrame({
    "customer_id": [f"C{100000 + i}" for i in range(N)],
    "county": rng.choice(counties, N, p=[.28, .16, .13, .09, .1, .09, .08, .07]),
    "age": rng.integers(18, 70, N),
    "plan": rng.choice(plans, N, p=[.5, .35, .15]),
    "tenure_months": rng.integers(1, 72, N),
    "monthly_transactions": rng.poisson(18, N),
    "avg_txn_amount_kes": np.round(rng.lognormal(6.2, 0.7, N), 2),
    "failed_txn_pct": np.round(rng.beta(1.5, 14, N) * 100, 2),
    "support_calls_3m": rng.poisson(1.2, N),
    "days_since_last_txn": rng.integers(0, 60, N),
    "uses_savings_product": rng.choice(["Yes", "No"], N, p=[.35, .65]),
})

# Hidden churn mechanism
z = (
    -2.8
    + 0.045 * df["days_since_last_txn"]
    + 0.09 * df["failed_txn_pct"]
    + 0.30 * df["support_calls_3m"]
    - 0.025 * df["tenure_months"]
    - 0.03 * df["monthly_transactions"]
    - 0.6 * (df["uses_savings_product"] == "Yes")
    + 0.4 * (df["plan"] == "Basic")
)
p = 1 / (1 + np.exp(-z))
df["churned"] = (rng.random(N) < p).astype(int)

# ---- Inject realistic data-quality problems ----
for col, frac in [("age", .03), ("avg_txn_amount_kes", .04), ("failed_txn_pct", .02)]:
    idx = rng.choice(N, int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

idx = rng.choice(N, 60, replace=False)           # impossible values
df.loc[idx[:30], "age"] = rng.choice([-5, 0, 150, 200], 30)
df.loc[idx[30:], "avg_txn_amount_kes"] = -abs(df.loc[idx[30:], "avg_txn_amount_kes"].fillna(100))

idx = rng.choice(N, 250, replace=False)           # inconsistent labels
df.loc[idx[:100], "county"] = df.loc[idx[:100], "county"].str.upper()
df.loc[idx[100:200], "county"] = " " + df.loc[idx[100:200], "county"].str.lower() + " "
df.loc[idx[200:], "uses_savings_product"] = rng.choice(["Y", "N", "yes", "no"], 50)

dups = df.sample(80, random_state=1)              # duplicate rows
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/raw_customers.csv", index=False)
print("Saved data/raw_customers.csv", df.shape)
print("Raw churn rate:", round(df["churned"].mean(), 3))
