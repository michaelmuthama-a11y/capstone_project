"""Step 2: Clean data, explore it, train/compare models, save the best one."""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             precision_recall_fscore_support, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
log = {}

# ------------------------------------------------------------------ 1. LOAD
raw = pd.read_csv("data/raw_customers.csv")
log["raw_rows"] = len(raw)

# ------------------------------------------------------------------ 2. CLEAN
df = raw.copy()
log["duplicate_rows_removed"] = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

df["county"] = df["county"].str.strip().str.title()
df["uses_savings_product"] = (df["uses_savings_product"].str.strip().str.lower()
                              .map({"yes": "Yes", "y": "Yes", "no": "No", "n": "No"}))

bad_age = (df["age"] < 18) | (df["age"] > 100)
bad_amt = df["avg_txn_amount_kes"] < 0
log["impossible_age_set_to_nan"] = int(bad_age.sum())
log["negative_amount_set_to_nan"] = int(bad_amt.sum())
df.loc[bad_age, "age"] = np.nan
df.loc[bad_amt, "avg_txn_amount_kes"] = np.nan

log["missing_before_imputation"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
log["clean_rows"] = len(df)
df.to_csv("data/clean_customers.csv", index=False)

# ------------------------------------------------------------------ 3. EDA
churn_rate = df["churned"].mean()
log["churn_rate"] = round(float(churn_rate), 4)

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax = ax.ravel()
df["churned"].value_counts().sort_index().plot.bar(ax=ax[0], color=["#4C78A8", "#E45756"])
ax[0].set_title("Class balance (0=stayed, 1=churned)")
ax[0].set_xticklabels(["Stayed", "Churned"], rotation=0)
ax[0].set_xlabel("")
for i, col in enumerate(["days_since_last_txn", "failed_txn_pct", "tenure_months", "support_calls_3m"], 1):
    for label, colr in [(0, "#4C78A8"), (1, "#E45756")]:
        ax[i].hist(df.loc[df.churned == label, col].dropna(), bins=25, alpha=.6,
                   color=colr, label=["Stayed", "Churned"][label], density=True)
    ax[i].set_title(col)
    ax[i].legend()
df.groupby("plan")["churned"].mean().reindex(["Basic", "Standard", "Premium"]).plot.bar(ax=ax[5], color="#F58518")
ax[5].set_title("Churn rate by plan")
ax[5].set_xticklabels(["Basic", "Standard", "Premium"], rotation=0)
plt.tight_layout()
plt.savefig("outputs/eda_overview.png", dpi=130)
plt.close()

num_cols = df.select_dtypes("number").drop(columns=["churned"]).columns
corr = df[list(num_cols) + ["churned"]].corr()["churned"].drop("churned").sort_values()
fig, a = plt.subplots(figsize=(7, 5))
corr.plot.barh(ax=a, color=np.where(corr > 0, "#E45756", "#4C78A8"))
a.set_title("Correlation of numeric features with churn")
plt.tight_layout()
plt.savefig("outputs/correlation_with_churn.png", dpi=130)
plt.close()

# ------------------------------------------------------------------ 4. SPLIT (before imputation => no leakage)
X = df.drop(columns=["customer_id", "churned"])
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

num_features = X.select_dtypes("number").columns.tolist()
cat_features = X.select_dtypes("object").columns.tolist()

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_features),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced", random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}

# ------------------------------------------------------------------ 5. MODEL COMPARISON (CV on train only)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
results, pipes = {}, {}
for name, clf in models.items():
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    results[name] = {"cv_auc_mean": round(scores.mean(), 4), "cv_auc_std": round(scores.std(), 4)}
    pipes[name] = pipe
    print(f"{name:20s} CV AUC = {scores.mean():.3f} +/- {scores.std():.3f}")
log["cv_results"] = results

best_name = max(results, key=lambda k: results[k]["cv_auc_mean"])
best = pipes[best_name].fit(X_train, y_train)
log["best_model"] = best_name

# ------------------------------------------------------------------ 6. FINAL TEST EVALUATION (touched once)
proba = best.predict_proba(X_test)[:, 1]
# Choose decision threshold on TRAIN CV-free grid using train predictions would overfit;
# use F1-optimal threshold from out-of-fold training predictions instead.
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(pipes[best_name], X_train, y_train, cv=cv, method="predict_proba")[:, 1]
ths = np.linspace(0.05, 0.9, 86)
f1s = [precision_recall_fscore_support(y_train, oof >= t, average="binary", zero_division=0)[2] for t in ths]
threshold = float(ths[int(np.argmax(f1s))])
pred = (proba >= threshold).astype(int)

p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
log["test"] = {"roc_auc": round(roc_auc_score(y_test, proba), 4), "threshold": round(threshold, 2),
               "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
print("\nBest model:", best_name)
print(json.dumps(log["test"], indent=2))
print(classification_report(y_test, pred, target_names=["Stayed", "Churned"]))

fig, a = plt.subplots(1, 2, figsize=(11, 4.5))
RocCurveDisplay.from_predictions(y_test, proba, ax=a[0], name=best_name)
a[0].plot([0, 1], [0, 1], "k--", alpha=.4)
a[0].set_title("ROC curve (test set)")
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stayed", "Churned"], ax=a[1], colorbar=False)
a[1].set_title(f"Confusion matrix (threshold={threshold:.2f})")
plt.tight_layout()
plt.savefig("outputs/model_evaluation.png", dpi=130)
plt.close()

# ------------------------------------------------------------------ 7. FEATURE IMPORTANCE (permutation, on raw columns)
imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=10,
                             random_state=SEED, n_jobs=-1)
imp_s = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
fig, a = plt.subplots(figsize=(7, 5))
imp_s.plot.barh(ax=a, color="#54A24B")
a.set_title("Permutation importance (drop in AUC)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=130)
plt.close()
log["top_features"] = imp_s.sort_values(ascending=False).head(5).round(4).to_dict()

# ------------------------------------------------------------------ 8. SAVE
joblib.dump({"pipeline": best, "threshold": threshold, "features": X.columns.tolist()},
            "models/churn_model.joblib")
with open("outputs/run_log.json", "w") as fh:
    json.dump(log, fh, indent=2, default=str)
print("\nSaved model, charts and run_log.json")
