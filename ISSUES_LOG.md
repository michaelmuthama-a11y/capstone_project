# Issues Log: Customer Churn Capstone

Every problem that actually occurred while building this project, how it was handled, and what is still unresolved.

| # | Stage | Issue | Impact | Resolution | Status |
|---|-------|-------|--------|------------|--------|
| 1 | Setup | Project folder was first created outside the working directory (`/root`) | Files would have been unreachable for delivery | Removed it and recreated everything under the working directory | Fixed |
| 2 | Data | The sandbox blocks downloads from public dataset sites (HTTP 403 from the network proxy), so no real dataset could be fetched | No real customer data | Generated a **synthetic** dataset with a hidden churn formula and injected realistic data-quality problems | **Open limitation** |
| 3 | Data | First synthetic dataset had a 47.8% churn rate, unrealistic for a mobile-money service | Distorted class balance and results | Lowered the intercept of the churn formula; the churn rate is now 20.7% | Fixed |
| 4 | Data | My calibration loop (trying intercepts -2.8, -3.2, -3.5) overwrote the script and left the last value (13% churn) instead of the one I wanted | Wrong dataset would have been used silently | Noticed from the output, set -2.8 explicitly and regenerated | Fixed |
| 5 | Cleaning | Data contained duplicate rows, missing values, impossible values (age -5 to 200, negative amounts) and inconsistent labels (`NAIROBI`, ` nairobi `, `Y`/`yes`) | Would bias models | Dropped duplicates, standardised labels, converted impossible values to missing and imputed inside the pipeline | Fixed |
| 6 | Modelling | Class imbalance (about 1 churner in 5) | Accuracy alone would be misleading | Used ROC-AUC, precision, recall and F1; used `class_weight="balanced"`; picked the decision threshold from out-of-fold training predictions | Fixed |
| 7 | Modelling | Risk of data leakage if imputing or scaling before the split | Inflated test scores | Split first and put all preprocessing inside the scikit-learn Pipeline, so it is fitted on training folds only | Prevented |
| 8 | Results | Logistic Regression beat Random Forest and Gradient Boosting | Looks like simple beats complex | Expected, because the synthetic data was generated from a logistic formula. **This result may not hold on real data** | **Open limitation** |
| 9 | Results | Precision at the chosen threshold is only about 0.42 (recall about 0.70) | Many false alarms in retention campaigns | Accepted as the trade-off; threshold can be raised if retention calls are expensive | Documented |
| 10 | Visuals | Stray `churned` axis label on the class-balance chart | Cosmetic | Removed the label | Fixed |

## What I would do with more time or real data
- Replace the synthetic data with real or anonymised operator data and re-run the pipeline unchanged.
- Tune hyperparameters and compare with calibrated probabilities.
- Add a cost-based threshold (cost of a retention offer vs. value of a retained customer).
- Add a simple dashboard or API for scoring.
