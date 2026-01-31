# Model Card — Employee Attrition Prediction

**Intended use**  
Early-warning tool to flag employees at higher risk of leaving, so HR/managers can intervene. Advisory only; not for termination or punitive action.

**Data**  
- Source: HR Analytics dataset (~15K employees, 10 features).  
- Target: `left` (1 = departed).  
- Key features: satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, promotion_last_5years, Work_accident, Department, salary.

**Preprocessing**  
- Train/validation/test split with stratification.  
- Imputation (median for numeric, most_frequent for categorical).  
- One-hot encoding for categorical features.

**Models compared**  
- Logistic Regression (balanced)  
- RandomForest (balanced_subsample)  
- Gradient Boosting  
- Threshold sweep for best F1 and recall@precision ≥ 0.70.

**Selected model**  
- RandomForest with tuned threshold 0.39 (best-F1).

**Performance (test set)**  
- Best-F1 threshold 0.39: Accuracy 0.988 | Precision 0.984 | Recall 0.965 | F1 0.975 | ROC-AUC 0.992  
- Recall@Precision≥0.70 threshold 0.09: Accuracy 0.896 | Precision 0.701 | Recall 0.983 | F1 0.818

**Fairness & monitoring**  
- Monitor precision/recall by department and salary band.  
- Retrain quarterly; recalibrate thresholds if class balance shifts.  
- Keep human-in-the-loop for decisions; avoid punitive use.

**Limitations**  
- Historical data may embed bias; model captures correlation, not causation.  
- Does not account for recent policy changes or external labor market factors.

**Versioning**  
- Code: current main branch.  
- Artifacts: `models/model.joblib`, `models/metrics.json`, `reports/metrics.json`, plots in `figures/`.
