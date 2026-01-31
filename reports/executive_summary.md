# Executive Summary — Employee Attrition Prediction & Retention Insights

**Business Problem**  
Salifort Motors faces elevated employee turnover, driving up recruiting/training costs and impacting productivity. Leadership needs early warning on at-risk employees and clear drivers to guide retention actions.

**Approach**  
- Analyzed ~15K HR records (satisfaction, workload, tenure, salary, department, promotion history).  
- Built a reproducible ML pipeline (preprocess → train → threshold tuning → evaluate).  
- Compared Logistic Regression, RandomForest, Gradient Boosting; applied class-weighting and decision-threshold tuning.

**Model Performance (test set)**  
- Best-F1 mode (threshold 0.39): Precision 0.984 | Recall 0.965 | F1 0.975 | ROC-AUC 0.992.  
- High-recall mode (threshold 0.09, Precision≥0.70): Precision 0.701 | Recall 0.983 | F1 0.818.  
- Selected model: RandomForest with tuned thresholds.

**Key Drivers**  
- Low satisfaction level combined with high monthly hours and multiple projects.  
- Longer tenure without recent promotion.  
- Lower salary bands; department effects present but smaller.

**Recommendations**  
1) Use the high-recall mode as an early-warning system; route flagged employees to manager review.  
2) Rebalance workload for high-hour / low-satisfaction staff; monitor burnout signals.  
3) Audit promotion pathways for long-tenured employees; increase career progression transparency.  
4) Track fairness: monitor precision/recall by department and salary band; retrain quarterly.  
5) Keep the model advisory only—human-in-the-loop for decisions.

**Artifacts**  
- Metrics: `reports/metrics.json`, Model card: `reports/model_card.md`.  
- Plots: ROC / PR curves in `figures/`.  
- Code & pipeline: GitHub (`employee-attrition-ml-pipeline`).
