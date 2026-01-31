# Employee Attrition Prediction & Retention Insights (ML Pipeline)

End-to-end ML pipeline to predict employee attrition and surface actionable retention insights. Includes preprocessing, model training with class-weighting and threshold tuning, evaluation, and plots. Built with Python/scikit-learn.


## Results (threshold tuning, best model = RandomForest)
- Best-F1 threshold 0.39:
  - Accuracy 0.988 | Precision 0.984 | Recall 0.965 | F1 0.975 | ROC-AUC 0.992
  - Confusion matrix: TN 2275, FP 11, FN 25, TP 689
- Recall@Precision≥0.70 threshold 0.09:
  - Accuracy 0.896 | Precision 0.701 | Recall 0.983 | F1 0.818
  - Confusion matrix: TN 1986, FP 300, FN 12, TP 702
- Plots: figures/roc_curve.png, figures/pr_curve.png


## How to run
python3 -m venv .venv
source .venv/bin/activate           # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python run_pipeline.py

## Project structure
configs/          # YAML config for paths/params
data/             # raw/processed (gitignored)
figures/          # evaluation plots
models/           # saved model + metrics (gitignored)
notebooks/        # EDA
reports/          # metrics + executive summary/model card
src/              # pipeline code (prep, train, eval, utils)
run_pipeline.py   # one-command runner

## Recommendations

Use the high-recall threshold (0.09) as an early-warning mode; follow with human review.

Balance workload for high-hour / low-satisfaction staff; monitor long-tenured without promotion.

Retrain quarterly; track precision/recall by department and salary band to avoid drift/bias.

Keep model advisory only; pair with HR policy and manager check-ins.