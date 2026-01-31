# Employee Attrition Prediction & Retention Insights (ML Pipeline)
[![CI](https://github.com/patilchetan25/Employee-Attrition-ML-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/patilchetan25/Employee-Attrition-ML-Pipeline/actions/workflows/ci.yml)

End-to-end ML pipeline to predict employee attrition and surface actionable retention insights using Python and scikit-learn.

## Results (test set)
| Mode                    | Threshold | Accuracy | Precision | Recall | F1    | ROC-AUC | Confusion Matrix (TN, FP, FN, TP) |
|-------------------------|-----------|----------|-----------|--------|-------|---------|-----------------------------------|
| Best-F1 (selected)      | 0.39      | 0.988    | 0.984     | 0.965  | 0.975 | 0.992   | (2275, 11, 25, 689)               |
| High-recall (prec ≥0.70)| 0.09      | 0.896    | 0.701     | 0.983  | 0.818 | 0.992   | (1986, 300, 12, 702)              |

Plots: `figures/roc_curve.png`, `figures/pr_curve.png`  
Metrics: `models/metrics.json`, `models/metrics_by_model.json`, `reports/metrics.json`

## Demo
- Streamlit app: https://employee-attrition-ml-pipeline.streamlit.app
- Run locally: `streamlit run app.py`

## How to Run Locally
```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py           # preprocess + train + evaluate
# Optional: streamlit run app.py

## Project Structure

configs/          YAML config (paths/params)
data/             raw/processed (gitignored)
figures/          evaluation plots
models/           saved model + metrics (gitignored)
notebooks/        EDA
reports/          metrics, model_card.md, executive_summary.md/pdf
src/              pipeline code (prep, train, eval, utils)
run_pipeline.py   one-command runner
app.py            Streamlit app
.github/workflows/ci.yml  CI workflow

## Key Drivers (from model importance)
Low satisfaction + high monthly hours and multiple projects
Longer tenure without recent promotion
Lower salary bands; department effects are smaller

# Recommendations
Use high-recall mode as an early-warning system; route flagged employees to manager review.
Rebalance workload for high-hour / low-satisfaction staff; monitor burnout signals.
Audit promotion pathways for long-tenured employees; increase transparency.
Track fairness: monitor precision/recall by department and salary band; retrain quarterly.
Keep the model advisory only—human-in-the-loop for decisions.

Artifacts

Metrics: metrics.json
Model card: model_card.md
Executive summary: executive_summary.md (export to PDF for sharing)
Plots: roc_curve.png, pr_curve.png

License
MIT License (see LICENSE).

Credits
Dataset: HR Analytics (~15K employees, target left).
Author: Chetz (patilchetan25)
