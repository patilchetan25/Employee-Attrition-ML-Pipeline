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
```

## 📁 Project Structure

```text
configs/                  YAML configuration files (paths, parameters)
data/                     Raw and processed datasets (gitignored)
figures/                  Evaluation plots (ROC, PR curves)
models/                   Trained models and metrics (gitignored)
notebooks/                Exploratory Data Analysis (EDA)
reports/                  Metrics, model card, executive summary (PDF/MD)
src/                      Pipeline code (preprocessing, training, evaluation, utils)
run_pipeline.py           One-command end-to-end pipeline runner
app.py                    Streamlit web application
.github/workflows/ci.yml  Continuous Integration (CI) workflow
```

---

## 🔍 Key Drivers of Attrition

*(Derived from model feature importance)*

* Low job satisfaction combined with **high monthly working hours** and **multiple active projects**
* **Long tenure without recent promotion**
* **Lower salary bands**
  *(Department-level effects were comparatively smaller)*

---

## 🧠 Recommendations

* Use the **high-recall model mode** as an early-warning system and route flagged employees for **manager review**
* Rebalance workload for employees with **high hours and low satisfaction**; actively monitor burnout indicators
* Audit and improve **promotion pathways** for long-tenured employees to increase transparency
* Track **fairness metrics** (precision and recall) across departments and salary bands; **retrain quarterly**
* Keep the model **advisory only** with a **human-in-the-loop** for all decisions

---

## 📦 Artifacts

* **Metrics:** `metrics.json`
* **Model Card:** `model_card.md`
* **Executive Summary:** `executive_summary.md` (exportable to PDF)
* **Plots:** `roc_curve.png`, `pr_curve.png`

---

## 📜 License

MIT License — see `LICENSE` for details.

---

## 🙌 Credits

* **Dataset:** HR Analytics (~15,000 employees, target variable: *attrition*)
* **Author:** Chetz ([@patilchetan25](https://github.com/patilchetan25))

