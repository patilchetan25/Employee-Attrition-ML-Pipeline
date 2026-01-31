import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.title("Employee Attrition Risk")
st.write("Predict attrition risk and view top drivers.")

MODEL_PATH = Path("models/model.joblib")
PREP_PATH = Path("data/processed/preprocessor.joblib")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREP_PATH)
    feature_names = preprocessor.get_feature_names_out()
    return model, preprocessor, feature_names

def get_importances(model, feature_names, top_k=10):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return pd.Series(imp, index=feature_names).sort_values(ascending=False).head(top_k)
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
        return pd.Series(imp, index=feature_names).sort_values(ascending=False).head(top_k)
    else:
        return None

model, preprocessor, feature_names = load_artifacts()

st.subheader("Enter employee features")
satisfaction = st.slider("satisfaction_level", 0.0, 1.0, 0.5, 0.01)
last_eval = st.slider("last_evaluation", 0.0, 1.0, 0.7, 0.01)
number_project = st.slider("number_project", 1, 10, 4, 1)
avg_hours = st.slider("average_montly_hours", 80, 320, 200, 1)
tenure = st.slider("time_spend_company", 1, 15, 5, 1)
work_accident = st.selectbox("Work_accident", [0, 1])
promotion = st.selectbox("promotion_last_5years", [0, 1])
department = st.selectbox("Department", [
    "sales","accounting","hr","technical","support","management",
    "IT","product_mng","marketing","RandD"
])
salary = st.selectbox("salary", ["low","medium","high"])

row = pd.DataFrame([{
    "satisfaction_level": satisfaction,
    "last_evaluation": last_eval,
    "number_project": number_project,
    "average_montly_hours": avg_hours,
    "time_spend_company": tenure,
    "Work_accident": work_accident,
    "promotion_last_5years": promotion,
    "Department": department,
    "salary": salary,
}])

if st.button("Predict"):
    X = preprocessor.transform(row)
    proba = model.predict_proba(X)[:, 1][0]
    st.metric("Attrition risk (probability)", f"{proba:.3f}")
    st.progress(proba)

    imp = get_importances(model, feature_names, top_k=10)
    if imp is not None:
        st.subheader("Top drivers")
        st.dataframe(imp.rename("importance"))
    else:
        st.info("Feature importances not available for this model.")
