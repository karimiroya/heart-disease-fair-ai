"""Heart disease risk explainer (Streamlit app).

Run from the project folder:
    streamlit run app/app.py

This is a learning project, not a medical tool.
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.explain import explain  # noqa: E402

bundle = joblib.load(ROOT / "models" / "model.joblib")

st.title("Heart disease risk explainer")
st.caption("Learning project on the public UCI Heart Disease data. Not a medical tool.")

# ---- patient information (left side) ----
st.sidebar.header("Patient")
age = st.sidebar.slider("Age", 25, 80, 55)
sex = st.sidebar.radio("Sex", ["Woman", "Man"])
cp = st.sidebar.selectbox("Chest pain type", {
    1: "1 – Typical angina",
    2: "2 – Atypical angina",
    3: "3 – Non-anginal pain",
    4: "4 – Asymptomatic",
}.items(), format_func=lambda item: item[1])[0]
trestbps = st.sidebar.slider("Resting blood pressure (mm Hg)", 90, 200, 130)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.sidebar.radio("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.sidebar.selectbox("Resting ECG", {
    0: "0 – Normal",
    1: "1 – ST-T wave abnormality",
    2: "2 – Left ventricular hypertrophy",
}.items(), format_func=lambda item: item[1])[0]
thalach = st.sidebar.slider("Max heart rate during exercise", 60, 210, 150)
exang = st.sidebar.radio("Chest pain during exercise", ["No", "Yes"])
oldpeak = st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)

patient = {
    "age": age,
    "sex": 1 if sex == "Man" else 0,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": 1 if fbs == "Yes" else 0,
    "restecg": restecg,
    "thalach": thalach,
    "exang": 1 if exang == "Yes" else 0,
    "oldpeak": oldpeak,
}

# ---- prediction and explanation ----
risk, table = explain(bundle, patient)

st.metric("Predicted risk of heart disease", f"{risk:.0%}")

st.subheader("Why?")
st.write("Red factors raise the risk, blue factors lower it, compared with an average patient.")

top = table.head(6).iloc[::-1]
colors = ["tab:red" if c > 0 else "tab:blue" for c in top["contribution"]]
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.barh(top["label"], top["contribution"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Contribution to the risk")
fig.tight_layout()
st.pyplot(fig)

up = table[table["contribution"] > 0].head(3)["label"].tolist()
down = table[table["contribution"] < 0].head(3)["label"].tolist()
if up:
    st.write("**Raises the risk most:** " + ", ".join(up))
if down:
    st.write("**Lowers the risk most:** " + ", ".join(down))
