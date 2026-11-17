"""Turn a model prediction into a simple explanation.

For a logistic regression, each feature adds a fixed amount to the
prediction (on the log-odds scale):

    contribution = coefficient × (patient value − average value)

Positive contribution = pushes the risk UP.
Negative contribution = pushes the risk DOWN.
For a linear model this is exactly what SHAP computes.
"""
import numpy as np
import pandas as pd

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak"]
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang"]

LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate (exercise)",
    "exang": "Chest pain during exercise",
    "oldpeak": "ST depression (oldpeak)",
}


def explain(bundle, patient):
    """patient: a dict with the 10 features. Returns (risk, table of contributions)."""
    model = bundle["model"]
    row = pd.DataFrame([patient])[FEATURES]

    # 1) predicted risk (probability of heart disease)
    risk = model.predict_proba(row)[0, 1]

    # 2) contribution of each transformed column
    x = model[:-1].transform(row)[0]                 # the patient after preprocessing
    coef = model[-1].coef_[0]                        # one number per column
    contrib = coef * (x - bundle["means"])           # how much each column moves the risk

    # 3) add up the one-hot columns that belong to the same original feature
    names = bundle["column_names"]                   # e.g. "categorical__cp_4.0"
    original = [n.split("__")[1].rsplit("_", 1)[0] if n.startswith("categorical") else n.split("__")[1]
                for n in names]
    table = (pd.DataFrame({"feature": original, "contribution": contrib})
             .groupby("feature", as_index=False)["contribution"].sum())
    table["label"] = table["feature"].map(LABELS)
    table = table.sort_values("contribution", key=np.abs, ascending=False)
    return risk, table
