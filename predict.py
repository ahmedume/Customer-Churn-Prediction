"""Predict churn for a single customer from command-line or JSON input."""

import sys
import json
import joblib
import pandas as pd

MODELS = {
    "Random Forest (tuned)": "models/churn_model.pkl",
    "Logistic Regression": "models/logistic_regression.pkl",
    "Random Forest (baseline)": "models/random_forest_baseline.pkl",
    "XGBoost": "models/xgboost.pkl",
}
FEATURES_PATH = "models/feature_cols.pkl"

model = joblib.load(MODELS["Random Forest (tuned)"])
feature_cols = joblib.load(FEATURES_PATH)
col_idx = {c: i for i, c in enumerate(feature_cols)}

BINARY_MAP = {"Yes": 1, "No": 0}
OH_MAP = {
    "InternetService": {"Fiber optic": "InternetService_Fiber optic", "No": "InternetService_No"},
    "OnlineSecurity": {"Yes": "OnlineSecurity_Yes", "No internet service": "OnlineSecurity_No internet service"},
    "OnlineBackup": {"Yes": "OnlineBackup_Yes", "No internet service": "OnlineBackup_No internet service"},
    "DeviceProtection": {"Yes": "DeviceProtection_Yes", "No internet service": "DeviceProtection_No internet service"},
    "TechSupport": {"Yes": "TechSupport_Yes", "No internet service": "TechSupport_No internet service"},
    "StreamingTV": {"Yes": "StreamingTV_Yes", "No internet service": "StreamingTV_No internet service"},
    "StreamingMovies": {"Yes": "StreamingMovies_Yes", "No internet service": "StreamingMovies_No internet service"},
    "Contract": {"One year": "Contract_One year", "Two year": "Contract_Two year"},
    "PaymentMethod": {
        "Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
        "Electronic check": "PaymentMethod_Electronic check",
        "Mailed check": "PaymentMethod_Mailed check",
    },
}
CONTRACT_ORD = {"Month-to-month": 0, "One year": 1, "Two year": 2}


def build_features(data):
    row = [0.0] * len(feature_cols)
    row[col_idx["SeniorCitizen"]] = float(data["SeniorCitizen"])
    row[col_idx["tenure"]] = data["tenure"]
    row[col_idx["MonthlyCharges"]] = data["MonthlyCharges"]
    row[col_idx["TotalCharges"]] = data["TotalCharges"]
    row[col_idx["charge_per_month"]] = data["TotalCharges"] / (data["tenure"] + 1)
    row[col_idx["tenure_x_monthly"]] = data["tenure"] * data["MonthlyCharges"]
    row[col_idx["monthly_minus_avg"]] = data["MonthlyCharges"] - (data["TotalCharges"] / (data["tenure"] + 1))
    row[col_idx["Partner_Yes"]] = float(BINARY_MAP[data["Partner"]])
    row[col_idx["Dependents_Yes"]] = float(BINARY_MAP[data["Dependents"]])
    row[col_idx["PhoneService_Yes"]] = float(BINARY_MAP[data["PhoneService"]])
    row[col_idx["PaperlessBilling_Yes"]] = float(BINARY_MAP[data["PaperlessBilling"]])
    for svc, col in [
        ("OnlineSecurity", "OnlineSecurity_bin"),
        ("OnlineBackup", "OnlineBackup_bin"),
        ("DeviceProtection", "DeviceProtection_bin"),
        ("TechSupport", "TechSupport_bin"),
        ("StreamingTV", "StreamingTV_bin"),
        ("StreamingMovies", "StreamingMovies_bin"),
    ]:
        row[col_idx[col]] = 1.0 if data[svc] == "Yes" else 0.0
    svc_cols = ["OnlineSecurity_bin", "OnlineBackup_bin", "DeviceProtection_bin",
                "TechSupport_bin", "StreamingTV_bin", "StreamingMovies_bin"]
    svc_count = sum(row[col_idx[c]] for c in svc_cols)
    row[col_idx["services_count"]] = svc_count
    row[col_idx["no_services"]] = 1.0 if svc_count == 0 else 0.0
    row[col_idx["has_streaming"]] = 1.0 if (row[col_idx["StreamingTV_bin"]] + row[col_idx["StreamingMovies_bin"]]) > 0 else 0.0
    is_fiber = 1.0 if data["InternetService"] == "Fiber optic" else 0.0
    no_support = 1.0 if data["TechSupport"] == "No" else 0.0
    no_security = 1.0 if data["OnlineSecurity"] == "No" else 0.0
    is_m2m = 1.0 if data["Contract"] == "Month-to-month" else 0.0
    row[col_idx["fiber_no_support"]] = is_fiber * no_support
    row[col_idx["fiber_no_security"]] = is_fiber * no_security
    row[col_idx["month_to_month_fiber"]] = is_m2m * is_fiber
    row[col_idx["new_customer"]] = 1.0 if data["tenure"] <= 6 else 0.0
    row[col_idx["long_contract"]] = float(CONTRACT_ORD.get(data["Contract"], 0))
    for raw_col, mapping in OH_MAP.items():
        val = data.get(raw_col)
        if val and val in mapping:
            row[col_idx[mapping[val]]] = 1.0
    return pd.DataFrame([row], columns=feature_cols)


def predict(data, model_name="Random Forest (tuned)"):
    row_df = build_features(data)
    m = joblib.load(MODELS.get(model_name, MODELS["Random Forest (tuned)"]))
    prob = float(m.predict_proba(row_df)[0, 1])
    return {"churn": int(prob >= 0.5), "probability": round(prob, 4)}


if __name__ == "__main__":
    model_name = "Random Forest (tuned)"
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in MODELS:
            model_name = arg
        else:
            with open(arg) as f:
                sample = json.load(f)
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    if "sample" not in locals():
        sample = {
            "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 12, "PhoneService": "Yes",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0, "TotalCharges": 840.0,
        }
    result = predict(sample, model_name)
    print(f"Model: {model_name}")
    print(json.dumps(result, indent=2))
