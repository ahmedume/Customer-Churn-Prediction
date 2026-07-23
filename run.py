import joblib, uvicorn, warnings, pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

warnings.filterwarnings("ignore")

models = {
    "Random Forest (tuned)": joblib.load("models/churn_model.pkl"),
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Random Forest (baseline)": joblib.load("models/random_forest_baseline.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl"),
}
feature_cols = joblib.load("models/feature_cols.pkl")
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
    "PaymentMethod": {"Credit card (automatic)": "PaymentMethod_Credit card (automatic)", "Electronic check": "PaymentMethod_Electronic check", "Mailed check": "PaymentMethod_Mailed check"},
}
CONTRACT_ORD = {"Month-to-month": 0, "One year": 1, "Two year": 2}

app = FastAPI(title="Churn Prediction API", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_NAMES = list(models.keys())

class CustomerInput(BaseModel):
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    model_name: str = Field(default="Random Forest (tuned)", description=f"Model to use: {MODEL_NAMES}")

class PredictionOut(BaseModel):
    churn: int
    probability: float

@app.post("/predict", response_model=PredictionOut)
def predict(data: CustomerInput):
    row = [0.0] * len(feature_cols)

    # Numerical features
    row[col_idx["SeniorCitizen"]] = float(data.SeniorCitizen)
    row[col_idx["tenure"]] = data.tenure
    row[col_idx["MonthlyCharges"]] = data.MonthlyCharges
    row[col_idx["TotalCharges"]] = data.TotalCharges

    # Engineered numerical features
    row[col_idx["charge_per_month"]] = data.TotalCharges / (data.tenure + 1)
    row[col_idx["tenure_x_monthly"]] = data.tenure * data.MonthlyCharges
    row[col_idx["monthly_minus_avg"]] = data.MonthlyCharges - (data.TotalCharges / (data.tenure + 1))

    # Binary features
    row[col_idx["Partner_Yes"]] = float(BINARY_MAP[data.Partner])
    row[col_idx["Dependents_Yes"]] = float(BINARY_MAP[data.Dependents])
    row[col_idx["PhoneService_Yes"]] = float(BINARY_MAP[data.PhoneService])
    row[col_idx["PaperlessBilling_Yes"]] = float(BINARY_MAP[data.PaperlessBilling])

    # Service binary flags
    for svc, col in [("OnlineSecurity", "OnlineSecurity_bin"), ("OnlineBackup", "OnlineBackup_bin"),
                     ("DeviceProtection", "DeviceProtection_bin"), ("TechSupport", "TechSupport_bin"),
                     ("StreamingTV", "StreamingTV_bin"), ("StreamingMovies", "StreamingMovies_bin")]:
        val = getattr(data, svc)
        row[col_idx[col]] = 1.0 if val == "Yes" else 0.0

    # Service count
    svc_count = sum(row[col_idx[c]] for c in ["OnlineSecurity_bin", "OnlineBackup_bin", "DeviceProtection_bin", "TechSupport_bin", "StreamingTV_bin", "StreamingMovies_bin"])
    row[col_idx["services_count"]] = svc_count
    row[col_idx["no_services"]] = 1.0 if svc_count == 0 else 0.0
    row[col_idx["has_streaming"]] = 1.0 if (row[col_idx["StreamingTV_bin"]] + row[col_idx["StreamingMovies_bin"]]) > 0 else 0.0

    # High-risk combos
    is_fiber = 1.0 if data.InternetService == "Fiber optic" else 0.0
    no_support = 1.0 if data.TechSupport == "No" else 0.0
    no_security = 1.0 if data.OnlineSecurity == "No" else 0.0
    is_m2m = 1.0 if data.Contract == "Month-to-month" else 0.0
    row[col_idx["fiber_no_support"]] = is_fiber * no_support
    row[col_idx["fiber_no_security"]] = is_fiber * no_security
    row[col_idx["month_to_month_fiber"]] = is_m2m * is_fiber
    row[col_idx["new_customer"]] = 1.0 if data.tenure <= 6 else 0.0
    row[col_idx["long_contract"]] = float(CONTRACT_ORD.get(data.Contract, 0))

    # One-hot encoded categoricals
    for raw_col, mapping in OH_MAP.items():
        val = getattr(data, raw_col)
        if val in mapping:
            row[col_idx[mapping[val]]] = 1.0

    row_df = pd.DataFrame([row], columns=feature_cols)
    model = models.get(data.model_name, models["Random Forest (tuned)"])
    prob = float(model.predict_proba(row_df)[0, 1])
    return PredictionOut(churn=int(prob >= 0.5), probability=round(prob, 4))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def index():
    return FileResponse("templates/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
