"""Train the churn prediction model end-to-end."""

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

SEED = 42
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# --- Load ---
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}")

# --- Clean ---
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df.drop("customerID", axis=1, inplace=True)

# --- Encode ---
binary_cols = ["Churn"]
for c in binary_cols:
    df[c] = (df[c] == "Yes").astype(int)

df.drop(["gender", "MultipleLines"], axis=1, inplace=True, errors="ignore")

cat_cols = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

# --- Split ---
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# --- Scale ---
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# --- Feature Engineering ---
def engineer_features(df):
    df = df.copy()
    df["charge_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]
    df["monthly_minus_avg"] = df["MonthlyCharges"] - (df["TotalCharges"] / (df["tenure"] + 1))
    for src, dst in [
        ("OnlineSecurity_Yes", "OnlineSecurity_bin"),
        ("OnlineBackup_Yes", "OnlineBackup_bin"),
        ("DeviceProtection_Yes", "DeviceProtection_bin"),
        ("TechSupport_Yes", "TechSupport_bin"),
        ("StreamingTV_Yes", "StreamingTV_bin"),
        ("StreamingMovies_Yes", "StreamingMovies_bin"),
    ]:
        if src in df.columns:
            df[dst] = df[src].astype(int)
    svc_cols = [c for c in df.columns if c.endswith("_bin")]
    df["services_count"] = df[svc_cols].sum(axis=1)
    df["no_services"] = (df["services_count"] == 0).astype(int)
    has_streaming = pd.Series(0, index=df.index)
    for c in ["StreamingTV_bin", "StreamingMovies_bin"]:
        if c in df.columns:
            has_streaming = has_streaming + df[c]
    df["has_streaming"] = (has_streaming > 0).astype(int)
    is_fiber = df["InternetService_Fiber optic"] if "InternetService_Fiber optic" in df.columns else pd.Series(0, index=df.index)
    no_support = pd.Series(0, index=df.index)
    if "TechSupport_Yes" in df.columns:
        no_support = (df["TechSupport_Yes"] == 0).astype(int)
    no_security = pd.Series(0, index=df.index)
    if "OnlineSecurity_Yes" in df.columns:
        no_security = (df["OnlineSecurity_Yes"] == 0).astype(int)
    is_m2m = pd.Series(1, index=df.index)
    if "Contract_One year" in df.columns:
        is_m2m = is_m2m - df["Contract_One year"]
    if "Contract_Two year" in df.columns:
        is_m2m = is_m2m - df["Contract_Two year"]
    df["fiber_no_support"] = (is_fiber * no_support).astype(int)
    df["fiber_no_security"] = (is_fiber * no_security).astype(int)
    df["month_to_month_fiber"] = (is_m2m * is_fiber).astype(int)
    df["new_customer"] = (df["tenure"] <= 6).astype(int)
    lc = pd.Series(0, index=df.index)
    if "Contract_One year" in df.columns:
        lc = lc + df["Contract_One year"]
    if "Contract_Two year" in df.columns:
        lc = lc + 2 * df["Contract_Two year"]
    df["long_contract"] = lc.astype(int)
    return df

X_train = engineer_features(X_train)
X_test = engineer_features(X_test)
print(f"Features: {X_train.shape[1]} (base + engineered)")

# --- Train models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest": RandomForestClassifier(random_state=SEED),
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    })
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred))

import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=SEED, eval_metric="logloss", use_label_encoder=False)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]
results.append({
    "Model": "XGBoost",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_prob),
})
print(f"\n--- XGBoost ---")
print(classification_report(y_test, y_pred))

print("\n=== Model Comparison ===")
print(pd.DataFrame(results).round(4).to_string(index=False))

# --- Hyperparameter Tuning (Random Forest) ---
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}
rf = RandomForestClassifier(random_state=SEED)
grid = GridSearchCV(rf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print(f"\nBest params: {grid.best_params_}")
print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]
print(f"\n--- Random Forest (tuned) ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# --- Save all models ---
joblib.dump(best_rf, "models/churn_model.pkl")
joblib.dump(models["Logistic Regression"], "models/logistic_regression.pkl")
joblib.dump(models["Random Forest"], "models/random_forest_baseline.pkl")
joblib.dump(xgb_model, "models/xgboost.pkl")
joblib.dump(scaler, "models/scaler.pkl")
feature_cols = X_train.columns.tolist()
joblib.dump(feature_cols, "models/feature_cols.pkl")
print(f"\nSaved models, scaler, and {len(feature_cols)} feature columns to models/")

# Save splits for notebooks
try:
    pd.DataFrame(X_train).to_parquet("data/X_train.parquet")
    pd.DataFrame(X_test).to_parquet("data/X_test.parquet")
    pd.DataFrame(y_train, columns=["Churn"]).to_parquet("data/y_train.parquet")
    pd.DataFrame(y_test, columns=["Churn"]).to_parquet("data/y_test.parquet")
    print("Saved train/test splits to data/")
except ImportError:
    print("Skipped parquet export (pyarrow not installed)")
