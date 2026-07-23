# Customer Churn Prediction

End-to-end machine learning application that predicts whether a customer is likely to leave a subscription service. Uses the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from IBM.

**Stack:** Python 3.14, FastAPI, scikit-learn, XGBoost, Bootstrap 5, Pandas, SHAP

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Web Interface](#web-interface)
- [Feature Engineering](#feature-engineering)
- [Model Explainability](#model-explainability)
- [Testing](#testing)
- [Development Workflow](#development-workflow)
- [Future Improvements](#future-improvements)

---

## Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | IBM Telco Customer Churn (Kaggle) |
| **Rows** | 7,043 customers |
| **Features** | 21 (demographics, account info, services subscribed) |
| **Target** | `Churn` — binary (73% No, 27% Yes — imbalanced) |
| **Missing values** | 11 in `TotalCharges` (filled with 0) |
| **Duplicates** | 0 |

### Raw Features

**Demographics:** `SeniorCitizen`, `Partner`, `Dependents`, `gender`

**Account Info:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

**Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

## Project Structure

```
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│   ├── X_train.parquet                         # Generated — train features
│   ├── X_test.parquet                          # Generated — test features
│   ├── y_train.parquet                         # Generated — train labels
│   └── y_test.parquet                          # Generated — test labels
├── models/
│   ├── churn_model.pkl                         # Trained Random Forest (joblib)
│   ├── feature_cols.pkl                        # 44 feature column names
│   └── scaler.pkl                              # StandardScaler (reference only)
├── notebooks/
│   ├── 01-eda.ipynb                            # Exploratory data analysis
│   ├── 02-preprocessing-feature-engineering    # Preprocessing + 17 engineered features
│   ├── 03-model-training-tuning.ipynb          # Training, tuning, SHAP
│   └── 04-api.ipynb                            # API development notebook
├── static/                                     # Screenshots for README
├── templates/
│   └── index.html                              # Bootstrap 5 web interface
├── run.py                                      # FastAPI application
├── train.py                                    # Training script
├── predict.py                                  # CLI prediction script
├── test_api.py                                 # API unit tests (pytest)
├── test_ui.py                                  # E2E UI test (Playwright)
├── start.bat                                   # Windows launcher
├── requirements.txt                            # pip dependencies
├── pyproject.toml                              # Project metadata
├── uv.lock                                     # uv lockfile
└── .python-version                             # Python version pin
```

---

## Installation

### Prerequisites

- Python 3.14+

### Using pip

```bash
pip install -r requirements.txt
```

### Using uv (recommended)

```bash
uv sync
```

This creates a virtual environment at `.venv/` and installs all dependencies.

---

## Usage

### Start the API server

```bash
python run.py
```

Or double-click `start.bat` on Windows.

The server starts at **http://localhost:8000**.

| Endpoint | Description |
|----------|-------------|
| http://localhost:8000 | Web interface (Bootstrap form) |
| http://localhost:8000/docs | Swagger UI / OpenAPI docs |
| http://localhost:8000/health | Health check |

### CLI prediction

```bash
python predict.py                # Predicts with default sample
python predict.py input.json     # Predicts from a JSON file
```

### Train from scratch

```bash
python train.py
```

This trains Logistic Regression, Random Forest, and XGBoost, performs GridSearchCV tuning on the Random Forest, and saves the best model to `models/churn_model.pkl`.

---

## Model Performance

### All Models Comparison

All models trained on the engineered 44-feature set and evaluated on the 20% held-out test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.8034 | 0.6571 | 0.5443 | 0.5955 | **0.8401** |
| XGBoost | 0.7956 | **0.6617** | **0.5248** | **0.5853** | 0.8303 |
| Random Forest (baseline) | 0.7835 | 0.6449 | 0.4822 | 0.5517 | 0.8213 |
| **Random Forest (tuned)** | **0.7949** | 0.6628 | 0.5085 | 0.5757 | 0.8317 |

**Key takeaways:**
- **Logistic Regression** has the highest **ROC-AUC** (0.8401) and **Accuracy** (80.34%), suggesting the decision boundary is roughly linear
- **XGBoost** has the best **Precision** (66.17%) and **Recall** (52.48%), making it the most balanced classifier
- **Random Forest (tuned)** closes the gap vs LR after tuning, improving ROC-AUC by +0.0104 and Accuracy by +1.14% over the untuned version

### Baseline Comparison (before vs after feature engineering)

| Model | Features | Accuracy | ROC-AUC | Δ ROC-AUC |
|-------|----------|----------|---------|-----------|
| Logistic Regression | 30 (base) | 0.8062 | 0.8422 | — |
| Logistic Regression | 47 (+17 engineered) | 0.8034 | 0.8401 | −0.0021 |
| Random Forest | 30 (base) | 0.7821 | 0.8187 | — |
| Random Forest | 47 (+17 engineered) | 0.7835 | 0.8213 | +0.0026 |

The engineered features had minimal impact on these models since the core signal was already captured. The interaction features (`fiber_no_support`, `month_to_month_fiber`) are most valuable for the tuned RF deployed in production.

### Hyperparameter Tuning (Random Forest)

GridSearchCV with 3-fold CV, scoring = ROC-AUC:

- `n_estimators`: [100, 200]
- `max_depth`: [5, 10, None]
- `min_samples_split`: [2, 5]

| Configuration | CV ROC-AUC |
|---------------|-----------|
| Default RF | 0.8213 |
| **Best: max_depth=10, min_samples_split=5, n_estimators=200** | **0.8327** |
| Improvement | **+0.0114** |

### Final Model: Tuned Random Forest

The tuned Random Forest was chosen as the production model because:
- Strong ROC-AUC (0.8317) — close to best-in-class (LR at 0.8401)
- Robust to outliers and does not require feature scaling
- Handles feature interactions natively (critical for engineered combos like `fiber_no_support`)
- Provides well-calibrated probability estimates via `predict_proba`
- More interpretable than XGBoost for non-technical stakeholders

---

## API Reference

### `POST /predict`

Predicts churn for a single customer.

**Request body:**

```json
{
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.0,
  "TotalCharges": 840.0
}
```

**Response:**

```json
{
  "churn": 1,
  "probability": 0.9576
}
```

| Field | Type | Description |
|-------|------|-------------|
| `churn` | `int` | 0 = will stay, 1 = will churn |
| `probability` | `float` | Probability of churn (0.0–1.0) |

**Error responses:**

| Status | Meaning |
|--------|---------|
| `422` | Validation error — wrong types or missing fields |

### `GET /health`

```json
{"status": "ok"}
```

### `GET /`

Serves the Bootstrap web interface.

### `GET /docs`

Auto-generated Swagger UI (OpenAPI documentation).

---

## Web Interface

A Bootstrap 5 HTML form (`templates/index.html`) collects all 17 customer fields and sends them to `POST /predict` via JavaScript `fetch()`.

**Features:**
- Dropdown selects for categorical fields (Contract, InternetService, PaymentMethod, etc.)
- Numeric inputs for tenure, MonthlyCharges, TotalCharges
- Color-coded result display (red = churn, green = stay)
- Probability shown as percentage

---

## Feature Engineering

17 engineered features were added to improve model performance:

| Feature | Type | Formula / Logic |
|---------|------|-----------------|
| `charge_per_month` | numeric | `TotalCharges / (tenure + 1)` — average billing rate |
| `tenure_x_monthly` | numeric | `tenure * MonthlyCharges` — engagement proxy |
| `monthly_minus_avg` | numeric | `MonthlyCharges - charge_per_month` — deviation from avg |
| `OnlineSecurity_bin` | binary | 1 if subscribed |
| `OnlineBackup_bin` | binary | 1 if subscribed |
| `DeviceProtection_bin` | binary | 1 if subscribed |
| `TechSupport_bin` | binary | 1 if subscribed |
| `StreamingTV_bin` | binary | 1 if subscribed |
| `StreamingMovies_bin` | binary | 1 if subscribed |
| `services_count` | count | Sum of subscribed services (0–6) |
| `no_services` | binary | 1 if no services subscribed |
| `has_streaming` | binary | 1 if TV or Movies streaming active |
| `fiber_no_support` | interaction | Fiber optic × No tech support |
| `fiber_no_security` | interaction | Fiber optic × No online security |
| `month_to_month_fiber` | interaction | Month-to-month × Fiber optic |
| `new_customer` | binary | 1 if tenure ≤ 6 months |
| `long_contract` | ordinal | 0 = Month-to-month, 1 = One year, 2 = Two year |

### Top Predictive Features (SHAP)

1. `charge_per_month` — average monthly rate
2. `monthly_minus_avg` — deviation from expected charge
3. `TotalCharges` — total billed amount
4. `tenure_x_monthly` — engagement proxy
5. `tenure` — customer lifetime

---

## Model Explainability

SHAP (SHapley Additive ExPlanations) is used in `03-model-training-tuning.ipynb`:

- **Global importance:** SHAP summary bar plot showing feature impact across all predictions
- **Local explanation:** SHAP waterfall plot for individual prediction breakdown

This provides transparency into why the model predicts churn for a given customer.

---

## Testing

### API Tests (`test_api.py`)

6 tests using `pytest` + FastAPI `TestClient` — no server needed:

| Test | What it checks |
|------|---------------|
| `test_health` | GET /health returns `{"status": "ok"}` |
| `test_predict_valid` | Valid input returns `{churn, probability}` |
| `test_predict_invalid_type` | Wrong types return 422 |
| `test_predict_missing_field` | Missing fields return 422 |
| `test_feature_vector_length` | Feature vector has exactly 44 columns |
| `test_model_consistency` | Same input always gives same output |

```bash
pytest test_api.py -v
```

### E2E UI Test (`test_ui.py`)

Playwright test that opens a headless browser, fills the form, clicks predict, and verifies the result.

```bash
pytest test_ui.py -v
```

---

## Development Workflow

### Build Phases

| Phase | Description |
|-------|-------------|
| **1. Environment & Data** | Folder structure, EDA, requirements |
| **2. Preprocessing & Feature Engineering** | Data cleaning, encoding, 17 new features |
| **3. Model Training** | 3 models trained, GridSearchCV tuning, SHAP |
| **4. Persistence & API** | Model saved with joblib, FastAPI endpoint |
| **5. Web Interface** | Bootstrap form with live prediction |
| **6. Documentation & Polish** | README, screenshots, tests, report |

### Adding a new feature

1. Edit `notebooks/02-preprocessing-feature-engineering.ipynb` to engineer the feature
2. Add the logic to the `engineer_features()` function in `train.py`
3. Add the same logic to `build_features()` in `predict.py`
4. Add the feature mapping to the feature construction in `run.py` (the `@app.post("/predict")` handler)
5. Retrain: `python train.py`
6. Test: `pytest test_api.py -v`

---

## Future Improvements

- [ ] **Docker** — Containerize with `Dockerfile`
- [ ] **Prediction logging** — Log predictions to SQLite/PostgreSQL for monitoring
- [ ] **LIME explanations** — Add alongside SHAP for comparison
- [ ] **Cloud deployment** — Deploy to AWS/GCP/Azure
- [ ] **CI/CD pipeline** — GitHub Actions for testing and deployment
- [ ] **Retraining pipeline** — Automate retraining on new data
- [ ] **Gender & MultipleLines** — Add back into model (currently only 44 of 47 possible features used)
- [ ] **Class imbalance** — Implement SMOTE or class weighting
- [ ] **Model versioning** — Track model versions with MLflow or DVC

---

## License

MIT
