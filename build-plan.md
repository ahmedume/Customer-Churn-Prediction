# Customer Churn Prediction — Build Plan

## Divide & Conquer Strategy

### Phase 1: Environment & Data
- [x] Set up folder structure (`data/`, `notebooks/`, `models/`, `api/`, `templates/`, `static/`, `utils/`)
- [x] Move dataset into `data/`
- [x] Initial EDA in notebook: shape, dtypes, missing values, duplicates, target distribution
- [ ] Create `requirements.txt` — skip until deployment (YAGNI)

### Phase 2: Preprocessing & Feature Engineering
- [x] Handle missing values, remove duplicates, encode categoricals, scale numerics
- [x] Train/test split
- [x] Feature importance (Random Forest baseline)
- [x] Compare model performance before/after feature engineering

### Phase 3: Model Training
- [x] Train baseline: Logistic Regression, Random Forest, XGBoost
- [x] Hyperparameter tuning via GridSearchCV/RandomizedSearchCV
- [x] Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC
- [x] Explainability with SHAP

### Phase 4: Persistence & API
- [x] Save best model with Joblib
- [x] FastAPI `POST /predict` endpoint
- [x] Load model without retraining
- [x] Input validation with Pydantic

### Phase 5: Web Interface
- [x] Bootstrap HTML form for customer input
- [x] Display prediction + probability
- [x] Connect to API

### Phase 6: Documentation & Polish
- [ ] Installation steps, dataset info, training procedure
- [ ] API docs (FastAPI auto-docs)
- [ ] README.md with screenshots

### Bonus (if time permits)
- [ ] Dockerfile
- [ ] Unit tests
- [ ] SQLite prediction logging
