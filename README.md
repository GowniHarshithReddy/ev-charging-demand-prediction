# EV Charging Demand Prediction and Usage Pattern Analysis

Machine learning project that analyses EV charging sessions, predicts
charging demand (regression), classifies sessions as High or Normal demand
(classification), explains predictions with SHAP and LIME, and presents
everything in a Streamlit dashboard.

## Dataset
Publicly available EV charging transactions (about 72,000 sessions,
Sept 2021 to Sept 2022). Place `ChargingRecords.csv` in the `data/` folder.

## Project structure
- `data/`: raw, cleaned and feature-engineered CSV files
- `notebooks/`: data understanding, EDA, feature engineering and modelling
- `models/`: saved models (`.pkl`)
- `app/dashboard.py`: Streamlit dashboard
- `requirements.txt`: dependencies

## Setup
1. Install Python 3.13.
2. Create and activate a virtual environment:
   - `python -m venv venv`
   - `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`

## Run the dashboard
From the project root folder:

    streamlit run app/dashboard.py

## Method summary
- **Cleaning:** removed 13 duplicates and 735 rows with invalid duration.
- **Leakage control:** Duration and end-time columns are excluded from models
  because they are only known after a session ends. Time-based train/test
  split (first 80% train, last 20% test).
- **Guest handling:** UserID 0 (43% of sessions) is a catch-all guest ID, so
  history features are set to missing for it and an `IsGuest` flag is used.
- **Regression:** best RMSE about 11.4, R2 about 0.33 (LightGBM, XGBoost).
- **Classification:** High Demand = demand at or above the training median
  (14.36). Best ROC-AUC about 0.79. Classes are balanced; resampling
  (Random Oversampling, SMOTE, ADASYN) was demonstrated on an artificially
  imbalanced training set.
- **Explainability:** feature importance, SHAP and LIME.
- **Forecasting:** simple daily baselines (4-week same-weekday average and
  previous-day).

## Limitations
- Models use only information available before a session starts, so R2 is modest.
- Dashboard prediction form uses typical placeholder values for user and charger history.