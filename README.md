# GridGuard — Electricity Theft Detection

GridGuard is a React + Vite frontend with a Python FastAPI backend. The user enters basic consumer details and consumption-profile features, then receives a class `0` or `1` prediction. Model training runs in the backend and is not shown as a user-facing workflow.

## What the user sees
- A single-consumer screening form with name, consumer ID, and daily readings. The backend calculates the 9 model features automatically.
- Prediction class (`0` likely safe / `1` potentially suspicious), model score and explanatory note.
- No training controls or model evaluation tables on the main screen. Model training is handled by the backend.

## Stack
- Frontend: React, Vite, CSS, Recharts, Lucide
- Backend: FastAPI, pandas, NumPy, scikit-learn
- Model: Random Forest classifier with class weighting
- Features: mean, standard deviation, minimum, maximum, median, interquartile range, zero-day ratio, missing-day ratio, consumption trend

## Dataset
A starter training dataset is included at `backend/data/sgcc.csv`. It is based on your `datasetsmall.csv` (25,863 rows, 26 daily reading columns, `CONS_NO`, `FLAG`) so the project has data available on first run.

You also have the larger `data set.csv` (42,372 rows and 1,034 daily reading columns). To train on that full version, copy it to `backend/data/sgcc.csv` and replace the included starter file. The full file is not duplicated inside the ZIP to keep the download manageable.

The app recognizes date-named reading columns, `CONS_NO`, and `FLAG`. `FLAG` is used only for training/evaluation and is never passed as a model input during prediction.

References:
- https://github.com/henryRDlab/ElectricityTheftDetection
- https://github.com/hamsukydev/Electricity_theft_detection

Check the source and confirm the exact meaning of `FLAG` before interpreting predictions. A class-1 output is a risk signal, not proof of theft.

## Saved model artifacts

- `backend/models/electricity_theft_rf.joblib`: fitted preprocessing + Random Forest pipeline with a validation-tuned prediction threshold.
- `backend/models/model_metadata.json`: training row count, class counts, feature names, holdout metrics, confusion matrix, top feature importances, and threshold.
- `backend/app/train.py`: reproducible training/evaluation script.
- `backend/app/ml.py`: daily-reading feature engineering and prediction helpers.

The starter model was trained on the included smaller dataset. Its holdout metrics are recorded in `model_metadata.json`; class 1 is imbalanced, so precision/recall trade-offs are important. Review these metrics before presenting the model as reliable.

## Project structure
```text
electricity-theft-detection/
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       └── styles.css
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── ml.py
│   ├── data/
│   │   └── sgcc.csv          # add dataset here
│   └── requirements.txt
└── README.md
```

## Setup on Windows

### Backend
```powershell
cd backend
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

A trained Random Forest model is included at `backend/models/electricity_theft_rf.joblib`, along with `model_metadata.json`. On startup, the backend loads this saved model instead of retraining on every run. The UI does not show training controls.

To retrain after replacing `backend/data/sgcc.csv`, run this from the `backend` folder:
```powershell
python -m app.train
```
This creates/overwrites `models/electricity_theft_rf.joblib` and `models/model_metadata.json`. Training includes a stratified holdout evaluation and tunes the class-1 probability threshold on a separate validation split. Keep the dataset and saved model together; if you change the dataset, retrain the model.

If you prefer to provide a dataset without saving it in that path, the backend also has a setup endpoint (`POST /api/setup-model`) that accepts an SGCC CSV and prepares the model. This is an API setup route, not a dashboard control.

### Frontend (second terminal)
```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually http://localhost:5173.

## Single-consumer input
The single-consumer form expects these engineered feature values (the starter values are examples, not measurements):
- `consumption_mean`: average daily reading
- `consumption_std`: standard deviation
- `consumption_min`, `consumption_max`: minimum and maximum readings
- `consumption_median`: median reading
- `consumption_iqr`: 75th percentile minus 25th percentile
- `zero_day_ratio`: share of valid days with zero consumption, from 0 to 1
- `missing_day_ratio`: share of days with missing readings, from 0 to 1
- `consumption_trend`: linear trend across readings

Use the same units and feature definitions as the training dataset. The demo form has starter values to show the interaction; replace them with measured features for a real record. Do not treat arbitrary sample inputs as an actual consumer assessment.

## API
- `GET /api/health`: service and model readiness
- `GET /api/model-status`: whether the model is trained
- `POST /api/predict`: predict one engineered feature record (API compatibility)\n- `POST /api/predict-readings`: calculate features and predict from one consumer's daily readings
- `POST /api/setup-model`: load an SGCC CSV and train the model in the backend

## Limitations
- Educational prototype, not a production utility system.
- The label mapping `0 = likely safe`, `1 = potentially suspicious` is provisional until verified against dataset documentation.
- Risk score is the model's estimated class-1 probability, not a guarantee of certainty.
- SGCC data comes from China; results may not generalize to Indian consumers.
- Use held-out evaluation and class-specific metrics before presenting model performance.


## Batch CSV predictions

After the model is ready, the **Batch Screening** panel accepts a CSV with one consumer per row and daily consumption columns (for example, date columns like `1/1/2014`). `CONS_NO` and `FLAG`, if present, are kept as source columns; they are not used as model features. The output download appends:

- `predicted_class`: model class `0` or `1`
- `prediction_label`: human-readable class label
- `risk_score_percent`: model probability for class 1, expressed as a percentage

Use the same SGCC-style column layout and units as the training dataset. Predictions are screening signals only, not proof of theft.
