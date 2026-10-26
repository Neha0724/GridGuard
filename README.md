# GridGuard ⚡
### Electricity Theft Detection Using Machine Learning

GridGuard is a web application that screens electricity-consumption patterns to identify consumers whose usage may be suspicious. It combines a React + Vite frontend with a FastAPI backend and a machine-learning pipeline developed around the SGCC electricity-consumption dataset.

> **Important:** GridGuard is a screening and decision-support prototype. A prediction is not proof of electricity theft and should not be used as the sole basis for enforcement or penalties. Results require review alongside meter checks, field evidence, and domain expertise.

## Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Using the Application](#using-the-application)
- [API Overview](#api-overview)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

## Features
- **Dashboard:** View dataset and model-status information exposed by the backend.
- **Single-consumer prediction:** Submit a consumer's daily readings and receive a model prediction.
- **Batch CSV screening:** Upload a CSV containing multiple consumers and review/download predictions where supported by the UI.
- **Feature engineering:** Derive summary features such as mean, standard deviation, minimum, maximum, median, interquartile range, zero-day ratio, missing-day ratio, and consumption trend.
- **Risk score:** Display the model's estimated score when returned by the API.
- **Model persistence:** Save the trained model and metadata so they can be loaded on later starts.

Feature availability and labels depend on the frontend/backend versions in your checkout.

## How It Works
```text
Consumer readings / CSV
          |
          v
React + Vite frontend
          | HTTP requests
          v
FastAPI backend
          |
          v
Input validation + feature engineering
          |
          v
Trained classification model
          |
          v
Prediction + available risk score
          |
          v
Results displayed in the frontend
```

The dataset's `FLAG` column is the target label: `0` represents a normal record and `1` represents a flagged record. A flagged output is an indication for further review, not a definitive finding.

## Technology Stack
| Layer | Technology |
|---|---|
| Frontend | React, Vite, CSS |
| Backend API | Python, FastAPI |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn and libraries listed in `backend/requirements.txt` |
| Model persistence | Joblib |
| Dataset | CSV |
| API documentation | FastAPI Swagger UI / OpenAPI |

Use the repository's `package.json` and `backend/requirements.txt` as the source of truth for exact dependency versions.

## Project Structure
The ML backend is expected to follow this structure; retain any additional files in your repository.

```text
GridGuard/
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.*
└── backend/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py
    │   ├── ml.py
    │   └── train.py
    ├── data/
    │   └── sgcc.csv
    ├── models/
    │   ├── electricity_theft_rf.joblib
    │   └── model_metadata.json
    └── requirements.txt
```

Model files may be generated automatically when the API starts if they are missing and the training dataset is present. Avoid committing large generated files or datasets unless licensing and repository-size requirements allow it.

## Dataset
GridGuard was developed around the **SGCC electricity-consumption dataset**.
- **Target column:** `FLAG`
- **Label `0`:** Normal
- **Label `1`:** Flagged / potentially suspicious
- **Input:** Daily electricity-consumption readings. The current UI flow is designed around 26 readings for one consumer or corresponding reading columns per row in a batch CSV.
- **Preprocessing:** The backend derives summary features from readings before prediction.

The default training-data path in the current backend is `backend/data/sgcc.csv`. Dataset variants can differ in rows, columns, reading windows, missing values, and class balance. Confirm the actual file and schema before training. Check the dataset source, licence, and permitted use before redistributing it; do not publish customer-identifying data.

## Getting Started
### Prerequisites
- Python version compatible with packages in `backend/requirements.txt` (Python 3.10+ is a reasonable starting point; use the version your dependencies support).
- Node.js and npm.
- Git, if cloning the repository.
- Dataset at `backend/data/sgcc.csv`, unless your configuration points elsewhere.

### 1. Clone the repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```
Replace the placeholders with your repository URL and folder name.

### 2. Install and start the backend
From the project root:
```bash
cd backend
python -m venv .venv
```
Activate on Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
Or on Windows Command Prompt:
```bat
.venv\Scripts\activate.bat
```
On macOS/Linux:
```bash
source .venv/bin/activate
```
Install dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
From the `backend/` directory, start the API:
```bash
uvicorn app.main:app --reload
```
The API should be available at `http://127.0.0.1:8000`.
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

The backend loads saved model/metadata files when available. If they are missing, it attempts to train using the configured dataset path. Check the terminal output if startup fails.

### 3. Install and start the frontend
Open a **second terminal** at the project root:
```bash
cd frontend
npm install
npm run dev
```
Open the local URL printed by Vite (commonly `http://localhost:5173`). If the frontend cannot reach the API, check its API base URL/configuration. The current FastAPI CORS configuration allows `http://localhost:5173` and `http://127.0.0.1:5173`.

### 4. Optional: train the model manually
If `backend/app/train.py` exists and you want to retrain with the configured dataset, run this from `backend/` with the virtual environment activated:
```bash
python -m app.train
```
Retraining may overwrite saved model and metadata files. Back them up first if needed.

## Using the Application
### Single-consumer prediction
1. Open the frontend.
2. Navigate to the single prediction screen.
3. Enter a consumer identifier if requested.
4. Provide the required number of daily readings (26 in the current UI flow).
5. Submit the form and review the predicted class, label, and risk score if returned.

### Batch CSV screening
1. Prepare a CSV with one consumer per row.
2. Include the expected daily-reading columns, following the schema accepted by the current backend.
3. Upload the CSV on the batch-screening screen.
4. Review predictions and download the output CSV if that action is available.

The input CSV generally does not need a `FLAG` column for inference. Column names, reading count, and order must match the backend's expected schema.

## API Overview
The backend is a FastAPI application. Use `/docs` to inspect the exact routes, request schemas, and response fields for your checked-out version. The current implementation includes functionality for loading/training the model, single-record prediction, and batch CSV prediction. Use the live Swagger documentation as the authoritative reference if the code changes.

## Model Training and Evaluation
The backend ML module handles dataset loading, feature engineering, model training/evaluation, and prediction. Model and metadata files are stored under `backend/models/`.

Because electricity-theft datasets may be imbalanced, do not rely on accuracy alone. Report precision, recall, F1-score for the flagged class, a confusion matrix, and the decision threshold if probabilities are thresholded. Use a held-out test set and avoid leakage between training and test data. Offline results do not by themselves establish real-world performance.

## Testing
These are **smoke tests** to verify that the application accepts readings and returns a response. The predicted class is model-dependent and is not guaranteed.

### Test 1 — Relatively stable consumption
Enter these 26 readings:
```text
10, 11, 10, 12, 11, 10, 11, 12, 10, 11, 10, 12, 11, 10, 11, 12, 10, 11, 10, 12, 11, 10, 11, 12, 10, 11
```
**Verify:** The request completes, a prediction appears, and no frontend/API error is shown.

### Test 2 — Sudden consumption drop
Enter these 26 readings:
```text
20, 21, 19, 22, 20, 21, 5, 4, 3, 2, 4, 3, 2, 5, 3, 2, 4, 3, 2, 4, 3, 2, 5, 3, 2, 3
```
**Verify:** The request completes and a prediction appears. Compare its score with Test 1, but do not assume this case must be classified as theft. Also test invalid input (such as fewer than 26 readings) and confirm the app shows a validation error instead of crashing.

## Troubleshooting
### `ModuleNotFoundError: No module named 'app'`
Run backend commands from `backend/` and confirm `backend/app/__init__.py` exists.

### `uvicorn` is not recognized
Activate the backend virtual environment and install dependencies:
```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### Training dataset not found
Confirm the dataset exists at `backend/data/sgcc.csv`. If using another file, update the configured path or follow the dataset-loading logic in `backend/app/main.py`.

### Model fails to load
Check that the saved model was generated with compatible Python and scikit-learn versions. If retraining is appropriate, back up the model and metadata first.

### Frontend shows a network or CORS error
Ensure the API is running on port `8000`, inspect the browser Network tab, and verify the frontend API URL and allowed CORS origin.

### Predictions look inaccurate or nearly identical
Check the input schema, preprocessing consistency, class imbalance, model threshold, and evaluation metrics. A successful response does not guarantee a strong model.

## Limitations
- A flagged prediction does not establish that theft occurred.
- Dataset patterns may not generalize to another region, meter type, season, or population.
- Class imbalance can make accuracy misleading.
- Results depend on the training data, preprocessing, threshold, and input schema.
- This is a prototype and requires further validation, monitoring, and security review before operational use.

## Future Improvements
- Evaluate with stratified cross-validation and a clearly separated test set.
- Compare precision, recall, F1-score, PR-AUC, and confusion matrices.
- Calibrate risk scores and select thresholds based on false-positive and false-negative costs.
- Improve CSV schema validation and error reporting.
- Add authentication, access controls, audit logs, and privacy safeguards before deployment.
- Monitor data drift and periodically re-evaluate the model.

## Contributing
1. Create a feature branch.
2. Make focused changes and test both frontend and backend.
3. Do not commit secrets, customer data, virtual environments, or unnecessary generated files.
4. Open a pull request describing the changes and tests performed.

## Disclaimer
GridGuard is an educational / decision-support prototype. It must not be treated as a standalone authority for accusing a consumer of electricity theft or taking enforcement action.
