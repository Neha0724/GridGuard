from pathlib import Path
import json
import joblib
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ml import engineer_features, features_from_readings, read_dataset, train_and_evaluate, predict_one, predict_csv_rows

app = FastAPI(
    title="GridGuard API",
    version="1.1.0",
    description="GridGuard electricity theft screening with automatic SGCC training and batch CSV predictions."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sgcc.csv"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "electricity_theft_rf.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
model_state: dict[str, Any] = {"model": None, "metrics": None, "metadata": None, "source": None, "error": None}


class ConsumerInput(BaseModel):
    consumption_mean: float = Field(..., ge=0, description="Average daily electricity consumption")
    consumption_std: float = Field(..., ge=0, description="Standard deviation of daily consumption")
    consumption_min: float = Field(..., ge=0, description="Minimum daily consumption")
    consumption_max: float = Field(..., ge=0, description="Maximum daily consumption")
    consumption_median: float = Field(..., ge=0, description="Median daily consumption")
    consumption_iqr: float = Field(..., ge=0, description="Interquartile range")
    zero_day_ratio: float = Field(..., ge=0, le=1, description="Fraction of days with zero consumption")
    missing_day_ratio: float = Field(0, ge=0, le=1, description="Fraction of days without readings")
    consumption_trend: float = Field(0, description="Trend in consumption across the period")


def load_and_train(frame: pd.DataFrame, source: str):
    model, metadata = train_and_evaluate(frame)
    metadata["dataset_file"] = Path(source).name
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model_state.update(model=model, metrics=metadata, metadata=metadata, source=source, error=None)


@app.on_event("startup")
def load_or_train_automatically():
    try:
        if MODEL_PATH.exists() and METADATA_PATH.exists():
            model = joblib.load(MODEL_PATH)
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            model_state.update(
                model=model, metrics=metadata, metadata=metadata,
                source=str(DATA_PATH), error=None
            )
            return
        if DATA_PATH.exists():
            frame = read_dataset(DATA_PATH.read_bytes())
            load_and_train(frame, str(DATA_PATH))
        else:
            model_state["error"] = f"Training CSV not found: {DATA_PATH}"
    except Exception as exc:
        model_state["error"] = str(exc)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_ready": model_state["model"] is not None,
        "training_source": Path(model_state["source"]).name if model_state["source"] else None,
        "message": model_state["error"],
    }


@app.post("/api/predict")
def predict(payload: ConsumerInput):
    model = model_state["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The model is not ready. Place the SGCC training CSV at backend/data/sgcc.csv and restart the backend."
        )
    try:
        result = predict_one(model, payload.model_dump())
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ReadingsInput(BaseModel):
    consumer_name: str = ""
    consumer_id: str = ""
    readings: list[float]


@app.post("/api/predict-readings")
def predict_readings(payload: ReadingsInput):
    model = model_state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="The model is still preparing. Wait for the status to show Model ready.")
    try:
        features = features_from_readings(payload.readings)
        row = features.iloc[0].to_dict()
        result = predict_one(model, row)
        result["consumer_name"] = payload.consumer_name
        result["consumer_id"] = payload.consumer_id
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/setup-model")
async def setup_model(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please select an SGCC CSV file.")
    try:
        frame = read_dataset(await file.read())
        load_and_train(frame, file.filename)
        return {"status": "ready", "message": "Model prepared in the background.", "source": file.filename}
    except Exception as exc:
        model_state.update(model=None, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please select a CSV file.")
    model = model_state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="The model is not ready. Place the SGCC training CSV at backend/data/sgcc.csv and restart the backend.")
    try:
        frame = read_dataset(await file.read())
        # A batch input may omit FLAG; when present, it is preserved as reference data but never used as an input feature.
        result = predict_csv_rows(model, frame)
        return {"filename": file.filename, "rows": result.astype(object).where(pd.notna(result), None).to_dict(orient="records"),
                "count": int(len(result)), "class_0": int((result["predicted_class"] == 0).sum()),
                "class_1": int((result["predicted_class"] == 1).sum())}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process this CSV: {exc}") from exc


@app.get("/api/model-status")
def model_status():
    if model_state["model"] is None:
        return {
            "ready": False,
            "message": model_state["error"] or "Model is waiting for the SGCC dataset.",
        }
    data = model_state["metadata"] or model_state["metrics"] or {}
    dataset = data.get("dataset", {})
    return {
        "ready": True,
        "source": Path(model_state["source"]).name if model_state["source"] else data.get("dataset_file", "SGCC dataset"),
        "records": data.get("training_rows", dataset.get("rows_with_valid_labels", 0)),
        "features": len(data.get("features", dataset.get("feature_names", []))),
        "model_name": data.get("model_name", "Random Forest"),
        "threshold": data.get("threshold", 0.5),
        "metrics": data.get("metrics", {}),
        "message": "Saved model loaded and ready for single-record and CSV predictions.",
    }
