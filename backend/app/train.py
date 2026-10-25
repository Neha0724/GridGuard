from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .ml import read_dataset, train_and_evaluate

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BACKEND_DIR / "data" / "sgcc.csv"
MODEL_DIR = BACKEND_DIR / "models"
MODEL_PATH = MODEL_DIR / "electricity_theft_rf.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {DATA_PATH}")
    print(f"Reading dataset: {DATA_PATH.name}")
    frame = read_dataset(DATA_PATH.read_bytes())
    print(f"Dataset shape: {frame.shape[0]:,} rows x {frame.shape[1]:,} columns")
    print("Engineering consumer features and evaluating a holdout split...")
    model, metadata = train_and_evaluate(frame)
    metadata["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["dataset_file"] = DATA_PATH.name
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nTraining complete.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Training rows: {metadata['training_rows']:,} | Test rows: {metadata['test_rows']:,}")
    print("Holdout metrics:")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"  {key.title():<10}: {metadata['metrics'][key]:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(metadata["metrics"]["confusion_matrix"])
    print("Top 5 features:")
    for item in metadata["feature_importance"][:5]:
        print(f"  {item['feature']:<24} {item['importance']:.5f}")


if __name__ == "__main__":
    main()
