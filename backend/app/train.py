from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import joblib
from .ml import read_dataset, train_and_evaluate

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_DIR / "models"
MODEL_PATH = MODEL_DIR / "electricity_theft_rf.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GridGuard candidate classifiers.")
    parser.add_argument("--data", default=str(BACKEND_DIR / "data" / "sgcc.csv"), help="Path to labelled SGCC CSV")
    parser.add_argument("--report", default=None, help="Optional separate JSON report path")
    parser.add_argument("--no-save-model", action="store_true", help="Evaluate only; do not overwrite deployed model")
    args = parser.parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.exists(): raise FileNotFoundError(f"Training dataset not found: {data_path}")
    print(f"Reading dataset: {data_path}")
    frame = read_dataset(data_path.read_bytes())
    print(f"Raw dataset shape: {frame.shape[0]:,} rows x {frame.shape[1]:,} columns")
    model, metadata = train_and_evaluate(frame)
    metadata["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["dataset_file"] = data_path.name
    metadata["dataset_path_used"] = str(data_path)
    report_path = Path(args.report).resolve() if args.report else (MODEL_DIR / f"evaluation_{data_path.stem.replace(' ', '_')}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nEvaluation report: {report_path}")
    print(f"Rows: {metadata['training_rows']:,} | Test rows: {metadata['test_rows']:,} | Readings used per row: {metadata['dataset']['reading_columns_used']}")
    print(f"Selected model: {metadata['selected_model']} | Threshold: {metadata['threshold']:.2f}")
    print("\nHeld-out test results:")
    for item in metadata["model_comparison"]:
        print(f"{item['name']:<20} F1={item['f1']:.4f} Precision={item['precision']:.4f} Recall={item['recall']:.4f} Accuracy={item['accuracy']:.4f} AP={item['average_precision']:.4f} threshold={item['threshold']:.2f}")
    print("\nSelected model confusion matrix [[TN, FP], [FN, TP]]:")
    print(metadata["metrics"]["confusion_matrix"])
    if not args.no_save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH, compress=3)
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"\nSaved deployment model: {MODEL_PATH}")
        print(f"Saved deployment metadata: {METADATA_PATH}")
    else:
        print("Model was not overwritten (--no-save-model).")

if __name__ == "__main__": main()