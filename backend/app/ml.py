from __future__ import annotations

import io
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_ORDER = [
    "consumption_mean", "consumption_std", "consumption_min",
    "consumption_max", "consumption_median", "consumption_iqr",
    "zero_day_ratio", "missing_day_ratio", "consumption_trend",
]
READING_WINDOW = 26

class ThresholdedClassifier:
    """Serializable estimator wrapper; preserves predict/predict_proba API."""
    def __init__(self, estimator, threshold=0.5, model_name="Random Forest"):
        self.estimator = estimator
        self.threshold = float(threshold)
        self.model_name = model_name
        self.classes_ = getattr(estimator, "classes_", np.array([0, 1]))
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

def read_dataset(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError("CSV has no rows.")
    return df

def col_name(df, name):
    return next((c for c in df.columns if str(c).strip().casefold() == name.casefold()), None)

def _reading_columns(df):
    label_col, consumer_col = col_name(df, "FLAG"), col_name(df, "CONS_NO")
    excluded = {c for c in (label_col, consumer_col) if c is not None}
    dated = []
    for c in df.columns:
        if c in excluded:
            continue
        s = str(c).strip()
        parsed = pd.to_datetime(s, errors="coerce")
        if pd.notna(parsed) and ("/" in s or "-" in s):
            dated.append((parsed, c))
    if dated:
        # Sort chronological columns and use the same 26-reading horizon for training and CSV inference.
        dated.sort(key=lambda pair: pair[0])
        return [c for _, c in dated[-READING_WINDOW:]]
    numeric = [c for c in df.columns if c not in excluded and pd.to_numeric(df[c], errors="coerce").notna().mean() >= .5]
    if not numeric:
        raise ValueError("No date-like or numeric consumption columns detected in the CSV.")
    return numeric[-READING_WINDOW:]

def engineer_features(df: pd.DataFrame):
    date_cols = _reading_columns(df)
    x = df[date_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    features = pd.DataFrame(index=df.index)
    features["consumption_mean"] = x.mean(axis=1)
    features["consumption_std"] = x.std(axis=1).fillna(0)
    features["consumption_min"] = x.min(axis=1)
    features["consumption_max"] = x.max(axis=1)
    features["consumption_median"] = x.median(axis=1)
    features["consumption_iqr"] = x.quantile(.75, axis=1) - x.quantile(.25, axis=1)
    valid_count = x.notna().sum(axis=1).replace(0, np.nan)
    features["zero_day_ratio"] = x.eq(0).sum(axis=1) / valid_count
    features["missing_day_ratio"] = x.isna().mean(axis=1)
    # Vectorized slope calculation; missing readings are excluded per consumer.
    arr = x.to_numpy(dtype=float)
    t = np.arange(arr.shape[1], dtype=float)[None, :]
    mask = np.isfinite(arr)
    n = mask.sum(axis=1)
    safe = np.where(mask, arr, 0.0)
    sum_t = (mask * t).sum(axis=1)
    sum_y = safe.sum(axis=1)
    sum_tt = (mask * t * t).sum(axis=1)
    sum_ty = (safe * t).sum(axis=1)
    denom = n * sum_tt - sum_t * sum_t
    slope = np.divide(n * sum_ty - sum_t * sum_y, denom, out=np.zeros(len(df), dtype=float), where=(n >= 2) & (denom != 0))
    features["consumption_trend"] = slope
    return features[FEATURE_ORDER].replace([np.inf, -np.inf], np.nan)

def _labelled_data(df):
    label_col = col_name(df, "FLAG")
    if label_col is None:
        raise ValueError("Required target column FLAG was not found in the training CSV.")
    labels = pd.to_numeric(df[label_col], errors="coerce")
    valid = labels.isin([0, 1])
    features = engineer_features(df).loc[valid].reset_index(drop=True)
    y = labels.loc[valid].astype(int).reset_index(drop=True)
    consumer_col = col_name(df, "CONS_NO")
    meta = {"rows_uploaded": int(len(df)), "columns_uploaded": int(df.shape[1]), "reading_columns_used": len(_reading_columns(df)),
            "rows_with_valid_labels": int(valid.sum()), "rows_missing_or_invalid_labels": int((~valid).sum()),
            "missing_cells": int(df.isna().sum().sum()), "class_counts": {str(int(k)): int(v) for k,v in y.value_counts().sort_index().items()},
            "features_created": len(FEATURE_ORDER), "feature_names": FEATURE_ORDER, "reading_window_policy": "latest 26 dated readings"}
    return features, y, meta

def _metric(y, pred, probabilities=None):
    result = {"accuracy": round(float(accuracy_score(y,pred)),4), "precision": round(float(precision_score(y,pred,zero_division=0)),4),
              "recall": round(float(recall_score(y,pred,zero_division=0)),4), "f1": round(float(f1_score(y,pred,zero_division=0)),4),
              "confusion_matrix": confusion_matrix(y,pred,labels=[0,1]).tolist()}
    if probabilities is not None:
        result["average_precision"] = round(float(average_precision_score(y, probabilities)),4)
        result["roc_auc"] = round(float(roc_auc_score(y, probabilities)),4) if len(np.unique(y)) == 2 else None
    return result

def _make_candidates():
    return {
        "Logistic Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42))]),
        "Random Forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))]),
        "Extra Trees": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))]),
    }

def train_and_evaluate(df: pd.DataFrame):
    """Compare candidates using validation F1; report once on untouched stratified test set; refit winner on all data."""
    X, y, meta = _labelled_data(df)
    counts = y.value_counts()
    if len(counts) < 2 or int(counts.min()) < 5:
        raise ValueError("Training needs both FLAG classes and at least 5 labelled records in each class.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=43, stratify=y_train)
    thresholds = np.arange(.10, .701, .02)
    candidates = []
    trained = {}
    for name, model in _make_candidates().items():
        model.fit(X_fit, y_fit)
        val_probs = model.predict_proba(X_val)[:, 1]
        threshold_rows = []
        for threshold in thresholds:
            pred = (val_probs >= threshold).astype(int)
            threshold_rows.append((float(f1_score(y_val, pred, zero_division=0)), float(threshold)))
        best_val_f1 = max(x[0] for x in threshold_rows)
        best_threshold = min(t for score, t in threshold_rows if score == best_val_f1)
        candidates.append({"name": name, "validation_f1": round(best_val_f1, 4), "threshold": round(best_threshold, 2)})
        trained[name] = (model, best_threshold)
    # Select only from validation results; test data is not used for selection.
    selected = max(candidates, key=lambda r: (r["validation_f1"], r["name"] == "Random Forest"))
    selected_name = selected["name"]
    evaluation_model, threshold = trained[selected_name]
    test_probs = evaluation_model.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= threshold).astype(int)
    metrics = _metric(y_test, test_pred, test_probs)
    # Fit the chosen candidate on all labelled records for deployment.
    final_estimator = _make_candidates()[selected_name]
    final_estimator.fit(X, y)
    wrapped = ThresholdedClassifier(final_estimator, threshold, selected_name)
    importances = []
    fitted_core = final_estimator.named_steps["model"]
    if hasattr(fitted_core, "feature_importances_"):
        importances = sorted([{"feature": n, "importance": round(float(v), 5)} for n,v in zip(FEATURE_ORDER, fitted_core.feature_importances_)], key=lambda a:a["importance"], reverse=True)
    comparison = []
    for row in candidates:
        name, thr = row["name"], row["threshold"]
        estimator = trained[name][0]
        probs = estimator.predict_proba(X_test)[:, 1]
        preds = (probs >= thr).astype(int)
        comparison.append({"name": name, "validation_f1": row["validation_f1"], "threshold": thr, **_metric(y_test, preds, probs)})
    metadata = {"model_name": selected_name, "model_version": "2.0", "dataset": meta, "metrics": metrics,
        "model_comparison": comparison, "selected_model": selected_name, "threshold": round(float(threshold),2),
        "threshold_selection": "Best F1 over validation split; candidate selection uses validation F1; test split remains held out",
        "feature_importance": importances, "training_rows": int(len(X)), "test_rows": int(len(X_test)),
        "class_counts": {str(int(k)): int(v) for k,v in y.value_counts().sort_index().items()}, "features": FEATURE_ORDER, "target": "FLAG",
        "label_note": "Confirm FLAG semantics with dataset documentation; class 1 is treated as flagged/suspicious for this prototype."}
    return wrapped, metadata

def features_from_readings(readings):
    values = pd.to_numeric(pd.Series(readings), errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = values.dropna()
    if len(valid) < 2:
        raise ValueError("Enter at least 2 valid daily consumption readings.")
    if (valid < 0).any():
        raise ValueError("Consumption readings cannot be negative.")
    x = valid.to_numpy(dtype=float)
    trend = float(np.polyfit(np.arange(len(x)), x, 1)[0])
    row = {"consumption_mean":float(np.mean(x)), "consumption_std":float(np.std(x,ddof=1)), "consumption_min":float(np.min(x)),
        "consumption_max":float(np.max(x)), "consumption_median":float(np.median(x)), "consumption_iqr":float(np.quantile(x,.75)-np.quantile(x,.25)),
        "zero_day_ratio":float(np.mean(x==0)), "missing_day_ratio":float((len(readings)-len(valid))/max(len(readings),1)), "consumption_trend":trend}
    return pd.DataFrame([row], columns=FEATURE_ORDER)

def predict_one(model, payload: dict[str, float]) -> dict:
    row = pd.DataFrame([{key:float(payload[key]) for key in FEATURE_ORDER}], columns=FEATURE_ORDER)
    probability=float(model.predict_proba(row)[0][1]); prediction=int(model.predict(row)[0])
    return {"prediction":prediction,"label":"Potentially suspicious" if prediction==1 else "Likely safe", "risk_score":round(probability*100,1),
        "confidence_note":"A model-based risk indicator, not proof of theft.","features":row.iloc[0].to_dict()}

def predict_csv_rows(model, df: pd.DataFrame):
    features=engineer_features(df)
    probabilities=model.predict_proba(features)[:,1]
    predictions=model.predict(features).astype(int)
    out=df.copy(); out["predicted_class"]=predictions
    out["prediction_label"]=["Potentially suspicious" if int(v)==1 else "Likely safe" for v in predictions]
    out["risk_score_percent"]=np.round(probabilities*100,1)
    return out