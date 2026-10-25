import io
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_ORDER = [
    "consumption_mean", "consumption_std", "consumption_min",
    "consumption_max", "consumption_median", "consumption_iqr",
    "zero_day_ratio", "missing_day_ratio", "consumption_trend",
 ]


class ThresholdedClassifier:
    """Small serializable wrapper that applies a validation-tuned probability threshold."""
    def __init__(self, estimator, threshold=0.5):
        self.estimator = estimator
        self.threshold = float(threshold)
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
    date_cols = []
    for c in df.columns:
        if c in excluded: continue
        s = str(c).strip()
        parsed = pd.to_datetime(s, errors="coerce")
        if pd.notna(parsed) and ("/" in s or "-" in s): date_cols.append(c)
    if not date_cols:
        date_cols = [c for c in df.columns if c not in excluded and pd.to_numeric(df[c], errors="coerce").notna().mean() >= .5]
    if not date_cols:
        raise ValueError("No date-like or numeric consumption columns detected in the CSV.")
    return date_cols

def engineer_features(df: pd.DataFrame):
    date_cols = _reading_columns(df)
    x = df[date_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Calculate features row-wise from the daily readings. Missing values are retained for the imputer.
    features = pd.DataFrame(index=df.index)
    features["consumption_mean"] = x.mean(axis=1)
    features["consumption_std"] = x.std(axis=1).fillna(0)
    features["consumption_min"] = x.min(axis=1)
    features["consumption_max"] = x.max(axis=1)
    features["consumption_median"] = x.median(axis=1)
    features["consumption_iqr"] = x.quantile(.75, axis=1) - x.quantile(.25, axis=1)
    features["zero_day_ratio"] = x.eq(0).sum(axis=1) / x.notna().sum(axis=1).replace(0, np.nan)
    features["missing_day_ratio"] = x.isna().mean(axis=1)
    def trend(row):
        vals = row.to_numpy(dtype=float); mask = np.isfinite(vals)
        if mask.sum() < 2: return 0.0
        return float(np.polyfit(np.flatnonzero(mask), vals[mask], 1)[0])
    features["consumption_trend"] = x.apply(trend, axis=1)
    features = features[FEATURE_ORDER].replace([np.inf, -np.inf], np.nan)
    return features

def _labelled_data(df):
    label_col = col_name(df, "FLAG")
    if label_col is None: raise ValueError("Required target column FLAG was not found in the training CSV.")
    labels = pd.to_numeric(df[label_col], errors="coerce")
    valid = labels.isin([0, 1])
    features = engineer_features(df).loc[valid].reset_index(drop=True)
    y = labels.loc[valid].astype(int).reset_index(drop=True)
    consumer_col = col_name(df, "CONS_NO")
    ids = df.loc[valid, consumer_col].astype(str).reset_index(drop=True) if consumer_col else pd.Series([f"Consumer {i+1}" for i in range(int(valid.sum()))])
    meta = {"rows_uploaded": int(len(df)), "columns_uploaded": int(df.shape[1]), "reading_columns": len(_reading_columns(df)),
            "rows_with_valid_labels": int(valid.sum()), "rows_missing_or_invalid_labels": int((~valid).sum()),
            "missing_cells": int(df.isna().sum().sum()), "class_counts": {str(int(k)): int(v) for k,v in y.value_counts().sort_index().items()},
            "features_created": len(FEATURE_ORDER), "feature_names": FEATURE_ORDER}
    return features, y, ids, meta

def _metric(y, pred):
    return {"accuracy": round(float(accuracy_score(y,pred)),4), "precision": round(float(precision_score(y,pred,zero_division=0)),4),
            "recall": round(float(recall_score(y,pred,zero_division=0)),4), "f1": round(float(f1_score(y,pred,zero_division=0)),4),
            "confusion_matrix": confusion_matrix(y,pred,labels=[0,1]).tolist()}

def train_models(df):
    X,y,ids,meta = _labelled_data(df)
    counts=y.value_counts()
    if len(counts)<2 or int(counts.min())<5: raise ValueError("Training needs both FLAG classes and at least 5 labelled records in each class.")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    models={
      "Logistic Regression": Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(class_weight="balanced",max_iter=1200,random_state=42))]),
      "Random Forest": Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=160,class_weight="balanced_subsample",min_samples_leaf=2,random_state=42,n_jobs=-1))])}
    comparisons=[]; fitted={}
    for name,model in models.items():
      model.fit(X_train,y_train); pred=model.predict(X_test); m=_metric(y_test,pred)
      comparisons.append({"name":name,**{k:m[k] for k in ("accuracy","precision","recall","f1")}}); fitted[name]=(model,pred,m)
    selected=max(comparisons,key=lambda r:r["f1"])["name"]
    rf=fitted["Random Forest"][0]
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    scores=cross_val_score(rf,X,y,cv=cv,scoring="f1",n_jobs=1)
    importances=[]
    for name,val in sorted(zip(FEATURE_ORDER,rf.named_steps["model"].feature_importances_),key=lambda p:p[1],reverse=True): importances.append({"feature":name,"importance":round(float(val),4)})
    pred=fitted[selected][1]
    return {"dataset":meta,"models":comparisons,"selected_model":selected,"selected_metrics":fitted[selected][2],
            "cross_validation":{"folds":5,"f1_scores":[round(float(s),4) for s in scores],"mean_f1":round(float(scores.mean()),4),"std_f1":round(float(scores.std()),4)},
            "feature_importance":importances[:6],"flagged_consumers":[],"test_size":int(len(y_test)),
            "test_flagged_actual":int(y_test.sum()),"test_flagged_predicted":int(np.sum(pred)),
            "note":"Predictions are risk signals for human review, not proof of theft. Verify FLAG semantics and dataset provenance."}

def build_prediction_model(frame: pd.DataFrame):
    X,y,_,_=_labelled_data(frame)
    if len(y.value_counts())<2: raise ValueError("The training CSV must contain both FLAG=0 and FLAG=1.")
    model=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=180,class_weight="balanced_subsample",min_samples_leaf=2,random_state=42,n_jobs=-1))])
    model.fit(X,y)
    return model

def features_from_readings(readings):
    """Build the same engineered feature vector from one consumer's daily readings."""
    values = pd.to_numeric(pd.Series(readings), errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = values.dropna()
    if len(valid) < 2:
        raise ValueError("Enter at least 2 valid daily consumption readings.")
    if (valid < 0).any():
        raise ValueError("Consumption readings cannot be negative.")
    x = valid.to_numpy(dtype=float)
    trend = float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) >= 2 else 0.0
    row = {
        "consumption_mean": float(np.mean(x)),
        "consumption_std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "consumption_min": float(np.min(x)),
        "consumption_max": float(np.max(x)),
        "consumption_median": float(np.median(x)),
        "consumption_iqr": float(np.quantile(x, .75) - np.quantile(x, .25)),
        "zero_day_ratio": float(np.mean(x == 0)),
        "missing_day_ratio": float((len(readings) - len(valid)) / max(len(readings), 1)),
        "consumption_trend": trend,
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def predict_one(model, payload: dict[str, float]) -> dict:
    row=pd.DataFrame([{key:float(payload[key]) for key in FEATURE_ORDER}],columns=FEATURE_ORDER)
    probability=float(model.predict_proba(row)[0][1]); prediction=int(model.predict(row)[0])
    return {"prediction":prediction,"label":"Potentially suspicious" if prediction==1 else "Likely safe",
            "risk_score":round(probability*100,1),"confidence_note":"A model-based risk indicator, not proof of theft.","features":row.iloc[0].to_dict()}

def predict_csv_rows(model, df: pd.DataFrame):
    features=engineer_features(df)
    probabilities=model.predict_proba(features)[:,1]
    predictions=model.predict(features).astype(int)
    out=df.copy()
    out["predicted_class"] = predictions
    out["prediction_label"] = ["Potentially suspicious" if int(v)==1 else "Likely safe" for v in predictions]
    out["risk_score_percent"] = np.round(probabilities*100,1)
    return out


def train_and_evaluate(df: pd.DataFrame):
    """Tune a classification threshold on validation data, report held-out metrics, then fit final model."""
    X, y, _, meta = _labelled_data(df)
    counts = y.value_counts()
    if len(counts) < 2 or int(counts.min()) < 5:
        raise ValueError("Training needs both FLAG classes and at least 5 labelled records in each class.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=43, stratify=y_train
    )

    def make_pipeline():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=180, class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )),
        ])

    tuning_model = make_pipeline()
    tuning_model.fit(X_fit, y_fit)
    val_probabilities = tuning_model.predict_proba(X_val)[:, 1]
    thresholds = [round(float(t), 2) for t in np.arange(0.10, 0.51, 0.02)]
    threshold_scores = []
    for threshold in thresholds:
        val_pred = (val_probabilities >= threshold).astype(int)
        threshold_scores.append((float(f1_score(y_val, val_pred, zero_division=0)), threshold))
    best_f1 = max(score for score, _ in threshold_scores)
    best_threshold = min(threshold for score, threshold in threshold_scores if score == best_f1)

    evaluation_model = make_pipeline()
    evaluation_model.fit(X_train, y_train)
    test_probabilities = evaluation_model.predict_proba(X_test)[:, 1]
    test_pred = (test_probabilities >= best_threshold).astype(int)
    metrics = _metric(y_test, test_pred)

    importances = sorted(
        [{"feature": name, "importance": round(float(value), 5)}
         for name, value in zip(FEATURE_ORDER, evaluation_model.named_steps["model"].feature_importances_)],
        key=lambda item: item["importance"], reverse=True
    )

    final_estimator = make_pipeline()
    final_estimator.fit(X, y)
    final_model = ThresholdedClassifier(final_estimator, best_threshold)
    metadata = {
        "model_name": "Random Forest",
        "model_version": "1.1",
        "dataset": meta,
        "metrics": metrics,
        "threshold": best_threshold,
        "threshold_selection": "Best F1 on an internal validation split of the training partition",
        "feature_importance": importances,
        "training_rows": int(len(X)),
        "test_rows": int(len(X_test)),
        "class_counts": {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items()},
        "features": FEATURE_ORDER,
        "target": "FLAG",
        "label_note": "Confirm FLAG semantics with the dataset documentation; class 1 is treated as flagged/suspicious for this prototype.",
    }
    return final_model, metadata
