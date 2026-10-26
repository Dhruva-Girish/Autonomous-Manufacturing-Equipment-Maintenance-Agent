"""Train and evaluate the equipment fault classification model.

Dataset labels are synthetic demo labels, not measurements from a real factory.
Run: python train_model.py
"""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "equipment_fault_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "fault_model.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
FEATURES = ["temperature_c", "vibration_mm_s", "current_a"]
TARGET = "fault_type"
EXPECTED_CLASSES = ["normal", "overheating", "excessive_vibration"]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    needed = set(FEATURES + [TARGET])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Dataset contains missing values")
    if not set(df[TARGET].unique()).issubset(set(EXPECTED_CLASSES)):
        raise ValueError(f"Unexpected labels. Expected only: {EXPECTED_CLASSES}")
    if set(df[TARGET].unique()) != set(EXPECTED_CLASSES):
        raise ValueError(f"Dataset must include all classes: {EXPECTED_CLASSES}")
    if (df[FEATURES] < 0).any().any():
        raise ValueError("Sensor values must be non-negative")

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    report = classification_report(
        y_test, predictions, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0
    )
    metrics = {
        "dataset": DATA_PATH.name,
        "dataset_type": "synthetic demonstration data",
        "total_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "classes": EXPECTED_CLASSES,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix_labels_order": EXPECTED_CLASSES,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=EXPECTED_CLASSES).tolist(),
        "classification_report": report,
        "important_warning": (
            "Metrics on synthetic data do not establish accuracy on real industrial equipment. "
            "Validate with representative, labeled machine data before operational use."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "classes": EXPECTED_CLASSES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Dataset: {DATA_PATH}")
    print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Test accuracy on synthetic holdout: {metrics['test_accuracy']:.2%}")
    print("Classification report:")
    print(classification_report(y_test, predictions, labels=EXPECTED_CLASSES, zero_division=0))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print("WARNING: synthetic test performance is not real-world fault-detection accuracy.")


if __name__ == "__main__":
    main()
