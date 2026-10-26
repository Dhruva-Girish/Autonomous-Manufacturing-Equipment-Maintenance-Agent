"""ML-based equipment fault detector using a trained Random Forest model.

Public API stays compatible with the team's interface:
    detect_fault(sensor_data) -> {fault_detected, fault_type, severity, evidence}

Train/regenerate the model with `python train_model.py`.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fault_model.joblib"
FEATURES = ["temperature_c", "vibration_mm_s", "current_a"]


def _load_model_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run `python train_model.py` first."
        )
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise RuntimeError("The model file has an unsupported format; rerun train_model.py")
    return bundle


def _validate_sensor_data(sensor_data: dict[str, Any]) -> dict[str, float]:
    """Validate readings and accept both model and simulator field names."""
    if not isinstance(sensor_data, dict):
        raise ValueError("sensor_data must be a dictionary")

    # The simulator emits temperature/vibration/current, while the model
    # training CSV uses temperature_c/vibration_mm_s/current_a.
    aliases = {
        "temperature_c": "temperature",
        "vibration_mm_s": "vibration",
        "current_a": "current",
    }
    readings: dict[str, float] = {}
    for feature in FEATURES:
        source_key = feature if feature in sensor_data else aliases[feature]
        if source_key not in sensor_data:
            raise ValueError(
                f"Missing required sensor field: {feature} (or {aliases[feature]})"
            )
        try:
            value = float(sensor_data[source_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Sensor field {source_key!r} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"Sensor field {source_key!r} must be finite")
        if value < 0:
            raise ValueError(f"Sensor field {source_key!r} cannot be negative")
        readings[feature] = value
    return readings


def detect_fault(sensor_data: dict[str, Any]) -> dict[str, Any]:
    """Predict the fault class from sensor readings and return the fixed schema.

    The optional `scenario` field is deliberately ignored to avoid label leakage.
    Severity thresholds are illustrative and should be tuned to the target machine.
    """
    readings = _validate_sensor_data(sensor_data)
    bundle = _load_model_bundle()
    model = bundle["model"]
    row = pd.DataFrame([[readings[name] for name in FEATURES]], columns=FEATURES)
    predicted = str(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    confidence = float(max(probabilities))

    if predicted == "normal":
        return {
            "fault_detected": False,
            "fault_type": "normal",
            "severity": "low",
            "evidence": [],
        }

    if predicted == "overheating":
        severity = "high" if readings["temperature_c"] >= 105.0 else "medium"
        evidence = [
            f"ML model classified the readings as overheating (confidence {confidence:.1%}).",
            f"Temperature: {readings['temperature_c']:.2f} °C.",
        ]
    elif predicted == "excessive_vibration":
        severity = "high" if readings["vibration_mm_s"] >= 9.0 else "medium"
        evidence = [
            f"ML model classified the readings as excessive_vibration (confidence {confidence:.1%}).",
            f"Vibration: {readings['vibration_mm_s']:.2f} mm/s.",
        ]
    else:
        # Prevent unexpected labels from being silently treated as an expected fault.
        raise RuntimeError(f"Model returned an unsupported class: {predicted!r}")

    return {
        "fault_detected": True,
        "fault_type": predicted,
        "severity": severity,
        "evidence": evidence,
    }


if __name__ == "__main__":
    from simulator import SCENARIOS, get_sensor_data
    for scenario in sorted(SCENARIOS):
        sensor_data = get_sensor_data(scenario)
        print(f"\nScenario: {scenario}\nSensors: {sensor_data}\nResult: {detect_fault(sensor_data)}")
