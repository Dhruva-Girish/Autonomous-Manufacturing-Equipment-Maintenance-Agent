# Member 1 — Equipment Simulation and ML Fault Detection

This module supplies synthetic industrial sensor readings and a Random Forest classifier for three trained classes: `normal`, `overheating`, and `excessive_vibration`.

## Files

- `simulator.py` — creates sample telemetry for six demonstration scenarios.
- `fault_detector.py` — validates readings and returns the shared output schema: `fault_detected`, `fault_type`, `severity`, and `evidence`.
- `train_model.py` — trains and evaluates the Random Forest model from the CSV.
- `data/equipment_fault_data.csv` — synthetic labeled training data.
- `models/fault_model.joblib` — pre-trained model so the detector can run immediately.
- `models/evaluation_metrics.json` — evaluation details from the training run.
- `test_fault_detection.py` — automated tests.

## Setup and run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python fault_detector.py
```

## Retrain

```bash
python train_model.py
```

The detector accepts both the simulator's field names (`temperature`, `vibration`, `current`) and model feature names (`temperature_c`, `vibration_mm_s`, `current_a`).

**Important:** the CSV is synthetic demo data. Only three labels are included in the trained model. Simulator scenarios `high_current`, `low_pressure`, and `multiple_faults` are not separately modeled; add appropriately labeled real or validated data and retrain before claiming support for them. Accuracy on synthetic test data is not real-world industrial accuracy.
