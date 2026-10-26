"""
simulator.py
Synthetic telemetry generator for the Autonomous Manufacturing Equipment
Maintenance Agent with Tool Integration.

Public interface:
    get_sensor_data(scenario="normal", equipment_id="M-101", seed=None) -> dict

The output is a Python dictionary, so fault_detector.py can consume it directly
without JSON serialization.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Dict, Optional


SCENARIOS = {
    "normal",
    "overheating",
    "excessive_vibration",
    "high_current",
    "low_pressure",
    "multiple_faults",
}


def get_sensor_data(
    scenario: str = "normal",
    equipment_id: str = "M-101",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate one simulated telemetry sample.

    Args:
        scenario: One of SCENARIOS.
        equipment_id: Identifier for the simulated asset.
        seed: Optional random seed for reproducible tests.

    Returns:
        Dictionary containing metadata and numeric sensor readings.

    Raises:
        ValueError: If scenario is not supported.
    """
    normalized_scenario = scenario.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_scenario not in SCENARIOS:
        allowed = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unsupported scenario {scenario!r}. Choose from: {allowed}")

    rng = random.Random(seed)

    # Baseline ranges are illustrative demo values, not universal industrial limits.
    temperature = rng.uniform(65.0, 78.0)   # degrees Celsius
    vibration = rng.uniform(1.0, 4.0)       # mm/s RMS
    current = rng.uniform(5.0, 9.5)         # amperes
    pressure = rng.uniform(4.5, 6.5)        # bar
    rpm = rng.randint(1400, 1600)            # revolutions per minute

    if normalized_scenario == "overheating":
        temperature = rng.uniform(95.0, 110.0)
    elif normalized_scenario == "excessive_vibration":
        vibration = rng.uniform(8.0, 12.0)
    elif normalized_scenario == "high_current":
        current = rng.uniform(13.0, 18.0)
    elif normalized_scenario == "low_pressure":
        pressure = rng.uniform(1.0, 2.5)
    elif normalized_scenario == "multiple_faults":
        temperature = rng.uniform(95.0, 110.0)
        vibration = rng.uniform(8.0, 12.0)
        current = rng.uniform(13.0, 18.0)
        pressure = rng.uniform(1.0, 2.5)

    return {
        "equipment_id": equipment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": normalized_scenario,
        "temperature": round(temperature, 2),
        "vibration": round(vibration, 2),
        "current": round(current, 2),
        "pressure": round(pressure, 2),
        "rpm": rpm,
        "units": {
            "temperature": "degC",
            "vibration": "mm/s RMS",
            "current": "A",
            "pressure": "bar",
            "rpm": "rpm",
        },
    }


if __name__ == "__main__":
    import json

    for selected_scenario in ("normal", "overheating", "excessive_vibration", "high_current", "low_pressure", "multiple_faults"):
        print(f"\nScenario: {selected_scenario}")
        print(json.dumps(get_sensor_data(selected_scenario, seed=42), indent=2))
