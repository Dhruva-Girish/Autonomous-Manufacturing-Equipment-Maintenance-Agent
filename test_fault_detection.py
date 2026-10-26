"""Tests for Member 1's simulator and ML detector. Run: python -m pytest -q"""
import pytest
from simulator import SCENARIOS, get_sensor_data
from fault_detector import detect_fault

EXPECTED = {
    "normal": (False, "normal"),
    "overheating": (True, "overheating"),
    "excessive_vibration": (True, "excessive_vibration"),
}

@pytest.mark.parametrize("scenario", sorted(EXPECTED))
def test_primary_simulated_scenarios(scenario):
    sensor_data = get_sensor_data(scenario, seed=42)
    result = detect_fault(sensor_data)
    assert (result["fault_detected"], result["fault_type"]) == EXPECTED[scenario]
    assert set(result) == {"fault_detected", "fault_type", "severity", "evidence"}
    assert result["severity"] in {"low", "medium", "high"}
    assert isinstance(result["evidence"], list)

@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_all_simulator_scenarios_return_valid_schema(scenario):
    # Only the three EXPECTED classes are trained labels. Other simulator
    # scenarios should return a schema-valid best-match prediction, not be
    # considered independently trained classes.
    result = detect_fault(get_sensor_data(scenario, seed=42))
    assert set(result) == {"fault_detected", "fault_type", "severity", "evidence"}
    assert result["fault_type"] in {"normal", "overheating", "excessive_vibration"}
    assert result["severity"] in {"low", "medium", "high"}
    assert isinstance(result["evidence"], list)

@pytest.mark.parametrize("data", [
    {"temperature_c": 70, "current_a": 8},
    {"temperature_c": 70, "vibration_mm_s": float("nan"), "current_a": 8},
    {"temperature_c": -1, "vibration_mm_s": 2, "current_a": 8},
])
def test_invalid_inputs_raise_value_error(data):
    with pytest.raises(ValueError):
        detect_fault(data)
