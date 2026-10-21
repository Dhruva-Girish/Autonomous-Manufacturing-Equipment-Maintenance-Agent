
from maintenance_agent import run_maintenance_agent


def test_overheating():
    sensor_data = {
        "machine_id": "MOTOR_01",
        "temperature_c": 105.0,
        "vibration_mm_s": 2.4,
        "current_a": 8.2,
    }

    fault_result = {
        "fault_detected": True,
        "fault_type": "overheating",
        "severity": "high",
        "evidence": ["Temperature exceeds the configured threshold"],
    }

    result = run_maintenance_agent(sensor_data, fault_result, use_llm=False)

    assert result["status"] == "needs_investigation"
    assert "overheating" in result["diagnosis"]
    assert result["evidence"] == fault_result["evidence"]


def test_normal_operation():
    sensor_data = {
        "machine_id": "MOTOR_01",
        "temperature_c": 65.0,
        "vibration_mm_s": 1.5,
        "current_a": 7.0,
    }

    fault_result = {
        "fault_detected": False,
        "fault_type": "normal",
        "severity": "low",
        "evidence": [],
    }

    result = run_maintenance_agent(sensor_data, fault_result, use_llm=False)

    assert result["status"] == "completed"
    assert result["work_order"] is None
