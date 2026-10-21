
from maintenance_agent import run_maintenance_agent


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

result = run_maintenance_agent(sensor_data, fault_result)

print("Status:", result["status"])
print("Diagnosis:", result["diagnosis"])
print("Agent log:", result["agent_log"])


from unittest.mock import patch

from maintenance_agent import run_maintenance_agent


def test_missing_api_key():
    sensor_data = {"machine_id": "MOTOR_01"}
    fault_result = {
        "fault_detected": True,
        "fault_type": "overheating",
        "severity": "high",
        "evidence": [],
    }

    with patch.dict("os.environ", {}, clear=True):
        result = run_maintenance_agent(sensor_data, fault_result)

    assert result["status"] == "error"
    assert "API key" in result["diagnosis"]


def test_invalid_input():
    result = run_maintenance_agent(
        sensor_data=None,
        fault_result={"fault_detected": True},
        use_llm=False,
    )

    assert result["status"] == "error"


def test_tool_results_are_returned():
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
        "evidence": ["High temperature"],
    }

    # Mock the Gemini SDK so this test makes no network request.
    with patch("maintenance_agent.genai.Client") as client_class:
        client = client_class.return_value
        chat = client.chats.create.return_value

        # The SDK normally executes the registered tool automatically.
        # This test instead checks the response path without an API call.
        chat.send_message.return_value.text = "Inspect the cooling system."

        result = run_maintenance_agent(sensor_data, fault_result)

    assert result["status"] == "completed"
    assert result["diagnosis"] == "Inspect the cooling system."
    assert isinstance(result["tool_results"], list)

print("Tool results:", result["tool_results"])