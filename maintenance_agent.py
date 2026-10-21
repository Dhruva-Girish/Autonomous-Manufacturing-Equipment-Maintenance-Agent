
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-3.8-flash"


# Temporary mock tools.
# Replace their implementations when Member 3's tools are ready.

def mock_search_manual(query: str) -> dict:
    return {
        "success": True,
        "query": query,
        "source": "mock_motor_manual.txt",
        "guidance": (
            "Illustrative guidance: inspect the cooling system "
            "and consult the approved maintenance procedure."
        ),
    }


def mock_get_maintenance_history(machine_id: str) -> dict:
    return {
        "success": True,
        "machine_id": machine_id,
        "history": ["No previous maintenance records in mock data."],
    }


def run_maintenance_agent(
    sensor_data: dict[str, Any],
    fault_result: dict[str, Any],
    use_llm: bool = True,
) -> dict[str, Any]:
    """Investigate a simulated equipment fault using Gemini and tools."""

    result = {
        "status": "completed",
        "diagnosis": "",
        "evidence": list(fault_result.get("evidence", [])),
        "manual_guidance": [],
        "recommended_actions": [],
        "work_order": None,
        "approval_required": False,
        "verification": None,
        "agent_log": [],
        "tool_results": [],
    }

    # Validate the basic input structure.
    if not isinstance(sensor_data, dict) or not isinstance(fault_result, dict):
        result["status"] = "error"
        result["diagnosis"] = "Invalid input: expected dictionaries."
        return result

    if not fault_result.get("fault_detected", False):
        result["diagnosis"] = "No fault reported by the fault detector."
        result["agent_log"].append("No fault investigation required.")
        return result

    if not use_llm:
        result["status"] = "needs_investigation"
        result["diagnosis"] = (
            f"Possible fault: {fault_result.get('fault_type', 'unknown')}"
        )
        result["agent_log"].append("LLM call skipped.")
        return result

    if not os.getenv("GEMINI_API_KEY"):
        result["status"] = "error"
        result["diagnosis"] = "Gemini API key is not configured."
        result["agent_log"].append("Missing GEMINI_API_KEY.")
        return result

    # Record actual tool executions and their returned values.
    tool_results = result["tool_results"]

    def search_manual(query: str) -> dict:
        tool_result = mock_search_manual(query)
        tool_results.append({
            "tool": "search_manual",
            "success": tool_result.get("success", False),
            "result": tool_result,
        })
        result["agent_log"].append("Executed search_manual.")
        return tool_result

    def get_maintenance_history(machine_id: str) -> dict:
        tool_result = mock_get_maintenance_history(machine_id)
        tool_results.append({
            "tool": "get_maintenance_history",
            "success": tool_result.get("success", False),
            "result": tool_result,
        })
        result["agent_log"].append("Executed get_maintenance_history.")
        return tool_result

    prompt = f"""
You are the reasoning component of a simulated equipment maintenance agent.

Sensor readings:
{sensor_data}

Fault detector result:
{fault_result}

Investigate the reported fault.
Use search_manual to retrieve relevant guidance.
Use get_maintenance_history when useful.
Base recommendations on the sensor readings and actual tool results.
Clearly distinguish evidence from hypotheses.
Do not invent manual passages or maintenance history.
Do not claim a physical repair has occurred.
Do not create work orders or execute consequential actions.
Recommend qualified human inspection when appropriate.

Return a concise diagnosis and recommended next steps.
"""

    try:
        client = genai.Client()

        chat = client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                tools=[search_manual, get_maintenance_history],
                temperature=0.2,
            ),
        )

        response = chat.send_message(prompt)

        result["diagnosis"] = (
            response.text or "Gemini returned no text response."
        )

        # Preserve manual results separately for the dashboard.
        result["manual_guidance"] = [
            item["result"]
            for item in tool_results
            if item["tool"] == "search_manual"
        ]

        result["evidence"].extend(
            item["result"]
            for item in tool_results
            if item["success"]
        )

        result["status"] = "completed"
        result["agent_log"].append(
            f"Gemini investigation completed using {MODEL_NAME}."
        )

    except Exception as exc:
        result["status"] = "error"
        result["diagnosis"] = "The agent could not complete its investigation."
        result["agent_log"].append(
            f"Error: {type(exc).__name__}: {exc}"
        )

    return result
