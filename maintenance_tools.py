
import json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

import faiss
from sentence_transformers import SentenceTransformer

RAG_INDEX_PATH = Path(__file__).resolve().parent / "maintenance_faiss.index"
RAG_CHUNKS_PATH = Path(__file__).resolve().parent / "maintenance_chunks.json"
RAG_MODEL_NAME = "all-MiniLM-L6-v2"
ABB_MANUAL_SOURCE = "ABB Low Voltage Motors manual (2009)"

# Load the saved index and source chunks once when this module is imported.
RAG_INDEX = faiss.read_index(str(RAG_INDEX_PATH))
with RAG_CHUNKS_PATH.open("r", encoding="utf-8") as file:
    RAG_CHUNKS = json.load(file)

RAG_MODEL = SentenceTransformer(RAG_MODEL_NAME)

# Load the maintenance manual stored beside this Python file
MANUAL_PATH = Path(__file__).resolve().parent / "maintenance_manual.json"

with MANUAL_PATH.open("r", encoding="utf-8") as file:
    MAINTENANCE_MANUAL = json.load(file)

# Illustrative thresholds for our simulated motor only
TEMPERATURE_LIMIT = 80.0
VIBRATION_LIMIT = 5.0
CURRENT_LIMIT = 20.0

# Sample maintenance history (simulated)
MAINTENANCE_HISTORY = {
    "MOTOR-001": [
        {
            "date": "2026-10-10",
            "issue": "Routine inspection",
            "action": "Mounting and ventilation checked",
            "status": "completed"
        }
    ]
}

# In-memory work-order and notification records
WORK_ORDERS = []
TECHNICIAN_NOTIFICATIONS = []


def search_manual(fault):
    """Return structured guidance and relevant passages from the ABB manual."""

    fault_key = str(fault).strip().lower()
    fault_key = fault_key.replace(" ", "_").replace("-", "_")

    aliases = {
        "normal_operation": "normal",
        "healthy": "normal",
        "overheating": "overheating",
        "excessive_vibration": "excessive_vibration",
        "high_vibration": "excessive_vibration"
    }

    fault_key = aliases.get(fault_key, fault_key)

    if fault_key not in MAINTENANCE_MANUAL:
        if any(word in fault_key for word in ("overheat", "hot", "temperature", "cooling")):
            fault_key = "overheating"
        elif any(word in fault_key for word in ("vibration", "shaking")):
            fault_key = "excessive_vibration"
        elif any(word in fault_key for word in ("normal", "healthy", "no_fault", "no_issue")):
            fault_key = "normal"

    if fault_key not in MAINTENANCE_MANUAL:
        return {
            "success": False,
            "fault": fault,
            "message": "No matching maintenance instructions found."
        }

    query = str(fault).replace("_", " ").replace("-", " ").strip()
    query = f"Electric motor {query}: causes, inspection, maintenance and corrective actions"

    query_embedding = RAG_MODEL.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = RAG_INDEX.search(query_embedding, 3)

    retrieved_passages = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        chunk = RAG_CHUNKS[int(idx)]
        retrieved_passages.append({
            "source": ABB_MANUAL_SOURCE,
            "pdf_page": chunk["page"],
            "similarity_score": round(float(score), 3),
            "text": chunk["text"]
        })

    return {
        "success": True,
        "fault": fault_key,
        "instructions": MAINTENANCE_MANUAL[fault_key],
        "retrieved_passages": retrieved_passages,
        "source": ABB_MANUAL_SOURCE,
        "message": "Structured guidance and relevant manual passages retrieved."
    }


def get_maintenance_history(machine_id):
    """Retrieve simulated maintenance history for a machine."""

    machine_id = str(machine_id).strip()

    if not machine_id:
        return {
            "success": False,
            "machine_id": machine_id,
            "history": [],
            "message": "Machine ID is required."
        }

    history = MAINTENANCE_HISTORY.get(machine_id, [])

    return {
        "success": True,
        "machine_id": machine_id,
        "history": history,
        "message": (
            "History retrieved."
            if history else "No previous maintenance records found."
        )
    }


def create_work_order(fault, recommendation):
    """Create a simulated maintenance work order."""

    if not str(fault).strip():
        return {
            "success": False,
            "message": "Fault description is required."
        }

    if not recommendation:
        return {
            "success": False,
            "message": "A maintenance recommendation is required."
        }

    if isinstance(recommendation, str):
        recommendation = [recommendation]

    if not isinstance(recommendation, list) or not all(
        isinstance(item, str) and item.strip()
        for item in recommendation
    ):
        return {
            "success": False,
            "message": "Recommendation must be a string or list of strings."
        }

    order = {
        "work_order_id": "WO-" + uuid4().hex[:8].upper(),
        "fault": str(fault).strip(),
        "recommendation": recommendation,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    WORK_ORDERS.append(order)

    return {
        "success": True,
        "work_order": order,
        "message": "Simulated work order created successfully."
    }


def notify_technician(work_order):
    """Record a simulated technician notification."""

    if not isinstance(work_order, dict):
        return {
            "success": False,
            "message": "A valid work-order dictionary is required."
        }

    order = work_order.get("work_order", work_order)
    order_id = order.get("work_order_id")

    if not order_id:
        return {
            "success": False,
            "message": "Work-order ID is missing."
        }

    notification = {
        "notification_id": "NT-" + uuid4().hex[:8].upper(),
        "work_order_id": order_id,
        "recipient": "Simulated maintenance technician",
        "message": (
            f"Maintenance required for fault: {order.get('fault', 'Unknown')}"
        ),
        "status": "simulated_notification_recorded",
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    TECHNICIAN_NOTIFICATIONS.append(notification)

    return {
        "success": True,
        "notification": notification,
        "message": "Simulated technician notification recorded."
    }


def verify_machine_health(sensor_data):
    """Check whether sensor readings are within illustrative limits."""

    if not isinstance(sensor_data, dict):
        return {
            "success": False,
            "message": "Sensor data must be a dictionary."
        }

    required = ["temperature", "vibration", "current"]

    if any(key not in sensor_data for key in required):
        return {
            "success": False,
            "message": "Missing required readings: temperature, vibration, current."
        }

    try:
        readings = {
            key: float(sensor_data[key])
            for key in required
        }
    except (TypeError, ValueError):
        return {
            "success": False,
            "message": "All sensor readings must be numeric."
        }

    import math

    if any(
        not math.isfinite(value) or value < 0
        for value in readings.values()
    ):
        return {
            "success": False,
            "message": "Sensor readings must be finite, non-negative numbers."
        }

    issues = []

    if readings["temperature"] > TEMPERATURE_LIMIT:
        issues.append("overheating")

    if readings["vibration"] > VIBRATION_LIMIT:
        issues.append("excessive_vibration")

    if readings["current"] > CURRENT_LIMIT:
        issues.append("excessive_current")

    return {
        "success": True,
        "healthy": len(issues) == 0,
        "status": "healthy" if not issues else "fault_detected",
        "detected_issues": issues,
        "readings": readings,
        "message": (
            "All readings are within configured limits."
            if not issues
            else "Abnormal readings detected; further inspection is required."
        ),
        "note": "Thresholds are illustrative and apply only to this simulation."
    }
