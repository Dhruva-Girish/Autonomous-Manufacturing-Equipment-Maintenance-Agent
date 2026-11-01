
import os
import streamlit as st
from dotenv import load_dotenv

# Load API key from .env before importing the maintenance agent
load_dotenv()

from simulator import get_sensor_data
from fault_detector import detect_fault
from maintenance_agent import run_maintenance_agent

from maintenance_tools import (
    create_work_order,
    notify_technician,
    verify_machine_health
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Autonomous Maintenance Dashboard",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Autonomous Manufacturing Equipment Maintenance")
st.caption("AI-powered equipment monitoring, fault detection and maintenance")


# --------------------------------------------------
# API KEY CONFIGURATION
# --------------------------------------------------

api_key_available = bool(os.getenv("GEMINI_API_KEY"))


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Equipment Configuration")

scenario = st.sidebar.selectbox(
    "Select Motor Scenario",
    [
        "Normal Operation",
        "Overheating",
        "Excessive Vibration"
    ]
)

scenario_mapping = {
    "Normal Operation": "normal",
    "Overheating": "overheating",
    "Excessive Vibration": "excessive_vibration"
}

equipment_id = st.sidebar.text_input(
    "Equipment ID",
    value="M-101"
)

st.sidebar.divider()

use_gemini = st.sidebar.checkbox(
    "Use Gemini Maintenance Agent",
    value=api_key_available,
    disabled=not api_key_available
)

if api_key_available:
    st.sidebar.success("Gemini API key detected")
else:
    st.sidebar.warning(
        "Gemini API key not configured. "
        "Agent will run without LLM investigation."
    )


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def display_sensor_data(sensor_data):

    st.subheader("📊 Sensor Readings")

    columns = st.columns(5)

    readings = [
        ("Temperature", sensor_data.get("temperature"), "°C"),
        ("Vibration", sensor_data.get("vibration"), "mm/s"),
        ("Current", sensor_data.get("current"), "A"),
        ("Pressure", sensor_data.get("pressure"), "bar"),
        ("RPM", sensor_data.get("rpm"), "RPM")
    ]

    for column, (label, value, unit) in zip(columns, readings):
        with column:
            if isinstance(value, (int, float)):
                st.metric(label, f"{value:.2f} {unit}")
            else:
                st.metric(label, f"{value} {unit}")


def display_fault_result(fault_result):

    st.subheader("🔍 Fault Detection Results")

    if fault_result.get("fault_detected"):
        st.error("⚠️ Fault Detected")

        col1, col2 = st.columns(2)

        col1.metric(
            "Fault Type",
            str(fault_result.get("fault_type", "Unknown"))
        )

        col2.metric(
            "Severity",
            str(fault_result.get("severity", "Unknown"))
        )

    else:
        st.success("✅ No Fault Detected")

    st.write("**Detection Evidence:**")
    st.json(fault_result.get("evidence", {}))


def display_agent_result(agent_result):

    st.subheader("🤖 Maintenance Agent Results")

    st.write("**Agent Status:**")
    st.write(agent_result.get("status", "Not specified"))

    st.write("**Diagnosis:**")
    st.write(agent_result.get("diagnosis", "No diagnosis provided"))

    st.write("**Retrieved Maintenance Evidence:**")

    manual_guidance = agent_result.get("manual_guidance", {})

    if isinstance(manual_guidance, dict):

        st.write(manual_guidance.get("message", ""))

        instructions = manual_guidance.get("instructions", [])

        if instructions:
            st.write("**Instructions:**")
            for instruction in instructions:
                st.write(f"- {instruction}")

        passages = manual_guidance.get("retrieved_passages", [])

        if passages:
            for index, passage in enumerate(passages, start=1):

                with st.expander(f"Retrieved Passage {index}"):

                    if isinstance(passage, dict):
                        st.write("**Source:**", passage.get("source", "Unknown"))
                        st.write("**PDF Page:**", passage.get("pdf_page", "N/A"))
                        st.write(
                            "**Similarity Score:**",
                            passage.get("similarity_score", "N/A")
                        )
                        st.write(passage.get("text", ""))
                    else:
                        st.write(passage)

    else:
        st.write(manual_guidance)

    st.write("**Recommended Actions:**")
    actions = agent_result.get("recommended_actions", [])

    if actions:
        for action in actions:
            st.write(f"- {action}")
    else:
        st.info("No structured recommended actions were returned by the agent.")

    st.write("**Agent Evidence:**")
    st.json(agent_result.get("evidence", {}))

    with st.expander("View Agent Execution Log"):
        st.write(agent_result.get("agent_log", []))

    with st.expander("View Tool Results"):
        st.json(agent_result.get("tool_results", {}))


# --------------------------------------------------
# RUN SIMULATION
# --------------------------------------------------

if st.sidebar.button("▶ Run Simulation", type="primary"):

    try:

        # Step 1: Generate sensor data
        with st.spinner("Generating sensor readings..."):

            sensor_data = get_sensor_data(
                scenario=scenario_mapping[scenario],
                equipment_id=equipment_id
            )

        st.session_state["sensor_data"] = sensor_data
        st.session_state.pop("fault_result", None)
        st.session_state.pop("agent_result", None)

        # Step 2: Detect fault
        with st.spinner("Running fault detection model..."):

            fault_result = detect_fault(sensor_data)

        st.session_state["fault_result"] = fault_result

        st.success("Simulation completed successfully!")

    except Exception as e:
        st.error(f"Simulation error: {str(e)}")


# --------------------------------------------------
# DISPLAY SIMULATION RESULTS
# --------------------------------------------------

if "sensor_data" in st.session_state:

    sensor_data = st.session_state["sensor_data"]
    fault_result = st.session_state["fault_result"]

    st.divider()

    st.subheader("🏭 Equipment Information")

    col1, col2, col3 = st.columns(3)

    col1.write(f"**Equipment ID:** {sensor_data.get('equipment_id')}")
    col2.write(f"**Scenario:** {sensor_data.get('scenario')}")
    col3.write(f"**Timestamp:** {sensor_data.get('timestamp')}")

    display_sensor_data(sensor_data)

    st.divider()

    display_fault_result(fault_result)

    st.divider()

    # --------------------------------------------------
    # MAINTENANCE AGENT
    # --------------------------------------------------

    st.subheader("🛠️ Maintenance Investigation")

    if st.button("🤖 Run Maintenance Agent", type="primary"):

        try:

            with st.spinner("Maintenance agent is investigating..."):

                agent_result = run_maintenance_agent(
                    sensor_data,
                    fault_result,
                    use_llm=use_gemini
                )

            st.session_state["agent_result"] = agent_result
            st.success("Maintenance investigation completed!")

        except Exception as e:
            st.error(f"Maintenance agent error: {str(e)}")

    if "agent_result" in st.session_state:

        agent_result = st.session_state["agent_result"]

        display_agent_result(agent_result)

        st.divider()

        # --------------------------------------------------
        # HUMAN APPROVAL AND WORK ORDER
        # --------------------------------------------------

        if fault_result.get("fault_detected"):

            st.subheader("👤 Human Approval Required")

            st.warning(
                "Review the maintenance diagnosis and recommendations "
                "before authorizing a work order."
            )

            with st.form("maintenance_approval_form"):

                approved = st.checkbox(
                    "I have reviewed the recommendations and approve "
                    "creating a maintenance work order and notifying "
                    "the technician."
                )

                submitted = st.form_submit_button(
                    "Approve and Create Work Order"
                )

            if submitted:

                if not approved:

                    st.warning("Please provide approval before proceeding.")

                elif st.session_state.get("work_order_created"):

                    st.info("A work order has already been created for this simulation.")

                else:

                    try:

                        with st.spinner("Creating maintenance work order..."):

                            recommendations = agent_result.get(
                                "recommended_actions", []
                            )

                            if not recommendations:
                                recommendations = [
                                    "Inspect the equipment and investigate "
                                    "the detected fault before restarting."
                                ]

                            order = create_work_order(
                                fault=fault_result.get("fault_type", "unknown"),
                                recommendation=recommendations
                            )

                            notification_result = notify_technician(order)

                            st.session_state["work_order_created"] = True
                            st.session_state["work_order"] = order

                        st.success("Maintenance work order created!")

                        st.write("**Work Order Details:**")
                        st.json(order)

                        st.write("**Technician Notification:**")
                        st.json(notification_result)

                    except Exception as e:
                        st.error(f"Work order creation error: {str(e)}")

            # --------------------------------------------------
            # VERIFICATION
            # --------------------------------------------------

            st.divider()
            st.subheader("✅ Equipment Health Verification")

            if st.button("Verify Machine Health"):

                try:

                    with st.spinner("Verifying machine health..."):

                        verification = verify_machine_health(sensor_data)

                    st.session_state["verification"] = verification

                    st.success("Verification completed!")
                    st.json(verification)

                    st.caption(
                        "Note: This verification uses the current simulated "
                        "sensor readings. It does not prove that a physical "
                        "repair has been completed."
                    )

                except Exception as e:
                    st.error(f"Verification error: {str(e)}")

        else:

            st.success(
                "No fault detected. A maintenance work order is not required."
            )

else:

    st.info(
        "Select a motor scenario from the sidebar and click "
        "'Run Simulation' to begin."
    )
