# 3. Setup & Configuration

!pip install -q -U google-generativeai

import os
import time
import uuid
import json
import re
import logging
from typing import Dict, Any, List

# --- API KEY SETUP ---
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")
    print("API Key loaded from Kaggle Secrets.")
except Exception:
    print("Could not load GOOGLE_API_KEY from Kaggle Secrets. Make sure it is set.")
    if "GOOGLE_API_KEY" not in os.environ:
        import getpass
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --- GOOGLE GENERATIVE AI INITIALIZATION ---
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# We'll *try* to use Gemini 1.5 Flash, and fall back gracefully if it isn't available.
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

def call_gemini(prompt: str, model: str = DEFAULT_GEMINI_MODEL) -> str:
    """
    Calls the configured generative model and returns response text.
    If the model is not available for this key, returns an empty string.
    """
    try:
        m = genai.GenerativeModel(model)
        resp = m.generate_content(prompt)
        return resp.text or ""
    except Exception as e:
        print(f"[WARN] LLM call failed for model '{model}': {e}")
        # Optional: Try a fallback model once
        try:
            fallback_model = "gemini-1.0-pro"
            m = genai.GenerativeModel(fallback_model)
            resp = m.generate_content(prompt)
            return resp.text or ""
        except Exception as e2:
            print(f"[WARN] Fallback LLM call also failed: {e2}")
            return ""  # stubbed response

def parse_json_response(raw_content: str) -> Dict[str, Any]:
    """
    Cleans and parses JSON from LLM output, handling markdown code fences.
    Returns {} if parsing fails.
    """
    try:
        clean = re.sub(r"```json\s*|\s*```", "", raw_content).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        if raw_content:
            print("[ERROR] Failed to parse JSON. Raw content was:")
            print(raw_content)
        return {}

print("âœ… Setup complete.")


test_raw = call_gemini('Return a JSON object: {"ok": true, "source": "netops-ally"}')
print(test_raw)


# 4. Tools: Mock Network Diagnostics

def ping_device(ip_address: str) -> str:
    """Simulates a network ping to check reachability."""
    # SIMULATION: 192.168.1.1 is the broken device
    if ip_address == "192.168.1.1":
        return "Request timed out. Packet loss: 100%. Latency: N/A."
    return f"Reply from {ip_address}: bytes=32 time=2ms TTL=64. Packet loss: 0%."

def check_device_logs(ip_address: str) -> str:
    """Simulates retrieving the last 10 log lines from the device."""
    if ip_address == "192.168.1.1":
        return (
            "[Oct 12 10:00:01] %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down\n"
            "[Oct 12 10:00:02] %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down\n"
            "[Oct 12 10:00:05] %SEC-4-PORT_SECURITY: Security violation occurred on interface GigabitEthernet0/1."
        )
    return "[INFO] System boot completed. No critical errors found."

def show_interface_status(ip_address: str, interface: str = "GigabitEthernet0/1") -> str:
    """Simulates 'show interface status' command."""
    if ip_address == "192.168.1.1":
        return f"{interface} is DOWN, line protocol is DOWN (err-disabled)."
    return f"{interface} is UP, line protocol is UP."

print("âœ… Tools loaded: ping_device, check_device_logs, show_interface_status")



# 5. Memory, Observability & IP Extraction

import re

metrics = {
    "incidents_count": 0,
    "tools_called": 0,
    "successful_resolutions": 0,
}

device_history: Dict[str, List[Dict[str, Any]]] = {}

def start_trace() -> str:
    return str(uuid.uuid4())[:8]

def log_event(trace_id: str, agent_name: str, message: str) -> None:
    print(f"ğŸ”¹ [{trace_id}] [{agent_name}] {message}")

def store_history(ip: str | None, data: Dict[str, Any]):
    if not ip:
        return
    device_history.setdefault(ip, []).append(data)

# Simple IPv4 extractor
IP_REGEX = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

def extract_ip_from_text(text: str) -> str | None:
    """Extracts the first IPv4 address from the given text, if any."""
    match = re.search(IP_REGEX, text)
    return match.group(0) if match else None



# 6. Intake Agent (LLM + regex fallback)

def intake_agent(incident_text: str, trace_id: str) -> Dict[str, Any]:
    log_event(trace_id, "Intake", "Analyzing incoming alert...")

    # 1) Regex-based IP extraction (works even if LLM fails)
    regex_ip = extract_ip_from_text(incident_text)

    # 2) Try to get richer structure from LLM (may fail / be stubbed)
    prompt = f"""
    You are the Intake Agent for a Network Operations Center.
    Analyze the following incident report.

    Incident: "{incident_text}"

    Return ONLY a JSON object with these keys:
    - "device_ip": The IP address mentioned (string, or null if none).
    - "main_symptom": A short summary of the issue.
    - "estimated_severity": "High", "Medium", or "Low".
    """

    raw = call_gemini(prompt)  # may be empty if model not available
    parsed_data = parse_json_response(raw)

    # 3) Fallbacks / merge logic
    # If LLM didn't give device_ip, use regex_ip
    if "device_ip" not in parsed_data or not parsed_data.get("device_ip"):
        parsed_data["device_ip"] = regex_ip

    # If LLM didn't give symptom, use a snippet of the original text
    if "main_symptom" not in parsed_data or not parsed_data.get("main_symptom"):
        parsed_data["main_symptom"] = incident_text[:80]

    # If LLM didn't give severity, set Medium
    if "estimated_severity" not in parsed_data or not parsed_data.get("estimated_severity"):
        parsed_data["estimated_severity"] = "Medium"

    log_event(trace_id, "Intake", f"Extracted: {parsed_data}")
    return parsed_data


# 7. Diagnostics Agent

def diagnostics_agent(state: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    """
    Decides which tools to run based on the IP and symptom.
    This is a simple linear "smart" flow; in a full LangGraph implementation,
    this would be a loop with LLM decisions between steps.
    """
    log_event(trace_id, "Diagnostics", "Starting investigation...")

    ip = state.get("device_ip")
    tools_used_count = 0
    findings: Dict[str, Any] = {}

    if not ip:
        findings["error"] = "No IP provided, cannot run diagnostics."
        state["findings"] = findings
        return state

    # Step 1: Check connectivity (Ping)
    log_event(trace_id, "Diagnostics", f"Pinging {ip}...")
    ping_result = ping_device(ip)
    findings["ping"] = ping_result
    tools_used_count += 1

    # Step 2: Adaptive logic
    if "timed out" in ping_result.lower() or "unreachable" in ping_result.lower():
        log_event(trace_id, "Diagnostics", "Ping failed. Fetching device logs...")
        logs_result = check_device_logs(ip)
        findings["logs"] = logs_result
        tools_used_count += 1

        if "GigabitEthernet0/1" in logs_result or "GigabitEthernet" in logs_result:
            log_event(trace_id, "Diagnostics", "Logs implicate GigabitEthernet0/1. Checking interface status...")
            int_status = show_interface_status(ip, "GigabitEthernet0/1")
            findings["interface_status"] = int_status
            tools_used_count += 1

    state["findings"] = findings
    metrics["tools_called"] += tools_used_count
    return state


# 8. Remediation Agent (LLM + rule-based fallback)

def remediation_agent(state: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    log_event(trace_id, "Remediation", "Synthesizing root cause and fix...")

    ip = state.get("device_ip")
    findings = state.get("findings", {})

    # 1) Try LLM first
    prompt = f"""
    You are a Senior Network Engineer. Analyze the diagnostic findings for device {ip}.

    Diagnostic Findings:
    {json.dumps(findings, indent=2)}

    Task:
    1. Determine the Root Cause (e.g., Cable cut, Port Security, Config error).
    2. Write a clear Explanation.
    3. Suggest a specific Cisco IOS command to fix it (or "none" if not safe).
    4. Write a professional Escalation Note for the ticket.

    Return ONLY a JSON object with keys:
    - "root_cause"
    - "explanation"
    - "cli_recommendation"
    - "escalation_note"
    """

    raw = call_gemini(prompt)
    remediation_data = parse_json_response(raw)

    # 2) If LLM failed or returned empty, use simple rule-based logic
    if not remediation_data:
        ping = findings.get("ping", "") or ""
        logs = findings.get("logs", "") or ""
        int_status = findings.get("interface_status", "") or ""

        if "err-disabled" in int_status.lower() or ("down" in int_status and "up" not in int_status.lower()):
            remediation_data = {
                "root_cause": "interface_errdisabled",
                "explanation": "The interface appears to be err-disabled or administratively down, causing loss of connectivity.",
                "cli_recommendation": "interface GigabitEthernet0/1\n no shutdown",
                "escalation_note": f"Device {ip}: Interface GigabitEthernet0/1 appears err-disabled/down. Verify port security and issue 'no shutdown' if appropriate.",
            }
        elif "100%" in ping or "timed out" in ping.lower():
            remediation_data = {
                "root_cause": "device_unreachable",
                "explanation": "Ping shows 100% packet loss; device is unreachable from the NOC.",
                "cli_recommendation": "none",
                "escalation_note": f"Device {ip} is unreachable (100% packet loss). Check power, cabling, and upstream links.",
            }
        else:
            remediation_data = {
                "root_cause": "unknown",
                "explanation": "Unable to determine root cause from current findings.",
                "cli_recommendation": "none",
                "escalation_note": f"Device {ip}: No clear root cause detected. Escalate to Tier-2 for deeper investigation.",
            }

    # 3) Ensure all keys exist
    remediation_data.setdefault("root_cause", "unknown")
    remediation_data.setdefault("explanation", "No explanation generated.")
    remediation_data.setdefault("cli_recommendation", "none")
    remediation_data.setdefault("escalation_note", "No escalation note generated.")

    state.update(remediation_data)
    log_event(trace_id, "Remediation", f"Root Cause identified: {state.get('root_cause')}")
    return state


# 9. Orchestration: NetOps Ally Workflow

def run_incident(incident_text: str) -> Dict[str, Any]:
    trace_id = start_trace()
    start_time = time.time()
    metrics["incidents_count"] += 1

    print(f"\nğŸš€ STARTING INCIDENT HANDLING [Trace: {trace_id}]")
    print(f"ğŸ“� Alert: '{incident_text}'")
    print("-" * 50)

    state: Dict[str, Any] = {
        "raw_text": incident_text,
        "trace_id": trace_id,
    }

    # 1. Intake
    state.update(intake_agent(incident_text, trace_id))

    # 2. Diagnostics
    state = diagnostics_agent(state, trace_id)

    # 3. Remediation
    state = remediation_agent(state, trace_id)

    # 4. Finalize
    duration = round(time.time() - start_time, 2)
    state["duration"] = duration

    # Update metrics
    cli_cmd = state.get("cli_recommendation")
    if cli_cmd and cli_cmd.lower() != "none":
        metrics["successful_resolutions"] += 1

    store_history(state.get("device_ip"), {
        "trace_id": trace_id,
        "root_cause": state.get("root_cause"),
        "timestamp": time.ctime(),
    })

    print("-" * 50)
    print(f"âœ… INCIDENT HANDLED in {duration}s")
    return state


# 10. Evaluation on Synthetic Incidents

test_cases = [
    {
        "text": "Core router 192.168.1.1 is unreachable.",
        "expected_cause_keyword": "port"  # we expect something around port / security / interface
    },
    {
        "text": "Check status of switch 10.0.0.5",
        "expected_cause_keyword": "unknown"  # likely no real issue in mock
    }
]

print("Running evaluation suite...")

results = []
for i, test in enumerate(test_cases, start=1):
    print(f"\nTest Case #{i}: {test['text']}")
    result = run_incident(test["text"])
    cause = str(result.get("root_cause", "None"))
    expected_kw = test["expected_cause_keyword"].lower()

    passed = expected_kw in cause.lower()
    results.append({
        "incident": test["text"],
        "root_cause": cause,
        "expected_keyword": expected_kw,
        "passed": passed,
    })

    if passed:
        print(f"PASSED: Found expected keyword '{expected_kw}' in root_cause='{cause}'")
    else:
        print(f"NOTE: Expected keyword '{expected_kw}', but root_cause was '{cause}'")

print("\nResults:")
for r in results:
    print(r)

print("\nMetrics summary:")
print(metrics)

print("\nDevice history:")
print(json.dumps(device_history, indent=2))


# 11. Interactive Demo

demo_incident = "URGENT: Cannot reach the main distribution switch at 192.168.1.1. Users are complaining."
final_state = run_incident(demo_incident)

print("\n" + "=" * 30)
print("FINAL TICKET OUTPUT")
print("=" * 30)
print(f"Root Cause:      {final_state.get('root_cause')}")
print(f"Suggested Fix:   {final_state.get('cli_recommendation')}")
print(f"Duration:        {final_state.get('duration')} seconds")
print("-" * 30)
print("Escalation Note:")
print(final_state.get('escalation_note'))
print("\nMetrics:")
print(metrics)


