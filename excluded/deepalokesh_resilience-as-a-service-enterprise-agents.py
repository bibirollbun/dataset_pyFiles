# ============================================================
# 0 . API Key Setup
# ============================================================

import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print("âš ï¸� GOOGLE_API_KEY not found. Please configure Kaggle Secret or env var before running agents.")
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# ============================================================
# 1. Imports 
# ============================================================
import os
import json
import uuid
import datetime as dt
from typing import Dict, Any, List, Literal

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner


print("âœ… ADK components imported successfully.")



# ============================================================
# 2. Retry Configuration for Gemini
# ============================================================

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)
print("âœ… Retry configuration defined.")


# ============================================================
# 3. Sample Signals (Raw, as seen at the network boundary)
# ============================================================

sample_signals = [
    # 0: GET API
    {
       
            "signal_type": "api",
            "timestamp": "2025-01-12T17:36:42.751408Z",
            "method": "GET",
            "destination_service": "products-service",
            "endpoint": "/v1/products",
            "query_params": {"q": "milk", "store": "1234"},
            "source_ip": "203.0.113.10",
            "headers": {
                "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
                "Content-Type": "application/json",
            },
            "body": None,
        
    },
    # 1: suspicious POST to inventory-service with empty body
    {
        
            "signal_type": "api",
            "timestamp": "2025-01-12T17:36:42.751408Z",
            "method": "POST",
            "destination_service": "inventory-service",
            "endpoint": "/v2/search",
            "body": None,
            "source_ip": "104.22.16.5",
            "headers": {
                "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
                "Content-Type": "application/json",
            },
        
    },
    # 2: phishing email
    {
       
            "signal_type": "email",
            "timestamp": "2025-01-12T17:36:42.751408Z",
            "from": "it-support@companny-security.com",
            "display_name": "Microsoft Support",
            "to": "employee@company.com",
            "subject": "URGENT: Reset your password now",
            "body": (
                "Your account will be deactivated. Click here to reset: "
                "http://fake-reset.com"
            ),
            "attachments": [],
        
    },
    # 3: kafka inventory update
    {
        
            "signal_type": "kafka",
            "timestamp": "2025-01-12T17:36:42.751408Z",
            "topic": "inventory-updates",
            "key": "store-1234",
            "payload": {"sku": "123456", "delta": 10, "reason": "delivery"},
        
    },
    # 4: malware url / exe download
    {
        
            "signal_type": "malware",
            "timestamp": "2025-01-12T17:36:42.751408Z",
            "url": "http://suspicious-downloads.com/setup.exe",
            "file_name": "invoice_viewer.exe",
            "source_ip": "198.51.100.23",
        
    },
    # 5: network port scan
    {
        
            "signal_type": "network",
            "source_ip": "203.44.12.101",
            "ports_attempted": [22, 80, 443, 8080, 3306],
            "attempt_count": 1731,
        
    },
    # 6: MFA fatigue / account takeover
    {
        
            "signal_type": "auth",
            "user": "employee_102",
            "mfa_push_attempts": 16,
            "result": "rejected",
        
    },
    # 7: unknown protocol
    {
        
            "signal_type": "unknown",
            "raw_data": "???????//////",
            "source_ip": "10.22.82.3",
            "notes": "unrecognized protocol",
       
    },
    # 8: external vendor uploading exe
    {
        
            "signal_type": "file_upload",
            "file_name": "Invoice_Q1.exe",
            "file_type": "application/x-msdownload",
            "size_kb": 1250,
            "uploaded_by": "external_vendor_42",
            "hash": "8d1a23fdba...",
       
    },
]


print(f"âœ… Defined {len(sample_signals)} sample raw signals.")



# ============================================================
# 4. Observability: Decision Log & Helper
# ============================================================

def utc_timestamp():
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


DECISION_LOG: List[Dict[str, Any]] = []
DECISION_COUNTERS: Dict[str, int] = {}

def log_decision(
    agent: str,
    signal_id: str,
    signal_type: str,
    decision: str,
    reason: str,
) -> None:
    """
    Append a decision entry to a global in-memory log.
    This simulates logging/metrics for the capstone.
    """
    entry = {
        "timestamp": utc_timestamp(),
        "agent": agent,
        "signal_id": signal_id,
        "signal_type": signal_type,
        "decision": decision,
        "reason": reason,
    }
    DECISION_LOG.append(entry)
    DECISION_COUNTERS[decision] = DECISION_COUNTERS.get(decision, 0) + 1

    print(
        f"ğŸ“˜ DECISION LOG: agent={agent}, signal_id={signal_id}, "
        f"signal_type={signal_type}, decision={decision}"
    )
print("âœ… Decision log and helper defined.")


# ============================================================
#     MOCK INTERNAL TOOLS (POLICY, SPLUNK-LIKE QUERY)
# ============================================================
# Design decision:
# - We cannot query real Splunk / SIEM.
# - Instead we use small "mock" tools that emulate policy checks and log queries.
# - They use clear docstrings and structured results, as recommended by ADK.

def mock_policy_check(signal_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates an internal organizational policy engine.

    Returns:
        {
          "status": "success",
          "policy_action": "allow" | "block" | "quarantine" | "none",
          "policy_reason": "..."
        }
    """
    # Very simple heuristics purely for demonstration:
    signal_type = signal_metadata.get("signal_type", "unknown")
    m = signal_metadata

    # Block all .exe uploads by policy
    if signal_type == "file_upload":
        name = m.get("file_name", "").lower()
        if name.endswith(".exe"):
            return {
                "status": "success",
                "policy_action": "block",
                "policy_reason": "Executable file uploads are blocked by policy."
            }

    # Block obvious spoofed email by policy
    if signal_type == "email":
        sec = m.get("security", {})
        if sec.get("spf") == "fail" and sec.get("dkim") == "fail":
            return {
                "status": "success",
                "policy_action": "block",
                "policy_reason": "Email fails SPF/DKIM; suspected spoofing."
            }

    # For demo, no strict policy for other cases
    return {
        "status": "success",
        "policy_action": "none",
        "policy_reason": "No explicit policy rule triggered."
    }


def mock_query_splunk_named(query_name: str, time_range: str = "last_24h") -> Dict[str, Any]:
    """
    Simulates a Splunk/SIEM style log query by name.

    Args:
        query_name: Logical name of the query to run (e.g., "ip_reputation", "sender_history").
        time_range: Time window to search.

    Returns:
        {
            "status": "success",
            "data": {...}
        }
    """
    # Hard-coded responses purely illustrative for capstone:
    if query_name == "ip_reputation":
        data = {
            "bad_ips": ["203.44.12.101", "185.54.10.254"],
            "source": "mock_reputation_list"
        }
    elif query_name == "sender_history":
        data = {
            "suspicious_senders": ["support@micosoft-support.com"],
            "source": "mock_email_history"
        }
    else:
        data = {"message": "No data for query_name", "query_name": query_name}

    return {
        "status": "success",
        "data": data,
        "time_range": time_range
    }

print("âœ… Mock internal tools defined.")



# ============================================================
# 4. Normalization Helper
# ============================================================

Decision = Literal["approved", "blocked", "quarantined"]

def make_signal_id() -> str:
    """Generate a stable unique ID for a signal."""
    return str(uuid.uuid4())

def normalize_raw_signal(raw_signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an incoming raw signal into the internal canonical structure.

    Output: normalized_signal:
        {
          'signal_id': <uuid>,
          'signal_type': 'api' | 'email' | 'kafka' | 'malware' | ...,
          'metadata': { ... }  # all original fields captured under metadata
        }
    """
    signal_type = raw_signal.get("signal_type", "unknown")
    signal_id = raw_signal.get("signal_id", make_signal_id())

    # Everything except 'signal_type' & 'signal_id' goes into metadata
    metadata = {
        k: v for k, v in raw_signal.items()
        if k not in ["signal_type", "signal_id"]
    }
    metadata.setdefault("signal_type", signal_type)

    normalized = {
        "signal_id": signal_id,
        "signal_type": signal_type,
        "metadata": metadata,
    }
    return normalized

print("âœ… Signal Normaliser defined.")


# ============================================================
# 5. Internal / Mock Tools
# ============================================================

def policy_engine_check(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock internal zero-trust policy engine.

    Business rules:
    - Block .exe downloads from untrusted domains.
    - Block obvious SQL injection patterns in API bodies.
    - Block emails with obvious phishing text.
    """
    s_type = signal.get("signal_type", "unknown")
    md = signal.get("metadata", {})

    # Malware / download
    if s_type == "malware":
        url = md.get("url", "") or md.get("download_url", "")
        file_name = md.get("file_name", "")
        if url.startswith("http://") and file_name.endswith(".exe"):
            return {
                "status": "blocked",
                "reason": "Executable download from non-HTTPS source is disallowed.",
            }

    # API SQL injection
    if s_type == "api":
        body = md.get("body", {}) or {}
        query = ""
        if isinstance(body, dict):
            query = body.get("query", "") or ""
        if any(token in query.lower() for token in ["drop table", "union select", "--"]):
            return {
                "status": "blocked",
                "reason": "SQL injection pattern detected in API payload.",
            }

    # Email phishing
    if s_type == "email":
        body = (md.get("body") or "").lower()
        if "reset your password" in body and "http://" in body:
            return {
                "status": "blocked",
                "reason": "Suspicious password-reset phishing email detected.",
            }

    return {"status": "allowed", "reason": "No internal policy violation detected."}


def splunk_like_query(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock log query tool to simulate querying SIEM/Splunk.

    For the capstone, we just echo back a small structured result.
    """
    md = signal.get("metadata", {})
    source_ip = md.get("source_ip", "unknown")
    s_type = signal.get("signal_type", "unknown")

    return {
        "status": "success",
        "source_ip": source_ip,
        "signal_type": s_type,
        "prior_incidents_for_ip": 2 if "104.22.16.5" in source_ip else 0,
    }


print("âœ… Mock internal tools defined.")


# ============================================================
# 6. Specialist Threat Detection Agents
# ============================================================

def create_specialist_agent(
    name: str,
    domain_hint: str,
    policy_rules: str,
) -> LlmAgent:
    """
    Factory for specialist LlmAgents with domain-specific hard rules.

    Each specialist:
      - Receives exactly ONE normalized signal:
        { "signal_id": str, "signal_type": str, "metadata": {...} }
      - MUST respond with JSON:
        { "decision": "blocked/approved/quarantined", "reason": "..." }
    """
    return LlmAgent(
        name=name,
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config,
            response_mime_type="application/json",
        ),
        instruction=(
            f"You are `{name}` in the 'AI Agents of Shield' system.\n"
            f"Domain: {domain_hint}.\n\n"
            "INPUT:\n"
            "- You receive exactly ONE normalized signal:\n"
            "  {\"signal_id\": str, \"signal_type\": str, \"metadata\": {...}}\n"
            "- All fields you need are inside `metadata`.\n\n"
            "SECURITY POLICY (hard rules, you MUST follow them exactly):\n"
            f"{policy_rules}\n\n"
            "DECISION LOGIC:\n"
            "- If any hard rule says BLOCK â†’ decision = \"blocked\".\n"
            "- If any hard rule says QUARANTINE â†’ decision = \"quarantined\".\n"
            "- If none of the rules are triggered and the event is clearly benign â†’ decision = \"approved\".\n"
            "- If you are uncertain â†’ prefer \"quarantined\" over \"approved\".\n\n"
            "OUTPUT FORMAT (VERY IMPORTANT):\n"
            "- Respond with ONE line of valid JSON ONLY, no commentary.\n"
            "- The JSON MUST have these keys:\n"
            "    - decision: one of [\"blocked\", \"approved\", \"quarantined\"]\n"
            "    - reason: short human-readable explanation referencing the rule or pattern.\n"
            "Example:\n"
            "{\"decision\": \"blocked\", \"reason\": \"SQL injection pattern detected in query parameter.\"}\n"
        ),
        tools=[policy_engine_check, splunk_like_query],
    )



print("âœ… Specialist threat detection agents created.")


# ============================================================
# 6.1 API threat detection agent
# ============================================================

api_policy_rules = """
You inspect HTTP/HTTPS API traffic. The normalized signal has:
- metadata.method           (e.g. GET, POST, PUT, DELETE)
- metadata.endpoint         (e.g. "/v2/search")
- metadata.destination_service (e.g. "inventory-service")
- metadata.body             (may be null or contain JSON/string)
- metadata.headers          (dict with keys like Authorization, Content-Type)
- metadata.source_ip        (string)
- metadata.timestamp        (ISO 8601 string)
- metadata.signal_type      (should be "api")

HARD RULES (you MUST apply them in order):

1) SQL injection / query abuse:
   - If metadata.body (string or JSON) contains ANY of these fragments (case-insensitive):
       \"DROP TABLE\", \"UNION SELECT\", \" OR 1=1\", \"--\" at end of query, \"xp_\", \"information_schema\"
     â†’ decision = \"blocked\", reason = \"SQL injection pattern detected in API payload.\".

2) Known bad IP or suspicious endpoint:
   - If metadata.source_ip == \"104.22.16.5\" AND metadata.method in [\"POST\", \"PUT\"]:
       â†’ decision = \"blocked\", reason = \"Known suspicious IP performing write call.\".
   - If metadata.endpoint in [\"/admin\", \"/internal/backup\", \"/debug\"] AND method != \"GET\":
       â†’ decision = \"blocked\", reason = \"Write access to sensitive admin endpoint.\".

3) Suspicious empty write:
   - If metadata.method in [\"POST\", \"PUT\", \"PATCH\"] AND metadata.body is null
     AND endpoint matches pattern like \"/v2/search\" OR \"/login\" OR \"/transfer\":
       â†’ decision = \"quarantined\", reason = \"Empty write request to sensitive endpoint; needs investigation.\".

4) Executable / binary download:
   - If endpoint or query parameters indicate a download of \".exe\", \".bat\", \".ps1\" or similar,
     OR Content-Type suggests a binary/executable, and destination_service is internal:
       â†’ decision = \"blocked\", reason = \"Potential malware download through API.\".

5) Default Safe:
   - If request is simple GET on a non-sensitive endpoint (e.g. /v1/products, /health, /ping),
     with no suspicious headers, and no rules 1-4 are triggered:
       â†’ decision = \"approved\", reason = \"Routine API request, no suspicious patterns.\".

6) Fallback:
   - If something feels unusual but does not match rules 1-4 explicitly:
       â†’ decision = \"quarantined\", reason = \"Unclear risk; quarantined for manual review.\".
"""

api_threat_detection_agent = create_specialist_agent(
    "api_threat_detection_agent",
    "Inspection of HTTP/HTTPS API requests for injection, abuse, and exfiltration.",
    api_policy_rules,
)



# ============================================================
# 6.2 Email threat detection agent
# ============================================================

email_policy_rules = """
You inspect INBOUND EMAIL security events.

Normalized signal structure:
- metadata.from             (e.g. "it-support@companny-security.com")
- metadata.display_name     (e.g. "Microsoft Support")
- metadata.to               (e.g. "employee@company.com")
- metadata.subject          (e.g. "URGENT: Reset your password now")
- metadata.body             (full email body as text)
- metadata.attachments      (list of attachment names or descriptors)
- metadata.timestamp        (ISO 8601 string)
- metadata.signal_type      (should be "email")

HARD RULES (apply top-down):

1) Clear phishing indicators â†’ BLOCK:
   Block the email if ANY of the following are true:
   - Subject or body has URGENCY + ACTION combo, e.g. patterns like:
       "URGENT", "IMMEDIATE ACTION", "reset your password now",
       "your account will be deactivated", "verify now" (case-insensitive).
   - The email body contains a CLICKABLE LINK (e.g. "http://", "https://")
     AND that link points to a non-corporate or suspicious domain,
     especially if it is not the legitimate vendor domain
     (e.g. "fake-reset.com" instead of "microsoft.com").
   - The FROM domain looks like a TYPOSQUATTED version of a known brand,
     e.g. "companny-security.com" vs "company-security.com"
     (extra letter, missing letter, swapped letters).
   - The FROM display name pretends to be a well-known vendor
     (Microsoft, Google, Bank, HR) but the FROM domain is different from
     the legitimate vendor domain.

   If any of the above hold:
     â†’ decision = "blocked"
     â†’ reason = "Phishing indicators: urgent language and/or suspicious link/sender domain."

2) Malicious attachment indicators â†’ BLOCK:
   - If there is any attachment whose file name ends with extensions like:
       ".exe", ".scr", ".js", ".vbs", ".bat", ".cmd", ".ps1", ".hta",
       or compressed archives like ".zip", ".rar" that are not expected
       from this sender.
   - Especially if the sender is external or unknown.

   Then:
     â†’ decision = "blocked"
     â†’ reason = "Potentially malicious attachment detected."

3) Suspicious but not clearly malicious â†’ QUARANTINED:
   - If the email has unusual wording, unexpected sender, or generic
     "security" notifications, but without clear phishing indicators from rule 1,
     and no known-safe patterns.
   - If the domain is unknown but not obviously typosquatting,
     and the email requests sensitive information (passwords, OTPs, banking).

   Then:
     â†’ decision = "quarantined"
     â†’ reason = "Suspicious email content; quarantined for manual review."

4) Clearly safe internal communications â†’ APPROVED:
   Approve if ALL are true:
   - Sender domain is an internal, known-good corporate domain.
   - No urgent scare tactics, no suspicious links, no unexpected attachments.
   - Subject and body look like routine notifications (status updates, meeting invites, etc.).

   Then:
     â†’ decision = "approved"
     â†’ reason = "Routine internal email with no suspicious indicators."

5) Fallback:
   - If you cannot confidently classify the email as safe or malicious:
       â†’ decision = "quarantined"
       â†’ reason = "Unclear risk; quarantined for manual review."
"""

email_threat_detection_agent = create_specialist_agent(
    "email_threat_detection_agent",
    "Inspection of inbound emails for phishing, social engineering, and malware links.",
    email_policy_rules,
)



# ============================================================
# 6.3 Kafka / event-stream threat detection agent
# ============================================================

kafka_policy_rules = """
You inspect KAFKA / EVENT-STREAM messages.

Normalized signal structure:
- metadata.topic        (e.g. "inventory-updates")
- metadata.key          (e.g. "store-1234")
- metadata.payload      (JSON/dict with business fields)
- metadata.timestamp    (ISO 8601 string)
- metadata.signal_type  (should be "kafka")

Think like a security-aware SRE + data engineer.

HARD RULES:

1) Obvious schema / payload abuse â†’ BLOCK:
   - If payload contains unexpected executable-like strings or URLs
     pointing to suspicious domains.
   - If payload includes raw SQL/OS commands ("DROP TABLE", "rm -rf /",
     "UNION SELECT", etc).
   - If payload fields that should be numeric contain strange text instead,
     suggesting injection or exfiltration.

   Then:
     â†’ decision = "blocked"
     â†’ reason = "Payload contains suspicious or injection-like content."

2) Fraud / corruption patterns â†’ QUARANTINED or BLOCKED:
   For inventory or financial topics such as "inventory-updates", "payments", etc:
   - If payload.delta or similar numeric field is extremely large in magnitude
     compared to typical ranges (e.g. +/- 1e6).
   - If there are frequent negative adjustments without matching reasons.
   - If topic is critical (payments, refunds, inventory) and payload structure
     deviates strongly from known schema (missing required keys, unexpected new keys).

   Default:
     â†’ decision = "quarantined"
     â†’ reason = "Anomalous payload for business-critical topic; potential fraud or data corruption."
   If pattern is clearly malicious / impossible:
     â†’ decision = "blocked"
     â†’ reason = "Clearly invalid or malicious payload for business-critical topic."

3) Routine operational updates â†’ APPROVED:
   Approve if ALL are true:
   - Topic is known operational topic (e.g. "inventory-updates",
     "order-events", "delivery-events").
   - Payload structure looks normal: expected keys present,
     numeric values within reasonable ranges (e.g. small deltas like +10, -2).
   - No suspicious strings (no SQL, shell commands, malware URLs).

   Then:
     â†’ decision = "approved"
     â†’ reason = "Routine Kafka message with expected payload and ranges."

4) Unknown topic or schema drift â†’ QUARANTINED:
   - If topic is unknown or not documented.
   - If payload structure or key-set suddenly changes without explanation.
   - If there is any mix of unexpected binary blobs or heavily encoded data.

   Then:
     â†’ decision = "quarantined"
     â†’ reason = "Unknown or drifting schema; quarantined for investigation."

5) Fallback:
   - When in doubt and there is ANY sign of risk:
       â†’ decision = "quarantined"
       â†’ reason = "Unclear risk in Kafka payload; quarantined by zero-trust policy."
"""

kafka_threat_detection_agent = create_specialist_agent(
    "kafka_threat_detection_agent",
    "Inspection of Kafka messages for strange payloads, schema abuse, and fraud.",
    kafka_policy_rules,
)



# ============================================================
# 6.4 Malware / download & file-transfer agent
# ============================================================

malware_policy_rules = """
You inspect DOWNLOAD and FILE-TRANSFER security events.

Normalized signal structure may include:
- metadata.url             (for downloads)
- metadata.file_name       (e.g. "invoice_viewer.exe", "Invoice_Q1.exe")
- metadata.file_type       (e.g. "application/x-msdownload", "application/pdf")
- metadata.size_kb         (integer size in kilobytes)
- metadata.source_ip       (for downloads)
- metadata.uploaded_by     (for file uploads, e.g. "external_vendor_42")
- metadata.hash            (file hash if provided)
- metadata.signal_type     (e.g. "malware", "file_upload")
- metadata.timestamp       (ISO 8601 string)

HARD RULES:

1) Executable from untrusted origin â†’ BLOCK:
   - If file_name ends with executable-like extensions:
       ".exe", ".dll", ".scr", ".msi", ".bat", ".cmd", ".ps1", ".jar", ".apk"
     OR file_type suggests executable/binary like:
       "application/x-msdownload", "application/x-dosexec".
   AND one of:
     - url domain is NOT a known trusted vendor or repository.
     - uploaded_by is external/non-employee identity (e.g. starts with "external_").
     - source_ip is external / internet-facing range.

   Then:
     â†’ decision = "blocked"
     â†’ reason = "Executable file from untrusted origin; blocked as potential malware."

2) Downloads from known suspicious / generic domains â†’ BLOCK:
   - If url contains suspicious host patterns such as:
       "suspicious", "downloadz", "cracks", "keygen", "pirated", etc.
   - If domain is very generic file host with weak reputation and file is executable.

   Then:
     â†’ decision = "blocked"
     â†’ reason = "Download from suspicious domain with potentially malicious payload."

3) Large executable upload from external entity â†’ BLOCK:
   - If signal_type == "file_upload"
   - AND file is executable-like (per rule 1)
   - AND uploaded_by is an external party (vendor, contractor, unknown).

   Then:
     â†’ decision = "blocked"
     â†’ reason = "Executable uploaded by external party; blocked for safety."

4) Non-executable documents â†’ APPROVED (with minor caveats):
   - If file is a common document type (e.g. .pdf, .docx, .xlsx, .pptx, .txt)
   - AND url and uploaded_by look consistent with normal business flows
     (internal tools, known vendors, etc).
   - AND no obviously suspicious domain patterns.

   Then:
     â†’ decision = "approved"
     â†’ reason = "Routine document/file transfer; no suspicious indicators."

5) Suspicious but not clearly malicious â†’ QUARANTINED:
   - If type is unusual or uncommon (e.g. .iso, .img, unknown MIME type),
     but not clearly malicious.
   - If file size is extremely large or extremely small vs normal expectations.
   - If metadata conflicts (e.g. executable file type but document-sounding name).

   Then:
     â†’ decision = "quarantined"
     â†’ reason = "Unusual file characteristics; quarantined for manual malware analysis."

6) Fallback:
   - When in doubt, prefer BLOCK for executables and QUARANTINE for others.
"""

malware_threat_detection_agent = create_specialist_agent(
    "malware_threat_detection_agent",
    "Inspection of download and file-transfer events for malware-like patterns.",
    malware_policy_rules,
)



# ============================================================
# 6.5 Identity / auth threat detection agent
# ============================================================

identity_policy_rules = """
You inspect AUTHENTICATION and IDENTITY events.

Normalized signal structure:
- metadata.user              (e.g. "employee_102")
- metadata.mfa_push_attempts (integer count)
- metadata.result            (e.g. "success", "rejected", "failed")
- metadata.source_ip         (optional, if present)
- metadata.timestamp         (ISO 8601 string)
- metadata.signal_type       (should be "auth")

Think like an IAM security engineer.

HARD RULES:

1) MFA push bombing / account takeover â†’ BLOCK:
   - If mfa_push_attempts is very high (e.g. >= 10)
   - AND result is predominantly "rejected" or "failed"
   - This is typical of MFA fatigue attacks / takeover attempts.

   Then:
     â†’ decision = "blocked"
     â†’ reason = "High number of rejected MFA pushes; likely account takeover attempt."

2) Repeated auth failures from unusual IPs or locations â†’ BLOCK:
   - If there are multiple failed logins over a short time window
     from unfamiliar or risky IP ranges (if indicated in metadata or tools).
   - If result is "rejected" and metadata suggests impossible travel
     or anomalous behavior (e.g. known internal identity logging in
     from unexpected geography).

   Then:
     â†’ decision = "blocked"
     â†’ reason = "Repeated auth failures from suspicious context; blocked for safety."

3) Suspicious but lower volume â†’ QUARANTINED:
   - If mfa_push_attempts is medium (e.g. between 3 and 9),
     especially if user behavior is unusual, but not clearly malicious.
   - If there is a mix of successes and failures that looks odd.

   Then:
     â†’ decision = "quarantined"
     â†’ reason = "Unusual MFA behavior; quarantined for security review."

4) Normal auth usage â†’ APPROVED:
   - Single or low number of MFA pushes.
   - Successful authentication with no anomalies.
   - Known user behavior pattern (if available from tools or hints).

   Then:
     â†’ decision = "approved"
     â†’ reason = "Routine authentication behavior; no suspicious indicators."

5) Fallback:
   - If you are unsure but there is non-trivial risk:
       â†’ decision = "quarantined"
       â†’ reason = "Ambiguous identity signal; quarantined under zero-trust."
"""

identity_threat_detection_agent = create_specialist_agent(
    "identity_threat_detection_agent",
    "Inspection of authentication and identity events.",
    identity_policy_rules,
)



# ============================================================
# 6.6 Misconfig / unknown-signal agent
# ============================================================

misconfig_policy_rules = """
You inspect UNKNOWN / MISCONFIGURED / UNCLASSIFIED signals.

Normalized signal structure:
- metadata.signal_type   (often "unknown" or malformed)
- metadata.raw_data      (raw payload, e.g. "???????//////")
- metadata.source_ip     (if present)
- metadata.notes         (e.g. "unrecognized protocol")
- metadata.timestamp     (if present)

Your job is resilience + safety: handle what other agents cannot classify.

HARD RULES:

1) Unrecognized protocol or garbage payload â†’ QUARANTINED:
   - If signal_type is "unknown" or clearly not mapped to any domain.
   - If raw_data looks like random bytes, gibberish, or an unrecognized
     protocol string (e.g. "???????//////").
   - If notes explicitly say "unrecognized protocol", "decoder error",
     "parse failure", etc.

   Then:
     â†’ decision = "quarantined"
     â†’ reason = "Unrecognized or malformed protocol; quarantined for manual triage."

2) Clear evidence of misconfiguration (but not malicious) â†’ QUARANTINED:
   - If signal appears to be legitimate system data being sent
     to the wrong channel or with wrong headers.
   - If configuration indicators suggest a misrouted or malformed event.

   Then:
     â†’ decision = "quarantined"
     â†’ reason = "Likely system misconfiguration; quarantined until fixed."

3) Clearly malicious but unknown category â†’ BLOCK:
   - If raw_data or notes indicate active exploitation, exfiltration,
     or known bad patterns, but the signal_type is still "unknown".
   - Examples: embedded exploit signatures, blacklisted IP noted in metadata.

   Then:
     â†’ decision = "blocked"
     â†’ reason = "Unknown signal with malicious indicators; blocked for safety."

4) Benign diagnostic / noise â†’ APPROVED (rare):
   - Only approve if metadata explicitly indicates test traffic,
     health-check pings, or intentionally fuzzed inputs by security teams
     AND no malicious content is present.

   Then:
     â†’ decision = "approved"
     â†’ reason = "Recognized diagnostic/test signal; no action needed."

5) Fallback:
   - When in doubt for unknown signal types:
       â†’ decision = "quarantined"
       â†’ reason = "Unknown/ambiguous signal; quarantined by default."
"""

misconfig_threat_detection_agent = create_specialist_agent(
    "misconfig_threat_detection_agent",
    "Inspection of unknown/misconfigured signals and resilience issues.",
    misconfig_policy_rules,
)




# ============================================================
# 6.7 Network threat detection agent
# ============================================================

network_policy_rules = """
You inspect NETWORK-LEVEL security events.

Normalized signal structure:
- metadata.source_ip        (e.g. "203.44.12.101")
- metadata.ports_attempted  (list of integers, e.g. [22, 80, 443, 8080, 3306])
- metadata.attempt_count    (integer count of connection attempts)
- metadata.timestamp        (optional)
- metadata.signal_type      (should be "network")

Think like a SOC analyst + network security engineer.

HARD RULES:

1) Clear port scan / brute-force behavior â†’ BLOCK:
   - If attempt_count is very high (e.g. > 100) AND
   - ports_attempted includes multiple distinct ports, especially a mix of:
       22 (SSH), 23 (Telnet), 3306 (MySQL), 3389 (RDP), 445 (SMB),
       plus common web ports (80, 443, 8080).
   This pattern is typical of scanning or enumeration.

   Then:
     â†’ decision = "blocked"
     â†’ reason = "Port scanning / brute-force pattern detected from source IP."

2) Repeated failed connections / anomalies from known bad IP ranges â†’ BLOCK:
   - If source_ip is in or similar to known malicious ranges (if indicated in metadata
     or known from tools).
   - If there are repeated connection anomalies with high attempt_count
     over a short period.

   Then:
     â†’ decision = "blocked"
     â†’ reason = "Repeated anomalous network activity from suspicious IP."

3) Elevated but not clearly malicious â†’ QUARANTINED:
   - If attempt_count is moderately high (e.g. between 20 and 100)
     with multiple ports, but pattern is not clearly a scan.
   - Or if there are unusual ports accessed but no clear signature.

   Then:
     â†’ decision = "quarantined"
     â†’ reason = "Unusual network activity; quarantined for deeper investigation."

4) Normal traffic â†’ APPROVED:
   - If ports_attempted consist only of standard application ports
     (e.g. 80/443/8080) with low attempt_count (e.g. < 20),
     and there is no sign of scanning across many ports.
   - If the pattern looks like routine user or service traffic.

   Then:
     â†’ decision = "approved"
     â†’ reason = "Routine network traffic; no scan or brute-force indicators."

5) Fallback:
   - When uncertain but the risk is non-zero:
       â†’ decision = "quarantined"
       â†’ reason = "Ambiguous network pattern; quarantined by zero-trust policy."
"""

network_threat_detection_agent = create_specialist_agent(
    "network_threat_detection_agent",
    "Inspection of network-level traffic and scans.",
    network_policy_rules,
)



from google.adk.runners import InMemoryRunner

async def run_specialist_and_parse(specialist_agent, normalized_signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a specialist agent on ONE normalized_signal and returns a clean dict:
      {
        "decision": "...",
        "reason": "...",
        "signal_id": "...",
        "specialist_agent": "..."
      }

    This centralises:
    - running the agent with InMemoryRunner
    - extracting its final text
    - parsing JSON safely
    """
    runner = InMemoryRunner(agent=specialist_agent)

    prompt = (
        "You are a specialist threat-detection agent.\n"
        "You must inspect the single normalized signal below and respond with ONLY JSON.\n"
        "The JSON MUST have keys: decision (blocked/approved/quarantined) and reason.\n\n"
        f"{json.dumps(normalized_signal)}"
    )

    # Get full trace of events
    events = await runner.run_debug(prompt)

    specialist_text = ""
    for ev in events:
        if ev.author == specialist_agent.name and ev.content and ev.content.parts:
            for part in ev.content.parts:
                if getattr(part, "text", None):
                    specialist_text = part.text.strip()

    # Default result
    base = {
        "signal_id": normalized_signal.get("signal_id"),
        "specialist_agent": specialist_agent.name,
        "decision": "unknown",
        "reason": "specialist did not return valid JSON",
    }

    if not specialist_text:
        return base

    try:
        parsed = json.loads(specialist_text)
        if not isinstance(parsed, dict):
            raise ValueError("Non-object JSON")

        base["decision"] = parsed.get("decision", base["decision"])
        base["reason"] = parsed.get("reason", base["reason"])
        return base

    except Exception:
        base["reason"] = f"Failed to parse specialist JSON. Raw: {specialist_text[:200]}"
        return base



# ============================================================
# 7. Orchestrator Tool & Agent
# ============================================================

async def route_to_specialist(normalized_signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool for the orchestrator_agent.

    Responsibilities:
    - Look at normalized_signal['signal_type'].
    - Route to the corresponding specialist agent.
    - Return a consolidated decision payload.
    """
    s_type = normalized_signal.get("signal_type", "unknown")

    routing_table = {
        "api": api_threat_detection_agent,
        "email": email_threat_detection_agent,
        "kafka": kafka_threat_detection_agent,
        "malware": malware_threat_detection_agent,
        "file_upload": malware_threat_detection_agent,
        "network": network_threat_detection_agent,
        "auth": identity_threat_detection_agent,
        "unknown": misconfig_threat_detection_agent,
    }

    specialist_agent = routing_table.get(s_type, misconfig_threat_detection_agent)

    # ğŸ”‘ Call the shared helper to run + parse specialist output
    specialist_result = await run_specialist_and_parse(specialist_agent, normalized_signal)

    return {
        "status": "success",
        "signal_type": s_type,
        "specialist_agent": specialist_result["specialist_agent"],
        "decision": specialist_result["decision"],
        "reason": specialist_result["reason"],
    }



orchestrator_agent = LlmAgent(
    name="orchestrator_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    instruction=(
        "You are the orchestrator in the 'Resilience as a Service - AI Agents of Shield' system.\n"
        "You NEVER analyze raw signals directly. Instead:\n"
        "  1. You receive a normalized_signal JSON.\n"
        "  2. You MUST call the `route_to_specialist` tool with that JSON.\n"
        "  3. Once the tool returns, you respond with ONLY the tool's JSON response.\n"
        "Do NOT add extra commentary. Output must be a single JSON object.\n"
    ),
    tools=[route_to_specialist],
)

print("âœ… Orchestrator agent defined.")


# ============================================================
# 8. Listener Tool & Agent (Network Boundary)
# ============================================================

async def forward_to_orchestrator(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Listener tool:
    - Takes a raw signal from the network boundary.
    - Normalizes it.
    - Calls the orchestrator_agent.
    - Returns:
        {
          "normalized_signal": {...},
          "decision": "...",
          "reason": "...",
          "specialist_agent": "..."
        }
    """
    normalized = normalize_raw_signal(signal)
    s_type = normalized["signal_type"]
    signal_id = normalized["signal_id"]

    runner = InMemoryRunner(agent=orchestrator_agent)

    prompt = (
        "You are the orchestrator. You have a tool called `route_to_specialist`.\n"
        "Use that tool with the normalized_signal below, then respond ONLY with "
        "the JSON it returns (no extra text).\n\n"
        f"{json.dumps({'normalized_signal': normalized})}"
    )

    events = await runner.run_debug(prompt)

    # ğŸ”� NEW: extract the tool response directly from function_response
    tool_payload = None
    for ev in events:
        if not ev.content or not ev.content.parts:
            continue
        for part in ev.content.parts:
            fr = getattr(part, "function_response", None)
            if fr and fr.name == "route_to_specialist":
                tool_payload = fr.response

    decision = "unknown"
    reason = "orchestrator did not return valid JSON"
    specialist_agent_name = "unknown"

    if tool_payload is not None:
        decision = tool_payload.get("decision", decision)
        reason = tool_payload.get("reason", reason)
        specialist_agent_name = tool_payload.get("specialist_agent", specialist_agent_name)
    else:
        # Optional: debug printing
        print("âš ï¸� No function_response found for route_to_specialist")

    # Log decision for observability
    log_decision(
        agent="orchestrator_agent",
        signal_id=signal_id,
        signal_type=s_type,
        decision=decision,
        reason=reason,
    )

    return {
        "normalized_signal": normalized,
        "decision": decision,
        "reason": reason,
        "specialist_agent": specialist_agent_name,
    }



listener_agent = LlmAgent(
    name="listener_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    instruction=(
        "You sit at the network boundary as a listener.\n"
        "You receive a raw signal as plain JSON.\n"
        "RULES:\n"
        "  - You MUST call the `forward_to_orchestrator` tool with that raw signal.\n"
        "  - You MUST then respond ONLY with the JSON returned by that tool.\n"
        "  - You MUST NOT try to interpret or block the signal yourself.\n"
        "Output format: a single JSON object, no explanations.\n"
    ),
    tools=[forward_to_orchestrator],
)

print("âœ… Listener agent defined.")



async def run_listener_on_signal(raw_signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    raw_signal -> listener_agent -> orchestrator_agent -> specialist_agent

    Preferred path:
      - listener calls forward_to_orchestrator(tool)
      - we read the function_response and return that dict

    Fallback path (if listener misbehaves):
      - we call forward_to_orchestrator(raw_signal) directly
    """

    runner = InMemoryRunner(agent=listener_agent)

    prompt = (
        "This is a new incoming network boundary signal.\n"
        "You MUST call your `forward_to_orchestrator` tool with the raw signal below.\n"
        "After the tool returns, you MUST respond with ONLY the tool's JSON response.\n\n"
        f"{json.dumps(raw_signal)}"
    )

    events = await runner.run_debug(prompt)

    tool_payload = None
    listener_text = None

    for ev in events:
        if not ev.content or not ev.content.parts:
            continue

        for part in ev.content.parts:
            # 1) Preferred: function_response from forward_to_orchestrator
            fr = getattr(part, "function_response", None)
            if fr and fr.name == "forward_to_orchestrator":
                tool_payload = fr.response

            # 2) Also capture any final text the listener printed
            text = getattr(part, "text", None)
            if text and ev.author == listener_agent.name:
                listener_text = text.strip()

    # âœ… If we got the tool payload, just return it
    if tool_payload is not None and isinstance(tool_payload, dict):
        return tool_payload

    # ğŸ”� Fallback #1: try to parse any JSON text from the listener
    if listener_text:
        try:
            parsed = json.loads(listener_text)
            if isinstance(parsed, dict):
                return parsed
            else:
                return {
                    "decision": "unknown",
                    "reason": f"Listener returned non-object JSON: {listener_text[:200]}",
                    "timestamp": utc_timestamp(),
                }
        except Exception:
            # we'll try direct tool call next
            pass

    # ğŸ”� Fallback #2: bypass listener and call forward_to_orchestrator directly
    try:
        direct_result = await forward_to_orchestrator(raw_signal)
        # You could also log here that you used the fallback
        # print("âš ï¸� Listener skipped tool; used direct forward_to_orchestrator fallback.")
        return direct_result
    except Exception as e:
        # Final safety net
        return {
            "decision": "unknown",
            "reason": f"No function_response and direct tool call failed: {e}",
            "timestamp": utc_timestamp(),
        }




print("âœ… Helper run_listener_on_signal defined.") 


# ============================================================
# 10. Demo: Run Pipeline on One Example Signal
# ============================================================

demo_raw_signal = sample_signals[1]  # SQL injection sample

print("ğŸš€ Running demo on SQL injection sample (index 1)...")

demo_result = await run_listener_on_signal(demo_raw_signal)

print("\nğŸ”� Listener structured response:")
print(json.dumps(demo_result, indent=2))

print("\nğŸ“Š Decision counters so far:", DECISION_COUNTERS)


# âœ… Keep your labeled_signals as-is
labeled_signals = [
    {"raw_signal": sample_signals[0], "expected_decision": "blocked"},
    {"raw_signal": sample_signals[1], "expected_decision": "blocked"},
    {"raw_signal": sample_signals[2], "expected_decision": "blocked"},
    {"raw_signal": sample_signals[3], "expected_decision": "approved"},
    {"raw_signal": sample_signals[4], "expected_decision": "blocked"},
    {"raw_signal": sample_signals[5], "expected_decision": "blocked"},
    {"raw_signal": sample_signals[6], "expected_decision": "blocked"},
    {"raw_signal": sample_signals[7], "expected_decision": "quarantined"},
    {"raw_signal": sample_signals[8], "expected_decision": "blocked"},
]


async def evaluate_system(labeled: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluation using the full LLM pipeline:
    raw_signal -> listener_agent -> orchestrator_agent -> specialist_agent.

    Calls run_listener_on_signal() for each sample and compares the decision.
    """
    results = []
    correct = 0

    for idx, item in enumerate(labeled):
        raw_signal = item["raw_signal"]
        expected = item["expected_decision"]

        print(f"\nğŸ§ª Evaluating sample #{idx} (expected={expected})...")

        # ğŸ”� Run the full listenerâ†’orchestratorâ†’specialist pipeline
        out = await run_listener_on_signal(raw_signal)

        predicted = out.get("decision", "unknown")
        normalized = out.get("normalized_signal", {})
        reason = out.get("reason", "")

        if predicted == expected:
            correct += 1

        results.append(
            {
                "index": idx,
                "expected": expected,
                "predicted": predicted,
                "normalized_signal": normalized,
                "reason": reason,
            }
        )

    accuracy = correct / len(labeled) if labeled else 0.0

    return {
        "total": len(labeled),
        "correct": correct,
        "accuracy": accuracy,
        "details": results,
    }


# âœ… Run the evaluation
eval_results = await evaluate_system(labeled_signals)

print("\nğŸ“Š EVALUATION SUMMARY")
print("Total samples:", eval_results["total"])
print("Correct predictions:", eval_results["correct"])
print("Accuracy:", round(eval_results["accuracy"] * 100, 2), "%")

print("\nğŸ”� Detailed results:")
for r in eval_results["details"]:
    print(r)





