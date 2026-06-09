# Install required packages for demos
%pip install google-genai pydantic requests --quiet

import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal

# =============================================================================
# MCP Envelope Schema (Model Context Protocol)
# Based on: https://modelcontextprotocol.io/docs/concepts/tools
# =============================================================================

class MCPToolInvocation(BaseModel):
    """
    MCP-compatible tool invocation envelope.
    
    This follows the Model Context Protocol specification for tool calls,
    enabling structured communication between AI agents and tools.
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tool_name: str = Field(..., description="Name of the tool to invoke")
    inputs: dict[str, Any] = Field(default_factory=dict)
    from_agent: str = Field(default="orchestrator")
    to_agent: Optional[str] = Field(default=None)
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    def model_post_init(self, __context):
        if not self.metadata.get("timestamp"):
            self.metadata["timestamp"] = datetime.utcnow().isoformat() + "Z"

class MCPToolResponse(BaseModel):
    """MCP tool response envelope with status and optional error."""
    id: str
    status: Literal["ok", "error"] = "ok"
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    trace_id: str
    elapsed_ms: Optional[float] = None

# Demo: Create MCP envelope for triage tool invocation
envelope = MCPToolInvocation(
    tool_name="triage",
    inputs={
        "features": {
            "failed_logins_last_hour": 50,
            "suspicious_file_activity": True,
            "rare_outgoing_connection": True,
            "source_ip": "192.168.1.100"
        }
    },
    from_agent="user_client",
    to_agent="triage_agent"
)

print("ğŸ“¦ MCP Tool Invocation Envelope:")
print(envelope.model_dump_json(indent=2))


from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
from datetime import datetime
import uuid

# =============================================================================
# A2A Message Protocol (Agent-to-Agent)
# Based on: https://a2a-protocol.org/latest/
# =============================================================================

class A2AMessage(BaseModel):
    """
    Agent-to-Agent communication message.
    
    Implements the A2A protocol specification for inter-agent messaging
    with full trace context for observability.
    
    References:
    - https://a2a-protocol.org/latest/
    - https://google.github.io/A2A/#section-2-agent-card
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    from_agent: str = Field(..., description="Sending agent identifier")
    to_agent: str = Field(..., description="Receiving agent identifier")
    message_type: Literal["request", "response", "event", "error"] = "request"
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: Optional[str] = Field(default=None, description="Parent message for threading")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() + "Z"}

class AgentCard(BaseModel):
    """A2A Agent Card - describes agent capabilities."""
    name: str
    description: str
    version: str = "1.0.0"
    capabilities: list[str] = Field(default_factory=list)
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None

# Define our agent cards
AGENTS = {
    "triage_agent": AgentCard(
        name="Triage Agent",
        description="Scores incident severity using weighted heuristics",
        capabilities=["incident_scoring", "severity_classification"]
    ),
    "explain_agent": AgentCard(
        name="Explain Agent", 
        description="Generates LLM-powered explanations of triage decisions",
        capabilities=["natural_language_explanation", "gemini_integration"]
    ),
    "runbook_agent": AgentCard(
        name="Runbook Agent",
        description="RAG-enhanced runbook generation from vector database",
        capabilities=["rag_retrieval", "runbook_generation", "pgvector"]
    )
}

# Demo: Create A2A message
trace_id = uuid.uuid4().hex
request_msg = A2AMessage(
    from_agent="orchestrator",
    to_agent="triage_agent",
    message_type="request",
    payload={"features": {"failed_logins": 23, "suspicious_ip": True}},
    trace_id=trace_id
)

# Simulate response
response_msg = A2AMessage(
    from_agent="triage_agent",
    to_agent="orchestrator",
    message_type="response",
    payload={"label": "HIGH", "score": 8, "contribs": [("failed_logins", 3)]},
    trace_id=trace_id,
    parent_id=request_msg.id
)

print("ğŸ”„ A2A Message Exchange:")
print("\nğŸ“¤ Request:")
print(request_msg.model_dump_json(indent=2))
print("\nğŸ“¥ Response:")
print(response_msg.model_dump_json(indent=2))
print("\nğŸ“‡ Agent Card:")
print(AGENTS["triage_agent"].model_dump_json(indent=2))


from pydantic import BaseModel, Field
from typing import Literal
from IPython.display import HTML, display

# =============================================================================
# Triage Agent with Pydantic Models
# Production version uses weighted scoring heuristics
# =============================================================================

class TriageResult(BaseModel):
    """Structured triage output with full traceability."""
    label: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    score: int = Field(..., ge=0, le=20)
    contribs: list[tuple[str, int]] = Field(default_factory=list)
    
# Feature weights matching our backend implementation
FEATURE_WEIGHTS = {
    "failed_logins_last_hour": (10, 3),    # (threshold, weight)
    "process_spawn_count": (30, 2),
    "suspicious_file_activity": (True, 2),
    "rare_outgoing_connection": (True, 2),
    "privilege_escalation": (True, 3),
    "data_exfil_bytes": (1000000, 3),
}

def score_incident(features: dict) -> TriageResult:
    """
    Score an incident using weighted heuristics.
    
    This is a simplified version of our backend's triage logic.
    Real implementation in: backend/app/agents/triage.py
    """
    score = 0
    contribs: list[tuple[str, int]] = []
    
    for feature, (threshold, weight) in FEATURE_WEIGHTS.items():
        value = features.get(feature)
        if value is None:
            continue
            
        triggered = False
        if isinstance(threshold, bool):
            triggered = bool(value) == threshold
        elif isinstance(threshold, (int, float)):
            triggered = value >= threshold
            
        if triggered:
            score += weight
            contribs.append((feature, weight))
    
    # Determine severity label
    if score >= 8:
        label = "CRITICAL"
    elif score >= 6:
        label = "HIGH"
    elif score >= 3:
        label = "MEDIUM"
    else:
        label = "LOW"
    
    return TriageResult(label=label, score=score, contribs=contribs)

# Demo: Triage a suspicious incident
test_features = {
    "failed_logins_last_hour": 50,
    "process_spawn_count": 45,
    "suspicious_file_activity": True,
    "rare_outgoing_connection": True
}

result = score_incident(test_features)

# Render as styled HTML card
severity_colors = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#fd7e14", "CRITICAL": "#dc3545"}
color = severity_colors[result.label]

html = f"""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            max-width: 500px; border: 2px solid {color}; border-radius: 12px; padding: 20px; 
            background: linear-gradient(135deg, {color}15 0%, white 100%);">
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span style="font-size: 32px; margin-right: 10px;">ğŸš¨</span>
        <div>
            <h3 style="margin: 0; color: #333;">Triage Result</h3>
            <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 20px; 
                         font-weight: bold; font-size: 14px;">{result.label}</span>
        </div>
    </div>
    <div style="background: white; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
        <div style="font-size: 48px; font-weight: bold; color: {color}; text-align: center;">
            {result.score}
        </div>
        <div style="text-align: center; color: #666;">Risk Score</div>
    </div>
    <div style="background: white; border-radius: 8px; padding: 15px;">
        <h4 style="margin: 0 0 10px 0; color: #333;">Contributing Factors</h4>
        {''.join(f'<div style="display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #eee;"><span>{feat}</span><span style="color: {color}; font-weight: bold;">+{pts}</span></div>' for feat, pts in result.contribs)}
    </div>
</div>
"""
display(HTML(html))


from pydantic import BaseModel, Field
from typing import Literal, Optional
from IPython.display import HTML, display
import hashlib

# =============================================================================
# RAG-Enhanced Runbook Generation
# Uses embeddings for semantic search (pgvector in production)
# =============================================================================

class RunbookStep(BaseModel):
    """A single remediation step with risk assessment."""
    step: str
    why: str
    risk: Literal["low", "medium", "high"]

class RunbookResponse(BaseModel):
    """Generated runbook with source attribution."""
    runbook: list[RunbookStep]
    source: Literal["rag", "llm", "stub", "rag_stub"]

# Simulated vector embeddings (768 dimensions like text-embedding-004)
def generate_embedding(text: str, dim: int = 768) -> list[float]:
    """
    Generate deterministic pseudo-embeddings for demo.
    
    Production uses: Vertex AI text-embedding-004
    See: backend/app/services/rag.py
    """
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    embedding = []
    for i in range(0, min(len(text_hash), dim * 2), 2):
        byte_val = int(text_hash[i:i+2], 16)
        embedding.append((byte_val - 128) / 128)
    while len(embedding) < dim:
        embedding.append(0.0)
    return embedding[:dim]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embeddings."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

# Runbook knowledge base with embeddings
RUNBOOK_KB = [
    {
        "incident_type": "brute_force",
        "description": "Multiple failed authentication attempts indicating credential attack",
        "steps": [
            RunbookStep(step="Block source IP at firewall", why="Stop ongoing attack", risk="low"),
            RunbookStep(step="Force password reset for affected accounts", why="Invalidate compromised credentials", risk="medium"),
            RunbookStep(step="Enable account lockout policy", why="Prevent future brute force", risk="low"),
            RunbookStep(step="Review authentication logs", why="Identify lateral movement", risk="low"),
        ]
    },
    {
        "incident_type": "data_exfiltration",
        "description": "Large data transfer to external destination indicating data theft",
        "steps": [
            RunbookStep(step="Isolate affected endpoint", why="Stop data loss", risk="medium"),
            RunbookStep(step="Capture memory dump", why="Forensic evidence", risk="low"),
            RunbookStep(step="Identify exfiltrated data scope", why="Compliance reporting", risk="low"),
            RunbookStep(step="Revoke compromised credentials", why="Prevent re-entry", risk="medium"),
        ]
    },
    {
        "incident_type": "malware",
        "description": "Suspicious file execution and process activity indicating malware infection",
        "steps": [
            RunbookStep(step="Quarantine infected system", why="Prevent spread", risk="medium"),
            RunbookStep(step="Run full EDR scan", why="Identify malware variant", risk="low"),
            RunbookStep(step="Check persistence mechanisms", why="Ensure complete removal", risk="low"),
            RunbookStep(step="Scan adjacent systems", why="Detect lateral movement", risk="low"),
        ]
    }
]

# Pre-compute embeddings for knowledge base
for kb in RUNBOOK_KB:
    kb["embedding"] = generate_embedding(kb["description"])

def retrieve_similar_runbooks(query: str, k: int = 2) -> list[dict]:
    """
    Semantic search for similar runbooks using embeddings.
    
    Production uses: pgvector with <=> operator
    SQL: SELECT * FROM runbooks ORDER BY embedding <=> $1 LIMIT $2
    """
    query_emb = generate_embedding(query)
    scored = [(kb, cosine_similarity(query_emb, kb["embedding"])) for kb in RUNBOOK_KB]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [kb for kb, score in scored[:k]]

def generate_runbook(features: dict, label: str) -> RunbookResponse:
    """Generate runbook using RAG retrieval + context."""
    # Build query from features
    query = f"Security incident severity {label}. "
    if features.get("failed_logins_last_hour", 0) > 10:
        query += "Multiple failed login attempts. "
    if features.get("data_exfil_bytes", 0) > 0:
        query += "Large data transfer detected. "
    if features.get("suspicious_file_activity"):
        query += "Suspicious file activity observed. "
    
    # RAG retrieval
    similar = retrieve_similar_runbooks(query, k=1)
    
    if similar:
        return RunbookResponse(runbook=similar[0]["steps"], source="rag")
    
    # Fallback
    return RunbookResponse(
        runbook=[
            RunbookStep(step="Investigate incident", why="Gather context", risk="low"),
            RunbookStep(step="Contain affected systems", why="Limit impact", risk="medium"),
        ],
        source="stub"
    )

# Demo
runbook = generate_runbook(test_features, result.label)

# Render as styled HTML
risk_colors = {"low": "#28a745", "medium": "#ffc107", "high": "#dc3545"}

steps_html = ""
for i, step in enumerate(runbook.runbook, 1):
    steps_html += f"""
    <div style="display: flex; align-items: flex-start; margin-bottom: 15px; padding: 15px;
                background: white; border-radius: 8px; border-left: 4px solid {risk_colors[step.risk]};">
        <span style="background: #6c757d; color: white; width: 30px; height: 30px; border-radius: 50%;
                     display: flex; align-items: center; justify-content: center; margin-right: 15px;
                     flex-shrink: 0; font-weight: bold;">{i}</span>
        <div style="flex: 1;">
            <div style="font-weight: 600; color: #333; margin-bottom: 5px;">{step.step}</div>
            <div style="font-size: 13px; color: #666;">{step.why}</div>
        </div>
        <span style="background: {risk_colors[step.risk]}20; color: {risk_colors[step.risk]}; 
                     padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 500;">
            {step.risk.upper()}
        </span>
    </div>
    """

html = f"""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            max-width: 600px; padding: 20px; background: #f8f9fa; border-radius: 12px;">
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <span style="font-size: 32px; margin-right: 10px;">ğŸ“‹</span>
        <div>
            <h3 style="margin: 0; color: #333;">Generated Runbook</h3>
            <span style="font-size: 12px; color: #666;">Source: <code style="background: #e9ecef; 
                         padding: 2px 6px; border-radius: 4px;">{runbook.source}</code></span>
        </div>
    </div>
    {steps_html}
</div>
"""
display(HTML(html))


from pydantic import BaseModel, Field
from typing import Literal
from IPython.display import HTML, display
import re

# =============================================================================
# Policy Agent: Safety Validation with LLM Integration
# Ensures runbook commands are safe before human execution
# =============================================================================

class PolicyViolation(BaseModel):
    """A detected policy violation."""
    step: str
    reason: str
    severity: Literal["warning", "blocked"]

class PolicyResult(BaseModel):
    """Complete policy validation result."""
    approved: bool
    approved_steps: list[str]
    violations: list[PolicyViolation]
    requires_human_approval: list[str]

# Dangerous command patterns (regex for flexibility)
FORBIDDEN_PATTERNS = [
    (r"rm\s+-rf\s+/(?!\w)", "Destructive system-wide deletion"),
    (r"format\s+[a-z]:", "Disk format command"),
    (r"DROP\s+DATABASE", "Database deletion"),
    (r"shutdown\s+-[hrf]", "System shutdown command"),
    (r"mkfs\.", "Filesystem format command"),
    (r"dd\s+if=.*of=/dev/", "Direct disk write"),
]

# Commands requiring human approval
APPROVAL_KEYWORDS = ["delete", "terminate", "revoke", "disable", "block", "isolate"]

def validate_runbook(steps: list[RunbookStep]) -> PolicyResult:
    """
    Validate runbook steps against security policies.
    
    This implements the Policy Agent's safety validation logic.
    Production includes LLM-based semantic analysis.
    """
    violations = []
    approved_steps = []
    requires_approval = []
    
    for step_obj in steps:
        step = step_obj.step
        step_lower = step.lower()
        blocked = False
        
        # Check forbidden patterns
        for pattern, reason in FORBIDDEN_PATTERNS:
            if re.search(pattern, step, re.IGNORECASE):
                violations.append(PolicyViolation(
                    step=step,
                    reason=reason,
                    severity="blocked"
                ))
                blocked = True
                break
        
        if not blocked:
            # Check for approval-required commands
            needs_approval = any(kw in step_lower for kw in APPROVAL_KEYWORDS)
            if needs_approval:
                requires_approval.append(step)
            approved_steps.append(step)
    
    return PolicyResult(
        approved=len(violations) == 0,
        approved_steps=approved_steps,
        violations=violations,
        requires_human_approval=requires_approval
    )

# Demo: Validate our generated runbook
validation = validate_runbook(runbook.runbook)

# Render as styled HTML
html = f"""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            max-width: 600px; padding: 20px; background: #f8f9fa; border-radius: 12px;">
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <span style="font-size: 32px; margin-right: 10px;">ğŸ”’</span>
        <div>
            <h3 style="margin: 0; color: #333;">Policy Validation</h3>
            <span style="background: {'#28a745' if validation.approved else '#dc3545'}; color: white;
                         padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 500;">
                {'âœ“ APPROVED' if validation.approved else 'âœ— BLOCKED'}
            </span>
        </div>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px;">
        <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 28px; font-weight: bold; color: #28a745;">{len(validation.approved_steps)}</div>
            <div style="font-size: 12px; color: #666;">Approved Steps</div>
        </div>
        <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 28px; font-weight: bold; color: #ffc107;">{len(validation.requires_human_approval)}</div>
            <div style="font-size: 12px; color: #666;">Needs Approval</div>
        </div>
        <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 28px; font-weight: bold; color: #dc3545;">{len(validation.violations)}</div>
            <div style="font-size: 12px; color: #666;">Violations</div>
        </div>
    </div>
    
    {''.join(f'<div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 15px; margin-bottom: 10px; border-radius: 4px;"><strong>âš ï¸� Requires Approval:</strong> {step}</div>' for step in validation.requires_human_approval) if validation.requires_human_approval else ''}
    
    {''.join(f'<div style="background: #f8d7da; border-left: 4px solid #dc3545; padding: 10px 15px; margin-bottom: 10px; border-radius: 4px;"><strong>ğŸš« Blocked:</strong> {v.step}<br><small>{v.reason}</small></div>' for v in validation.violations) if validation.violations else ''}
</div>
"""
display(HTML(html))


import requests
import json
from IPython.display import HTML, display

# Cloud Run Backend URL
BASE_URL = "https://incident-triage-agent-226861216522.us-central1.run.app"

def render_json_card(title: str, emoji: str, data: dict, status_code: int = 200) -> str:
    """Render API response as a styled HTML card."""
    status_color = "#28a745" if status_code == 200 else "#dc3545"
    json_str = json.dumps(data, indent=2)
    
    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 700px; border: 1px solid #dee2e6; border-radius: 12px; overflow: hidden; margin: 10px 0;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px 20px;
                    display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; color: white;">
                <span style="font-size: 24px; margin-right: 10px;">{emoji}</span>
                <span style="font-weight: 600;">{title}</span>
            </div>
            <span style="background: {status_color}; color: white; padding: 4px 12px; border-radius: 20px;
                         font-size: 12px; font-weight: 500;">
                {status_code}
            </span>
        </div>
        <pre style="margin: 0; padding: 20px; background: #1e1e1e; color: #d4d4d4; overflow-x: auto;
                    font-family: 'Fira Code', 'Consolas', monospace; font-size: 13px; line-height: 1.5;">
{json_str}</pre>
    </div>
    """

print("âœ… API helpers loaded. Base URL:", BASE_URL)


# Check service health
try:
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    data = response.json()
    
    # Create status dashboard
    status_icons = {"true": "âœ…", "false": "â�Œ", True: "âœ…", False: "â�Œ"}
    
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 600px; padding: 20px; background: #f8f9fa; border-radius: 12px;">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <span style="font-size: 32px; margin-right: 10px;">ğŸ’š</span>
            <div>
                <h3 style="margin: 0; color: #333;">Service Health Check</h3>
                <code style="font-size: 11px; color: #666;">{BASE_URL}</code>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: white; padding: 20px; border-radius: 8px; text-align: center;
                        border: 2px solid {'#28a745' if data.get('status') == 'healthy' else '#dc3545'};">
                <div style="font-size: 36px;">{'ğŸ’š' if data.get('status') == 'healthy' else 'ğŸ’”'}</div>
                <div style="font-weight: 600; color: #333; margin-top: 5px;">API Status</div>
                <div style="font-size: 12px; color: #666;">{data.get('status', 'unknown').upper()}</div>
            </div>
            <div style="background: white; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 36px;">{status_icons.get(data.get('services').get('database'), 'â�“')}</div>
                <div style="font-weight: 600; color: #333; margin-top: 5px;">Database</div>
                <div style="font-size: 12px; color: #666;">PostgreSQL + pgvector</div>
            </div>
            <div style="background: white; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 36px;">{status_icons.get(data.get('services').get('redis'), 'â�“')}</div>
                <div style="font-weight: 600; color: #333; margin-top: 5px;">Cache</div>
                <div style="font-size: 12px; color: #666;">Upstash Redis</div>
            </div>
            <div style="background: white; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 36px;">{status_icons.get(data.get('services').get('llm'), 'â�“')}</div>
                <div style="font-weight: 600; color: #333; margin-top: 5px;">LLM</div>
                <div style="font-size: 12px; color: #666;">Gemini 3 Pro</div>
            </div>
        </div>
        
        <div style="margin-top: 15px; padding: 10px; background: #e9ecef; border-radius: 8px;
                    font-size: 12px; color: #666; text-align: center;">
            Version: <code>{data.get('version', 'unknown')}</code> â€¢ 
            Timestamp: <code>{data.get('timestamp', 'N/A')}</code>
        </div>
    </div>
    """
    display(HTML(html))
    
except Exception as e:
    print(f"â�Œ Health check failed: {e}")


# Call triage endpoint with HTML visualization
triage_payload = {
    "features": {
        "failed_logins_last_hour": 50,
        "process_spawn_count": 40,
        "suspicious_file_activity": True,
        "rare_outgoing_connection": True
    }
}

try:
    response = requests.post(
        f"{BASE_URL}/triage",
        json=triage_payload,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    data = response.json()
    
    # Determine colors based on severity
    severity = data.get("label", "MEDIUM")
    severity_colors = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#fd7e14", "CRITICAL": "#dc3545"}
    color = severity_colors.get(severity, "#6c757d")
    
    contribs = data.get("contribs", [])
    contribs_html = "".join(
        f'<div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;"><span style="color: #333;">{feat}</span><span style="color: {color}; font-weight: 600;">+{pts}</span></div>'
        for feat, pts in contribs
    )
    
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 600px; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); padding: 25px; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 14px; opacity: 0.9;">INCIDENT TRIAGE</div>
                    <div style="font-size: 32px; font-weight: bold; margin-top: 5px;">{severity}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 48px; font-weight: bold;">{data.get('score', 0)}</div>
                    <div style="font-size: 12px; opacity: 0.9;">Risk Score</div>
                </div>
            </div>
        </div>
        
        <div style="background: white; padding: 20px;">
            <h4 style="margin: 0 0 15px 0; color: #333; display: flex; align-items: center;">
                <span style="margin-right: 8px;">ğŸ“Š</span> Contributing Factors
            </h4>
            {contribs_html if contribs_html else '<div style="color: #666;">No factors detected</div>'}
        </div>
        
        <div style="background: #f8f9fa; padding: 15px 20px; font-size: 12px; color: #666;">
            <strong>Endpoint:</strong> POST /triage â€¢ 
            <strong>Response Time:</strong> {response.elapsed.total_seconds()*1000:.0f}ms
        </div>
    </div>
    """
    display(HTML(html))
    
except Exception as e:
    print(f"â�Œ Triage call failed: {e}")


# Call full flow endpoint with comprehensive visualization
# NOTE: This endpoint makes multiple LLM calls and typically takes 30-40 seconds

print("â�³ Calling full flow API (this takes 30-40 seconds due to LLM processing)...")

flow_payload = {
    "incident": {
        "incident_id": "INC-DEMO-001",
        "features": {
            "failed_logins_last_hour": 50,
            "process_spawn_count": 40,
            "suspicious_file_activity": True,
            "rare_outgoing_connection": True,
            "source_ip": "192.168.1.100",
            "affected_user": "admin"
        }
    }
}

try:
    import time
    start_time = time.time()
    
    response = requests.post(
        f"{BASE_URL}/flow/full",
        json=flow_payload,
        headers={"Content-Type": "application/json"},
        timeout=120  # 2 minute timeout for LLM processing
    )
    
    elapsed = time.time() - start_time
    print(f"âœ… Response received in {elapsed:.1f}s")
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract sections
        triage = data.get("triage", {})
        explanation = data.get("explanation", {})
        runbook_data = data.get("runbook", {})
        timeline = data.get("timeline", [])
        
        severity = triage.get("label", "MEDIUM")
        severity_colors = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#fd7e14", "CRITICAL": "#dc3545"}
        color = severity_colors.get(severity, "#6c757d")
        
        # Build runbook steps HTML
        steps = runbook_data.get("runbook", []) or runbook_data.get("steps", [])
        risk_colors = {"low": "#28a745", "medium": "#ffc107", "high": "#dc3545"}
        steps_html = ""
        for i, step in enumerate(steps[:5], 1):
            if isinstance(step, dict):
                step_text = step.get("step", str(step))
                risk = step.get("risk", "low")
            else:
                step_text = str(step)
                risk = "low"
            steps_html += f"""
            <div style="display: flex; align-items: center; padding: 10px; margin: 5px 0; 
                        background: white; border-radius: 6px; border-left: 3px solid {risk_colors.get(risk, '#6c757d')};">
                <span style="background: #667eea; color: white; width: 24px; height: 24px; 
                             border-radius: 50%; display: flex; align-items: center; justify-content: center;
                             font-size: 12px; margin-right: 12px; flex-shrink: 0;">{i}</span>
                <span style="font-size: 13px; color: #333;">{step_text[:80]}{'...' if len(step_text) > 80 else ''}</span>
            </div>
            """
        
        # Build timeline HTML
        timeline_html = ""
        for event in timeline[:6]:
            actor = event.get("actor", "unknown")
            event_type = event.get("type", "event")
            timeline_html += f"""
            <div style="display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid #eee;">
                <span style="width: 100px; font-size: 11px; color: #666;">{actor}</span>
                <span style="background: #e9ecef; padding: 2px 8px; border-radius: 10px; font-size: 11px;">{event_type}</span>
            </div>
            """
        
        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                    max-width: 900px;">
            
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; 
                        border-radius: 12px 12px 0 0; color: white;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 12px; opacity: 0.9;">FULL INCIDENT FLOW</div>
                        <div style="font-size: 24px; font-weight: bold; margin-top: 5px;">
                            {flow_payload['incident']['incident_id']}
                        </div>
                    </div>
                    <div style="background: {color}; padding: 8px 20px; border-radius: 20px; font-weight: bold;">
                        {severity}
                    </div>
                </div>
            </div>
            
            <!-- Content Grid -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0; background: #f8f9fa; 
                        border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 12px 12px;">
                
                <!-- Left Column -->
                <div style="padding: 20px; border-right: 1px solid #dee2e6;">
                    <!-- Explanation -->
                    <div style="margin-bottom: 20px;">
                        <h4 style="margin: 0 0 10px 0; color: #333; display: flex; align-items: center;">
                            <span style="margin-right: 8px;">ğŸ’¡</span> Analysis
                        </h4>
                        <div style="background: white; padding: 15px; border-radius: 8px; font-size: 13px; 
                                    color: #555; line-height: 1.6;">
                            {explanation.get('explanation', 'Generating explanation...')[:300]}{'...' if len(explanation.get('explanation', '')) > 300 else ''}
                        </div>
                    </div>
                    
                    <!-- Timeline -->
                    <div>
                        <h4 style="margin: 0 0 10px 0; color: #333; display: flex; align-items: center;">
                            <span style="margin-right: 8px;">ğŸ“œ</span> Timeline ({len(timeline)} events)
                        </h4>
                        <div style="background: white; padding: 15px; border-radius: 8px;">
                            {timeline_html if timeline_html else '<div style="color: #666; font-size: 13px;">No timeline events</div>'}
                        </div>
                    </div>
                </div>
                
                <!-- Right Column - Runbook -->
                <div style="padding: 20px;">
                    <h4 style="margin: 0 0 10px 0; color: #333; display: flex; align-items: center;">
                        <span style="margin-right: 8px;">ğŸ“‹</span> Runbook Steps
                        <span style="margin-left: auto; font-size: 11px; color: #666; font-weight: normal;">
                            Source: {runbook_data.get('source', 'llm')}
                        </span>
                    </h4>
                    <div style="background: #f0f0f0; padding: 10px; border-radius: 8px;">
                        {steps_html if steps_html else '<div style="color: #666; padding: 20px; text-align: center;">No runbook generated</div>'}
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <div style="background: #e9ecef; padding: 12px 20px; font-size: 11px; color: #666; 
                        border-radius: 0 0 12px 12px; margin-top: -1px; display: flex; justify-content: space-between;">
                <span><strong>Endpoint:</strong> POST /flow/full</span>
                <span><strong>Response Time:</strong> {elapsed:.2f}s</span>
                <span><strong>Trace ID:</strong> {data.get('trace_id', 'N/A')[:12]}...</span>
            </div>
        </div>
        """
        display(HTML(html))
    else:
        print(f"â�Œ Error {response.status_code}: {response.text}")
        
except requests.exceptions.Timeout:
    print(f"â�Œ Request timed out after 120 seconds. The API is still processing - try again in a moment.")
except Exception as e:
    print(f"â�Œ Full flow call failed: {e}")

