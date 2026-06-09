# =========================
# AI LogOps Prototype - Full runnable code for Kaggle notebook
# Paste into one cell and run
# =========================

# 1) Optional Kaggle secrets setup (safe if not present)
try:
    from kaggle_secrets import UserSecretsClient
    import os
    try:
        GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        print("✅ Kaggle secret 'GOOGLE_API_KEY' loaded into environment.")
    except Exception:
        print("ℹ️ 'GOOGLE_API_KEY' not found in Kaggle secrets (this demo does not require it).")
except Exception:
    # Not running on Kaggle or kaggle_secrets unavailable — continue safely
    print("ℹ️ kaggle_secrets not available; skipping secrets setup (OK for prototype).")

# 2) Standard library imports
import time
import uuid
import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict

# 3) Prototype shims (FunctionTool, LlmAgentProto, GeminiModel)
class FunctionTool:
    """Wrap a Python function to mimic an ADK FunctionTool (callable)."""
    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

@dataclass
class GeminiModel:
    """Prototype descriptor for the model (no real calls performed)."""
    model: str
    retry_options: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LlmAgentProto:
    """
    Minimal, deterministic 'agent' object for prototype demo.
    - name: agent name
    - model: GeminiModel or descriptor
    - instruction: human-readable instruction string
    - tools: list of FunctionTool objects
    """
    name: str
    model: Any
    instruction: str
    tools: list = field(default_factory=list)

    def handle(self, payload: Dict[str, Any]):
        """
        Prototype deterministic handler:
          - If the agent has exactly one tool, call it with appropriate args from payload.
          - Otherwise, produce a mocked structured output guided by instruction.
        """
        # If single tool, attempt to call it using common arg conventions
        if self.tools and len(self.tools) == 1:
            tool = self.tools[0]
            try:
                # Common cases for our agents:
                if 'raw_logs' in payload:
                    return tool.func(payload['raw_logs'])
                if 'exception_count' in payload:
                    return tool.func(payload['exception_count'])
                if 'triage_result' in payload:
                    return tool.func(payload['triage_result'])
                # fallback: pass whole payload
                return tool.func(payload)
            except Exception as e:
                return {"error": f"tool_error: {str(e)}"}
        # No tools or multiple tools: return an info dict
        return {"notice": f"Agent {self.name} would use model {getattr(self.model,'model',self.model)}. Payload keys: {list(payload.keys())}"}

# 4) Helper: deterministic fingerprint
def make_fingerprint(*parts):
    """Deterministic fingerprint from input parts."""
    s = "||".join([str(p) for p in parts if p is not None])
    return "fp-" + hashlib.sha1(s.encode()).hexdigest()[:10]

# 5) Tools (FunctionTool-wrapped) used by agents
def parse_logs_tool(raw_logs: str):
    """
    Simple parser for demo:
      - counts 'Exception' or 'ERROR' occurrences
      - returns top_lines (first 20) and a short raw_excerpt
    """
    lines = raw_logs.splitlines()
    exception_count = sum(1 for l in lines if ("Exception" in l) or ("ERROR" in l) or ("Error" in l))
    top_lines = lines[:20]
    raw_excerpt = raw_logs[:400]
    return {"exception_count": exception_count, "top_lines": top_lines, "raw_excerpt": raw_excerpt}

def detect_anomaly_tool(exception_count: int):
    """
    Simple heuristic:
      0 -> low (not anomalous), 1-2 -> medium, >=3 -> high
    """
    if exception_count >= 3:
        return {"anomalous": True, "severity": "high"}
    elif exception_count >= 1:
        return {"anomalous": True, "severity": "medium"}
    else:
        return {"anomalous": False, "severity": "low"}

def store_fingerprint_tool(payload: dict):
    """
    Prototype memory store: simply returns a stored confirmation.
    Expect payload = {'fingerprint':..., 'severity':...}
    """
    return {"status": "stored", "fingerprint": payload.get("fingerprint"), "severity": payload.get("severity")}

def generate_auto_fix_tool(triage_result: dict):
    """
    Create a small remediation plan based on severity.
    Returns a plan dictionary.
    """
    severity = triage_result.get("severity", "low")
    if severity == "high":
        return {"action": "restart_service", "script": "kubectl rollout restart deployment/my-service", "notes": "Investigate NPE stacktrace in module X"}
    elif severity == "medium":
        return {"action": "scale_up", "script": "kubectl scale deployment/my-service --replicas=3", "notes": "Add validation/retry"}
    else:
        return {"action": "monitor", "script": None, "notes": "No immediate action"}

# Wrap tools into FunctionTool objects
parse_logs_fn = FunctionTool(parse_logs_tool, name="parse_logs")
detect_anomaly_fn = FunctionTool(detect_anomaly_tool, name="detect_anomaly")
store_fingerprint_fn = FunctionTool(store_fingerprint_tool, name="store_fingerprint")
generate_auto_fix_fn = FunctionTool(generate_auto_fix_tool, name="generate_auto_fix")

# 6) Agents in LlmAgent(...) style
retry_config = {"max_retries": 2}

log_parser_agent = LlmAgentProto(
    name="log_parser_agent",
    model=GeminiModel(model="gemini-2.5-flash-lite"),
    instruction="""
You are a LogOps parsing assistant. Input: raw_logs (string).
Use parse_logs to extract exception_count, top_lines, raw_excerpt.
Return the structured dict.
""",
    tools=[parse_logs_fn],
)

anomaly_detector_agent = LlmAgentProto(
    name="anomaly_detector_agent",
    model=GeminiModel(model="gemini-2.5-flash-lite"),
    instruction="""
You are an anomaly detector. Input: exception_count and top_lines.
Use detect_anomaly to label anomalous (bool) and severity (low/medium/high).
Return the structured dict.
""",
    tools=[detect_anomaly_fn],
)

llm_triage_agent = LlmAgentProto(
    name="llm_triage_agent",
    model=GeminiModel(model="gemini-2.5-flash"),
    instruction="""
You are a triage expert. Input: detection (anomalous + severity + features.top_lines).
If anomalous is False: return severity=low and null root/suggested_fix.
If anomalous is True: pick a root_cause line, propose a concise suggested_fix, create a fingerprint,
and call store_fingerprint with {'fingerprint','severity'}.
""",
    tools=[store_fingerprint_fn],
)

auto_fix_agent = LlmAgentProto(
    name="auto_fix_agent",
    model=GeminiModel(model="gemini-2.5-flash-lite"),
    instruction="""
You are an AutoFix assistant. Input: triage_result.
Use generate_auto_fix to produce an actionable plan (script + notes).
Return the plan as JSON (applied:false in prototype).
""",
    tools=[generate_auto_fix_fn],
)

print("✅ Prototype agents defined: log_parser_agent, anomaly_detector_agent, llm_triage_agent, auto_fix_agent")

# 7) Small in-memory memory adapter (used by orchestrator)
class InMemoryAdapter:
    def __init__(self):
        self.store = []
    def put(self, item):
        self.store.append(item)
    def query_recent(self, limit=10):
        return list(self.store[-limit:])
    def __repr__(self):
        return f"InMemoryAdapter(store_size={len(self.store)})"

memory_store = InMemoryAdapter()

# 8) Sequential Orchestrator (runs the 4-step pipeline)
class SequentialOrchestrator:
    def __init__(self, parser_agent, detector_agent, triage_agent, autofix_agent, memory_store=None, verbose=True):
        self.parser = parser_agent
        self.detector = detector_agent
        self.triage = triage_agent
        self.autofix = autofix_agent
        self.memory = memory_store
        self.verbose = verbose

    def _now_ms(self):
        return int(time.time() * 1000)

    def run(self, raw_logs: str, trace_id: str = None):
        trace_id = trace_id or f"trace-{uuid.uuid4().hex[:8]}"
        metrics = {"trace_id": trace_id, "start_ms": self._now_ms(), "steps": []}

        def record_step(name, start_ms, end_ms, extra=None):
            metrics["steps"].append({"name": name, "start_ms": start_ms, "end_ms": end_ms, "duration_ms": end_ms - start_ms, "extra": extra or {}})

        # Step 1: parse
        s0 = self._now_ms()
        features = self.parser.handle({"raw_logs": raw_logs})
        e0 = self._now_ms()
        record_step("log_parser", s0, e0, {"exception_count": features.get("exception_count")})
        if self.verbose:
            print(f"[{trace_id}] log_parser -> exception_count={features.get('exception_count')}")

        # Step 2: detect
        s1 = self._now_ms()
        detection = self.detector.handle({"exception_count": features.get("exception_count"), "top_lines": features.get("top_lines")})
        # ensure features echoed
        if isinstance(detection, dict) and "features" not in detection:
            detection["features"] = features
        e1 = self._now_ms()
        record_step("anomaly_detector", s1, e1, {"anomalous": detection.get("anomalous"), "severity": detection.get("severity")})
        if self.verbose:
            print(f"[{trace_id}] anomaly_detector -> anomalous={detection.get('anomalous')}, severity={detection.get('severity')}")

        # Step 3: triage
        s2 = self._now_ms()
        if detection.get("anomalous"):
            # pick first error/exception line if present
            rc_line = None
            for ln in features.get("top_lines", []):
                if ("Exception" in ln) or ("ERROR" in ln) or ("Error" in ln):
                    rc_line = ln
                    break
            fingerprint = make_fingerprint(rc_line or features.get("raw_excerpt", ""))
            sev = detection.get("severity", "low")
            if sev == "high":
                suggested_fix = "Restart service; inspect NPE stack; add null checks."
            elif sev == "medium":
                suggested_fix = "Add validation and retry logic."
            else:
                suggested_fix = None

            triage_result = {
                "severity": sev,
                "root_cause": rc_line,
                "suggested_fix": suggested_fix,
                "fingerprint": fingerprint,
            }

            # call triage agent's store tool (prototype) via its tools list
            try:
                if self.triage.tools and len(self.triage.tools) >= 1:
                    store_resp = self.triage.tools[0].func({"fingerprint": fingerprint, "severity": sev})
                    triage_result["store_status"] = store_resp
            except Exception as e:
                triage_result["store_status"] = {"status": "error", "error": str(e)}

            # persist into orchestrator memory
            if self.memory:
                try:
                    self.memory.put({"trace_id": trace_id, "fingerprint": fingerprint, "triage": triage_result, "ts": int(time.time())})
                except Exception:
                    pass
        else:
            triage_result = {"severity": "low", "root_cause": None, "suggested_fix": None, "fingerprint": None, "store_status": None}
        e2 = self._now_ms()
        record_step("llm_triage", s2, e2, {"fingerprint": triage_result.get("fingerprint")})
        if self.verbose:
            print(f"[{trace_id}] llm_triage -> severity={triage_result.get('severity')}, fingerprint={triage_result.get('fingerprint')}")

        # Step 4: autofix
        s3 = self._now_ms()
        try:
            plan = self.autofix.tools[0].func(triage_result) if (self.autofix.tools and len(self.autofix.tools) >= 1) else {}
            autofix_resp = {"applied": False, "plan": plan, "notes": plan.get("notes")}
        except Exception as e:
            autofix_resp = {"error": str(e)}
        e3 = self._now_ms()
        record_step("autofix", s3, e3, {"plan_present": isinstance(autofix_resp, dict)})
        if self.verbose:
            print(f"[{trace_id}] autofix -> plan_present={isinstance(autofix_resp, dict)}")

        final = {
            "trace_id": trace_id,
            "metrics": metrics,
            "features": features,
            "detection": detection,
            "triage": triage_result,
            "autofix": autofix_resp,
            "end_ms": self._now_ms()
        }
        final["metrics"]["duration_ms"] = final["end_ms"] - final["metrics"]["start_ms"]
        if self.verbose:
            print(f"[{trace_id}] pipeline complete duration_ms={final['metrics']['duration_ms']}")
        return final

# 9) Instantiate orchestrator and run sample logs
orch = SequentialOrchestrator(
    parser_agent=log_parser_agent,
    detector_agent=anomaly_detector_agent,
    triage_agent=llm_triage_agent,
    autofix_agent=auto_fix_agent,
    memory_store=memory_store,
    verbose=True
)

# Sample logs (you can replace with your own uploaded log text)
sample_raw_logs = (
    "2025-11-30 12:00:01 INFO Service: Starting up\n"
    "2025-11-30 12:00:05 ERROR ComponentA: java.lang.NullPointerException: value was null\n"
    "2025-11-30 12:00:06 INFO ComponentB: doing work\n"
    "2025-11-30 12:00:07 ERROR ComponentA: java.lang.ArrayIndexOutOfBoundsException: Index 5 out of bounds\n"
    "2025-11-30 12:00:08 INFO Health: OK\n"
)

# Run pipeline
result = orch.run(sample_raw_logs)

# Print result (pretty)
print("\n=== FINAL CONSOLIDATED RESULT ===")
print(json.dumps(result, indent=2))

# Optional: inspect memory store contents
print("\n=== MEMORY STORE CONTENTS ===")
print(memory_store.query_recent(limit=20))

# =========================
# End of prototype cell
# =========================


