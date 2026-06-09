!pip install -q --upgrade google-genai


import os
import json
import uuid
import time
import math
import logging
import functools
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Callable, Optional, Tuple

from google import genai
from google.genai import types

# ========= Logging / Observability Setup ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("devils-advocate-system")

# ========= Gemini Client Setup (Kaggle Secrets Only) ==========

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

try:
    API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
    logger.info("GEMINI_API_KEY loaded successfully from Kaggle Secrets.")
except Exception:
    API_KEY = None
    logger.error("GEMINI_API_KEY not found in Kaggle Secrets. Please add it.")

client = genai.Client(api_key=API_KEY)

DEFAULT_MODEL = "gemini-2.5-flash"   # fast + cheap, good for agents
EMBED_MODEL   = "text-embedding-004" # for memory bank

# ========= Simple Metrics (observability) ==========
METRICS = {
    "llm_calls": 0,
    "llm_input_tokens": 0,
    "llm_output_tokens": 0,
    "tool_calls": 0,
    "sessions_created": 0
}

def record_llm_usage(response: types.GenerateContentResponse):
    """Update metrics from a Gemini response if usage_metadata is available."""
    usage = getattr(response, "usage_metadata", None)
    if usage:
        METRICS["llm_input_tokens"] += getattr(usage, "prompt_token_count", 0) or 0
        METRICS["llm_output_tokens"] += getattr(usage, "candidates_token_count", 0) or 0

# ========= Simple Tracing Decorator ==========
def traced(name: str):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"[TRACE] {name} START")
            result = fn(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"[TRACE] {name} END in {duration:.2f}s")
            return result
        return wrapper
    return decorator

def get_text_from_response(response: types.GenerateContentResponse) -> str:
    """
    Safely extract text from a Gemini response.
    Works even if response.text is None by reading candidates/parts.
    """
    text = getattr(response, "text", None)
    if text:
        return text

    parts = []
    try:
        for cand in getattr(response, "candidates", []):
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []):
                t = getattr(part, "text", None)
                if t:
                    parts.append(t)
    except Exception as e:
        logger.warning(f"Failed to extract text from response: {e}")

    return "\n".join(parts).strip()


logger.info("Config & Gemini client initialized using Kaggle Secrets.")



# ========= A2A Message Protocol ==========
@dataclass
class AgentMessage:
    sender: str
    receiver: str
    role: str     # "user" / "agent" / "system"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_line(self) -> str:
        tag = self.metadata.get("tag", "")
        tag_str = f" [{tag}]" if tag else ""
        return f"{self.sender}->{self.receiver}{tag_str} ({self.role}): {self.content}"


# ========= Session Service ==========
class InMemorySessionService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "messages": [],
            "state": {},
            "created_at": time.time()
        }
        METRICS["sessions_created"] += 1
        logger.info(f"New session created: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return self.sessions[session_id]

    def add_message(self, session_id: str, msg: AgentMessage):
        self.sessions[session_id]["messages"].append(msg)

    def update_state(self, session_id: str, key: str, value: Any):
        self.sessions[session_id]["state"][key] = value

    def get_messages(self, session_id: str) -> List[AgentMessage]:
        return self.sessions[session_id]["messages"]

    def get_state(self, session_id: str) -> Dict[str, Any]:
        return self.sessions[session_id]["state"]

SESSION_SERVICE = InMemorySessionService()


# ========= Long-term Memory Bank ==========
class MemoryBank:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    @traced("MemoryBank.add")
    def add(self, text: str, metadata: Dict[str, Any]):
        embedding_response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text
        )
        emb_vec = embedding_response.embeddings[0].values
        self.entries.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "embedding": emb_vec,
            "metadata": metadata
        })

    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @traced("MemoryBank.search")
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.entries:
            return []
        embedding_response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=query
        )
        qvec = embedding_response.embeddings[0].values
        scored = [
            (self._cosine(qvec, e["embedding"]), e)
            for e in self.entries
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for score, e in scored[:k]]

MEMORY_BANK = MemoryBank()


# ========= Context Compaction ==========
@traced("compact_context")
def compact_context(messages: List[AgentMessage], max_chars: int = 4000) -> str:
    joined = "\n".join(m.to_prompt_line() for m in messages)
    if len(joined) <= max_chars:
        return joined

    # Keep last 8, summarize the rest
    recent = messages[-8:]
    older = messages[:-8]
    older_text = "\n".join(m.to_prompt_line() for m in older)

    summary_prompt = (
        "You are summarizing a multi-agent debate.\n"
        "Summarize the following older messages in bullet points, focusing on key arguments and risks.\n\n"
        f"{older_text}\n\n"
        "Summary:"
    )
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=summary_prompt
    )
    record_llm_usage(response)
    METRICS["llm_calls"] += 1
    summary = get_text_from_response(response)

    compact = (
        "=== Compacted summary of earlier conversation ===\n"
        f"{summary}\n\n"
        "=== Recent messages ===\n"
        + "\n".join(m.to_prompt_line() for m in recent)
    )
    return compact

logger.info("Session service & MemoryBank initialized.")



# ========= Tool interfaces ==========
@dataclass
class ToolResult:
    tool_name: str
    input: Dict[str, Any]
    output: Any

class BaseTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {},
        }


# ========= Custom Tools ==========
class ProsConsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="pros_cons",
            description="Given an option and context, list pros and cons."
        )

    @traced("ProsConsTool.run")
    def run(self, option: str, context: str = "") -> ToolResult:
        prompt = f"Option: {option}\nContext: {context}\n\nList 5 pros and 5 cons."
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt
        )
        record_llm_usage(response)
        METRICS["llm_calls"] += 1
        text = get_text_from_response(response)
        return ToolResult(self.name, {"option": option, "context": context}, text)


class RiskScoreTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="risk_score",
            description="Return risk score 1-10 and short explanation for an option."
        )

    @traced("RiskScoreTool.run")
    def run(self, option: str, context: str = "") -> ToolResult:
        prompt = (
            "You are a risk analyst. Rate the risk of this decision from 1 (very low) "
            "to 10 (extreme). Provide JSON: {\"score\": int, \"reason\": string}.\n\n"
            f"Decision: {option}\nContext: {context}"
        )
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt
        )
        record_llm_usage(response)
        METRICS["llm_calls"] += 1
        raw = get_text_from_response(response)

        parsed = {"score": 5, "reason": raw}
        try:
            json_str = raw
            if "```" in raw:
                json_str = raw.split("```")[-1]
            parsed2 = json.loads(json_str)
            if isinstance(parsed2, dict) and "score" in parsed2:
                parsed = parsed2
        except Exception as e:
            logger.warning(f"RiskScoreTool JSON parse failed: {e}; raw={raw[:200]!r}")

        return ToolResult(self.name, {"option": option, "context": context}, parsed)


# ========= Code Execution Tool ==========
class CodeExecutionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="python_exec",
            description="Execute Python code and return resulting variables."
        )

    @traced("CodeExecutionTool.run")
    def run(self, code: str) -> ToolResult:
        local_env = {}
        try:
            exec(code, {}, local_env)
            output = {k: v for k, v in local_env.items()}
        except Exception as e:
            output = {"error": str(e)}
        METRICS["tool_calls"] += 1
        return ToolResult(self.name, {"code": code}, output)


# ========= MCP-style Tool ==========
class MCPTool(BaseTool):
    def __init__(self, name: str, description: str, fn: Callable[[Dict[str, Any]], Any], param_schema: Dict[str, Any]):
        super().__init__(name, description)
        self.fn = fn
        self.param_schema = param_schema

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "json_schema": self.param_schema,
        }

    @traced("MCPTool.run")
    def run(self, **kwargs) -> ToolResult:
        output = self.fn(kwargs)
        METRICS["tool_calls"] += 1
        return ToolResult(self.name, kwargs, output)


# ========= OpenAPI-style Tool (mock) ==========
class OpenAPITool(BaseTool):
    def __init__(self, name: str, description: str, openapi_spec: Dict[str, Any]):
        super().__init__(name, description)
        self.openapi_spec = openapi_spec

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "openapi": self.openapi_spec,
        }

    @traced("OpenAPITool.run")
    def run(self, **kwargs) -> ToolResult:
        simulated = {
            "message": "Simulated OpenAPI tool response.",
            "input": kwargs,
            "endpoint": self.openapi_spec.get("paths", {})
        }
        METRICS["tool_calls"] += 1
        return ToolResult(self.name, kwargs, simulated)


# MCP function example
def simple_math_mcp(params: Dict[str, Any]) -> Dict[str, Any]:
    a = float(params.get("a", 0))
    b = float(params.get("b", 0))
    return {"a": a, "b": b, "sum": a + b, "product": a * b}

MCP_MATH_TOOL = MCPTool(
    name="mcp_math",
    description="Add and multiply two numbers.",
    fn=simple_math_mcp,
    param_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    }
)

# Mock OpenAPI weather spec
OPENAPI_WEATHER_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Mock Weather API", "version": "1.0"},
    "paths": {
        "/weather": {
            "get": {
                "summary": "Get weather for a city",
                "parameters": [
                    {"name": "city", "in": "query", "schema": {"type": "string"}}
                ]
            }
        }
    }
}
OPENAPI_WEATHER_TOOL = OpenAPITool(
    name="weather_api",
    description="Mock OpenAPI Weather tool",
    openapi_spec=OPENAPI_WEATHER_SPEC
)

ALL_TOOLS: Dict[str, BaseTool] = {
    "pros_cons": ProsConsTool(),
    "risk_score": RiskScoreTool(),
    "python_exec": CodeExecutionTool(),
    "mcp_math": MCP_MATH_TOOL,
    "weather_api": OPENAPI_WEATHER_TOOL
}

logger.info(f"Tools registered: {list(ALL_TOOLS.keys())}")


class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        system_instruction: str,
        tools: Optional[List[str]] = None
    ):
        self.name = name
        self.role = role
        self.system_instruction = system_instruction
        self.tools = tools or []

    @traced("Agent.call_llm")
    def call_llm(self, prompt: str) -> str:
        try:
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    max_output_tokens=512,
                    temperature=0.6
                )
            )
        except Exception as e:
            logger.error(f"LLM call failed for agent {self.name}: {e}")
            return ""

        record_llm_usage(response)
        METRICS["llm_calls"] += 1
        return get_text_from_response(response)

    def available_tools_description(self) -> str:
        if not self.tools:
            return "No tools available."
        descs = []
        for t in self.tools:
            tool = ALL_TOOLS[t]
            descs.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descs)

    def maybe_use_tools(self, decision: str, context: str) -> List[ToolResult]:
        results = []
        for tname in self.tools:
            tool = ALL_TOOLS[tname]
            if isinstance(tool, ProsConsTool):
                results.append(tool.run(option=decision, context=context))
            elif isinstance(tool, RiskScoreTool):
                results.append(tool.run(option=decision, context=context))
            elif isinstance(tool, MCPTool):
                results.append(tool.run(a=3, b=5))
            elif isinstance(tool, OpenAPITool):
                results.append(tool.run(city="Mumbai"))
        return results

    def format_tool_results(self, tool_results: List[ToolResult]) -> str:
        if not tool_results:
            return "No tool results."
        parts = []
        for r in tool_results:
            parts.append(
                f"[Tool:{r.tool_name}] input={r.input}\noutput={r.output}"
            )
        return "\n\n".join(parts)

    def act(
        self,
        session_id: str,
        user_question: str,
        decision_options: List[str],
        context: str
    ) -> AgentMessage:
        messages = SESSION_SERVICE.get_messages(session_id)
        compacted = compact_context(messages)

        main_option = decision_options[0] if decision_options else user_question
        tool_results = self.maybe_use_tools(main_option, context)
        tool_text = self.format_tool_results(tool_results)

        prompt = (
            f"{compacted}\n\n"
            f"---\nYou are agent '{self.name}' with role: {self.role}.\n"
            f"Available tools:\n{self.available_tools_description()}\n\n"
            f"User decision problem: {user_question}\n"
            f"Decision options: {decision_options}\n"
            f"Context: {context}\n\n"
            f"Tool results (if any):\n{tool_text}\n\n"
            "Now give your analysis in this JSON format:\n"
            "{\n"
            '  \"stance\": \"supportive|critical|neutral\",\n'
            '  \"summary\": \"short summary\",\n'
            '  \"arguments\": [\"point1\", \"point2\", ...],\n'
            '  \"recommended_option\": \"one of the decision options\"\n'
            "}\n"
        )

        raw = self.call_llm(prompt)

        if not raw:
            logger.warning(f"Agent {self.name} got empty response from LLM.")
            parsed = {
                "stance": "neutral",
                "summary": "Model returned empty response.",
                "arguments": [],
                "recommended_option": main_option
            }
        else:
            try:
                json_str = raw
                if "```" in raw:
                    json_str = raw.split("```")[-1]
                parsed = json.loads(json_str)
            except Exception as e:
                logger.warning(f"Failed to parse JSON from agent {self.name}: {e}. Raw: {raw[:200]!r}")
                parsed = {
                    "stance": "neutral",
                    "summary": raw[:300],
                    "arguments": [raw[:300]],
                    "recommended_option": main_option
                }

        content = json.dumps(parsed, indent=2)
        msg = AgentMessage(
            sender=self.name,
            receiver="coordinator",
            role="agent",
            content=content,
            metadata={"stance": parsed.get("stance", "neutral")}
        )
        SESSION_SERVICE.add_message(session_id, msg)
        return msg


class SupportiveAgent(Agent):
    def __init__(self):
        super().__init__(
            name="SupportiveAnalyst",
            role="Find reasons to support the decision, highlight benefits",
            system_instruction=(
                "You are a helpful analyst who tries to find the BEST case for a decision, "
                "but you must still be logically sound."
            ),
            tools=["pros_cons"]
        )


class DevilAdvocateAgent(Agent):
    def __init__(self):
        super().__init__(
            name="DevilsAdvocate",
            role="Find risks, weaknesses, and hidden assumptions in the decision",
            system_instruction=(
                "You are a devil's advocate. Your job is to challenge the decision, "
                "find flaws, risks, and worst-case scenarios. Be rigorous but fair."
            ),
            tools=["pros_cons", "risk_score", "mcp_math", "weather_api"]
        )


class CoordinatorAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Coordinator",
            role="Merge opinions of other agents and produce a final recommendation.",
            system_instruction=(
                "You coordinate multiple experts. You read their JSON analyses and produce a "
                "balanced final recommendation with clear justification and risk level."
            ),
            tools=[]
        )

    def act(
        self,
        session_id: str,
        user_question: str,
        decision_options: List[str],
        context: str
    ) -> AgentMessage:
        messages = SESSION_SERVICE.get_messages(session_id)
        compacted = compact_context(messages)

        prompt = (
            f"{compacted}\n\n"
            "You are the coordinator. Previous messages include analyses from other agents "
            "(SupportiveAnalyst, DevilsAdvocate), in JSON format.\n\n"
            "Create a final JSON decision object with fields:\n"
            "{\n"
            '  \"final_recommendation\": \"string\",\n'
            '  \"chosen_option\": \"one of the decision options or a new suggestion\",\n'
            '  \"confidence\": 0-1,\n'
            '  \"summary\": \"short explanation\",\n'
            '  \"key_risks\": [\"risk1\", \"risk2\"],\n'
            '  \"mitigations\": [\"mitigation1\", \"mitigation2\"]\n'
            "}\n"
            f"User decision problem: {user_question}\n"
            f"Decision options: {decision_options}\n"
            f"Context: {context}\n"
        )

        raw = self.call_llm(prompt)

        if not raw:
            logger.warning("Coordinator got empty response from LLM.")
            parsed = {
                "final_recommendation": "Model returned empty response.",
                "chosen_option": decision_options[0] if decision_options else user_question,
                "confidence": 0.5,
                "summary": "Fallback decision due to empty model response.",
                "key_risks": [],
                "mitigations": []
            }
        else:
            try:
                json_str = raw
                if "```" in raw:
                    json_str = raw.split("```")[-1]
                parsed = json.loads(json_str)
            except Exception as e:
                logger.warning(f"Failed to parse JSON in Coordinator: {e}. Raw: {raw[:200]!r}")
                parsed = {
                    "final_recommendation": raw[:300],
                    "chosen_option": decision_options[0] if decision_options else user_question,
                    "confidence": 0.7,
                    "summary": raw[:300],
                    "key_risks": [],
                    "mitigations": []
                }

        content = json.dumps(parsed, indent=2)
        msg = AgentMessage(
            sender=self.name,
            receiver="user",
            role="agent",
            content=content,
            metadata={"type": "final_decision"}
        )
        SESSION_SERVICE.add_message(session_id, msg)
        return msg


SUPPORT_AGENT = SupportiveAgent()
DEVIL_AGENT = DevilAdvocateAgent()
COORDINATOR = CoordinatorAgent()

logger.info("Agents initialized.")


from concurrent.futures import ThreadPoolExecutor

EXECUTOR = ThreadPoolExecutor(max_workers=4)

@traced("run_parallel_agents")
def run_parallel_agents(
    session_id: str,
    agents: List[Agent],
    user_question: str,
    decision_options: List[str],
    context: str
) -> List[AgentMessage]:
    """
    Run agents in parallel threads (no asyncio -> safe in Kaggle).
    """
    futures = []
    for agent in agents:
        futures.append(
            EXECUTOR.submit(
                agent.act,
                session_id,
                user_question,
                decision_options,
                context
            )
        )
    results = [f.result() for f in futures]
    return results


@traced("run_sequential_with_loop")
def run_sequential_with_loop(
    session_id: str,
    user_question: str,
    decision_options: List[str],
    context: str,
    max_iterations: int = 2
) -> AgentMessage:
    """
    1. Run supportive + devil agents "in parallel".
    2. Coordinator summarizes.
    3. Iterate up to max_iterations (loop agent).
    """
    for i in range(max_iterations):
        logger.info(f"=== Iteration {i+1}/{max_iterations} ===")
        _ = run_parallel_agents(
            session_id=session_id,
            agents=[SUPPORT_AGENT, DEVIL_AGENT],
            user_question=user_question,
            decision_options=decision_options,
            context=context
        )
        final_msg = COORDINATOR.act(
            session_id=session_id,
            user_question=user_question,
            decision_options=decision_options,
            context=context
        )
        try:
            final_data = json.loads(final_msg.content)
            if final_data.get("confidence", 0.0) >= 0.8:
                logger.info("Confidence high enough, stopping loop.")
                break
        except Exception:
            pass
    return final_msg


# ========= Long-running operations: pause & resume ==========
STATE_FILE = "session_state.json"

@traced("save_session_state")
def save_session_state(session_id: str):
    data = SESSION_SERVICE.get_session(session_id)
    serializable = {
        "session_id": session_id,
        "user_id": data["user_id"],
        "messages": [asdict(m) for m in data["messages"]],
        "state": data["state"],
        "created_at": data["created_at"],
        "metrics": METRICS
    }
    with open(STATE_FILE, "w") as f:
        json.dump(serializable, f)
    logger.info(f"Session {session_id} saved to {STATE_FILE}.")

@traced("load_session_state")
def load_session_state() -> str:
    with open(STATE_FILE, "r") as f:
        data = json.load(f)
    session_id = data["session_id"]
    SESSION_SERVICE.sessions[session_id] = {
        "user_id": data["user_id"],
        "messages": [AgentMessage(**m) for m in data["messages"]],
        "state": data["state"],
        "created_at": data["created_at"]
    }
    for k, v in data.get("metrics", {}).items():
        METRICS[k] = v
    logger.info(f"Session {session_id} loaded from {STATE_FILE}.")
    return session_id

logger.info("Orchestration helpers ready.")



@traced("evaluate_decision")
def evaluate_decision(final_message: AgentMessage) -> Dict[str, Any]:
    prompt = (
        "You are an evaluator of decision quality.\n"
        "You will receive a JSON decision object from a multi-agent debate system.\n"
        "Score it on a 0-100 scale for: reasoning_quality, risk_coverage, clarity.\n"
        "Answer in JSON like:\n"
        "{\n"
        '  "reasoning_quality": int,\n'
        '  "risk_coverage": int,\n'
        '  "clarity": int,\n'
        '  "comments": "short comments"\n'
        "}\n\n"
        f"Decision object:\n{final_message.content}\n"
    )

    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="Be strict but fair in your evaluation."
            )
        )
    except Exception as e:
        logger.error(f"Evaluator LLM call failed: {e}")
        return {
            "reasoning_quality": 60,
            "risk_coverage": 60,
            "clarity": 60,
            "comments": f"Evaluation failed: {e}"
        }

    record_llm_usage(response)
    METRICS["llm_calls"] += 1
    raw = get_text_from_response(response)

    if not raw:
        return {
            "reasoning_quality": 60,
            "risk_coverage": 60,
            "clarity": 60,
            "comments": "Model returned empty response during evaluation."
        }

    try:
        json_str = raw
        if "```" in raw:
            json_str = raw.split("```")[-1]
        parsed = json.loads(json_str)
    except Exception as e:
        logger.warning(f"Failed to parse evaluation JSON: {e}. Raw: {raw[:200]!r}")
        parsed = {
            "reasoning_quality": 70,
            "risk_coverage": 70,
            "clarity": 70,
            "comments": raw[:300]
        }
    return parsed

logger.info("Evaluation function ready.")



@traced("devils_advocate_decision_system")
def devils_advocate_decision_system(
    user_id: str,
    user_question: str,
    decision_options: List[str],
    context: str = ""
) -> Dict[str, Any]:
    # 1. Create new session
    session_id = SESSION_SERVICE.create_session(user_id=user_id)

    # 2. Initial user message
    initial_msg = AgentMessage(
        sender=user_id,
        receiver="system",
        role="user",
        content=f"Q: {user_question}\nOptions: {decision_options}\nContext: {context}",
        metadata={"type": "user_question"}
    )
    SESSION_SERVICE.add_message(session_id, initial_msg)

    # 3. Multi-agent debate
    final_msg = run_sequential_with_loop(
        session_id=session_id,
        user_question=user_question,
        decision_options=decision_options,
        context=context,
        max_iterations=2
    )

    # 4. Add to long-term memory
    MEMORY_BANK.add(
        text=f"Question: {user_question}\nDecision: {final_msg.content}",
        metadata={"user_id": user_id, "session_id": session_id}
    )

    # 5. Evaluate
    eval_result = evaluate_decision(final_msg)

    return {
        "session_id": session_id,
        "final_decision": json.loads(final_msg.content),
        "evaluation": eval_result,
        "metrics": METRICS
    }

logger.info("Top-level decision system function ready.")



USER_ID = "demo_user"

user_question = "Should our startup launch a freemium plan next quarter?"
decision_options = [
    "Launch freemium plan immediately",
    "Run a limited beta before fully launching",
    "Do not launch freemium, keep paid-only"
]
context = (
    "We are a B2B SaaS startup with 500 paying customers. Revenue is stable but growth is slow. "
    "Competitors offer free tiers. We worry about server costs, support load, and brand dilution."
)

# Run the full multi-agent decision system
result = devils_advocate_decision_system(
    user_id=USER_ID,
    user_question=user_question,
    decision_options=decision_options,
    context=context
)

print("=== SESSION ID ===")
print(result["session_id"])

print("\n=== FINAL DECISION (JSON) ===")
print(json.dumps(result["final_decision"], indent=2))

print("\n=== EVALUATION ===")
print(json.dumps(result["evaluation"], indent=2))

print("\n=== METRICS ===")
print(json.dumps(result["metrics"], indent=2))

# Pause/resume demo
save_session_state(result["session_id"])
restored_session_id = load_session_state()
print(f"\nRestored session id: {restored_session_id}")

# MemoryBank search demo
print("\n=== Similar past decisions from MemoryBank ===")
similar = MEMORY_BANK.search("freemium pricing strategy", k=3)
for entry in similar:
    print("- Memory:", entry["metadata"], "text snippet:", entry["text"][:120], "...")



def interactive_loop():
    print("Multi-Agent AI: Devil's Advocate Decision System")
    print("Type 'exit' to quit.")
    user_id = "cli_user"

    while True:
        question = input("\nEnter your decision problem: ")
        if question.lower().strip() == "exit":
            break
        opts_raw = input("Enter options (comma-separated): ")
        options = [o.strip() for o in opts_raw.split(",") if o.strip()]
        ctx = input("Extra context (optional): ")

        result = devils_advocate_decision_system(
            user_id=user_id,
            user_question=question,
            decision_options=options,
            context=ctx
        )
        print("\n=== Final Decision ===")
        print(json.dumps(result["final_decision"], indent=2))
        print("\n=== Evaluation ===")
        print(json.dumps(result["evaluation"], indent=2))

# Uncomment to use manually in a terminal:
# interactive_loop()

print("Interactive loop defined (commented out by default).")


try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:
    FastAPI = None

if FastAPI is not None:
    app = FastAPI(title="Devil's Advocate Decision System")

    class DecisionRequest(BaseModel):
        user_id: str
        question: str
        options: List[str]
        context: str = ""

    class DecisionResponse(BaseModel):
        session_id: str
        final_decision: Dict[str, Any]
        evaluation: Dict[str, Any]
        metrics: Dict[str, Any]

    @app.post("/decide", response_model=DecisionResponse)
    def decide(req: DecisionRequest):
        result = devils_advocate_decision_system(
            user_id=req.user_id,
            user_question=req.question,
            decision_options=req.options,
            context=req.context
        )
        return DecisionResponse(
            session_id=result["session_id"],
            final_decision=result["final_decision"],
            evaluation=result["evaluation"],
            metrics=result["metrics"]
        )

    print("FastAPI app 'app' created. Deploy with uvicorn in a real server.")
else:
    print("FastAPI not installed; deployment example skipped.")


print("=== Current METRICS ===")
for k, v in METRICS.items():
    print(f"{k}: {v}")

print("\nTotal sessions stored:", len(SESSION_SERVICE.sessions))
print("Session IDs:", list(SESSION_SERVICE.sessions.keys()))


