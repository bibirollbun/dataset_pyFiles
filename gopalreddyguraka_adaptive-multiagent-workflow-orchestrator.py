# %% [markdown]
# # AMWO Capstone (Hybrid ADK + Fallback) â€” Gemini-2.5-Flash-Lite Everywhere
# Adaptive Multi-Agent Workflow Orchestrator
# **Objective:** Capstone demonstrates a production-capable multi-agent system:
# - ADK-native when available; robust fallback when not
# - Proposer â†’ Solver â†’ Judge (MAE-style evolution, Option A)
# - Separate Validator agent for final QA
# - Orchestrator + Parallel Planner
# - Short-term + Long-term memory
# - Multimodal (text + image) reasoning example
# - Observability (traces), validation tests, and exportable agent bundle
#
# **Run order:** Run each notebook cell (not "Run All" to avoid quota bursts).
#
# **Model:** `gemini-2.5-flash-lite` (used everywhere)

# %% --- Install dependencies (for Colab/Kaggle) ---
import subprocess
import sys
import importlib
import pkg_resources

# Mapping of PyPI package name -> Import name
packages = {
    "google-generativeai": "google.generativeai",
    "nest_asyncio": "nest_asyncio",
    # "google-adk": "google.adk" # Uncomment if you have a valid pip source for ADK
}

def install_package(package_name):
    print(f"â¬‡ Installing/Updating {package_name} ...")
    try:
        # We use -U to ensure we get the version that supports Gemini 2.5
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package_name, "--quiet"])
        print(f"âœ“ {package_name} installed.")
    except subprocess.CalledProcessError:
        print(f"âš ï¸� Could not install {package_name} (might be private or unavailable).")

def check_and_install():
    for package, import_name in packages.items():
        try:
            importlib.import_module(import_name)
            print(f"âœ“ {import_name} already detected.")
        except ImportError:
            install_package(package)

# Run installation
check_and_install()

# Verify Imports specifically for our logic
print("\nğŸ”� Verifying critical modules:")
try:
    import google.generativeai as genai
    print(f"   âœ” google.generativeai version: {genai.__version__}")
except ImportError:
    print("   â�Œ google.generativeai failed to load.")

try:
    import nest_asyncio
    nest_asyncio.apply()
    print("   âœ” nest_asyncio applied.")
except ImportError:
    print("   â�Œ nest_asyncio failed.")

# Check for ADK (Optional/Hybrid)
try:
    import google.adk
    print("   âœ” google.adk found (Hybrid Mode: ENABLED)")
except ImportError:
    print("   â„¹ï¸� google.adk not found (Hybrid Mode: FALLBACK ACTIVE)")


# %% --- Unified API Key Setup (Works in Kaggle / Colab / Local) ---

import os
import sys
import logging
import google.generativeai as genai

def get_api_key():
    """Attempt to load API key from various platform secrets."""
    
    # 1. Try Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        key = secrets.get_secret("GOOGLE_API_KEY")
        if key:
            print("âœ“ Loaded key from Kaggle Secrets")
            return key
    except ImportError:
        pass # Not running on Kaggle
    except Exception as e:
        print(f"âš ï¸� Kaggle secrets found but failed: {e}")

    # 2. Try Google Colab Secrets (modern Colab)
    try:
        from google.colab import userdata
        key = userdata.get("GOOGLE_API_KEY")
        if key:
            print("âœ“ Loaded key from Colab UserData")
            return key
    except ImportError:
        pass # Not running on Colab
    except Exception:
        pass # Key not defined in Colab

    # 3. Try Environment Variables (Local / Docker)
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        print("âœ“ Loaded key from Environment Variable")
        return key

    return None

# Execute Load
GOOGLE_API_KEY = get_api_key()

# Validation & Hard Stop
if not GOOGLE_API_KEY or len(GOOGLE_API_KEY) < 20:
    raise RuntimeError("""
â�Œ Missing or invalid GOOGLE_API_KEY.

Please set the key using one of the following methods:
1. Kaggle: Add 'GOOGLE_API_KEY' to 'Add-ons' -> 'Secrets'
2. Colab: Add 'GOOGLE_API_KEY' to the 'Secrets' (key icon) sidebar
3. Local: export GOOGLE_API_KEY="AIza..."
""")

# Export to environment for libraries that check os.environ (like ADK/LangChain)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Configure the Global Client
genai.configure(api_key=GOOGLE_API_KEY)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info(f"ğŸ”‘ Gemini API key configured. (Key length: {len(GOOGLE_API_KEY)})")


# %% --- Guarded imports & model configuration (ADK detection + Gemini fallback)

import os
import time
import json
import uuid
import logging
import asyncio
import concurrent.futures
import nest_asyncio

# Enable nested async loops (required in notebooks)
nest_asyncio.apply()

# Logging setup once for the notebook session
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AMWO-Capstone")

logger.info("ğŸ”§ Initializing environment and dependency checks...")

# -------------------------
# Attempt ADK imports
# -------------------------
ADK_AVAILABLE = False
try:
    from google.adk.agents import Agent as ADKAgent
    from google.adk.models.google_llm import Gemini as ADKGemini
    from google.adk.runners import InMemoryRunner as ADKRunner
    from google.adk.tools import google_search as ADKGoogleSearch

    ADK_AVAILABLE = True
    logger.info("ğŸ§  Gemini ADK available. Advanced agent workflows enabled.")
except ImportError as e:
    logger.warning("âš  Gemini ADK NOT available. Using standard generative client. (%s)", e)


# -------------------------
# Configure generative AI client
# -------------------------
import google.generativeai as genai

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("â�Œ GOOGLE_API_KEY missing in environment. Cannot initialize Gemini models.")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
logger.info("ğŸ”� google.generativeai client configured successfully.")


# -------------------------
# Model selection
# -------------------------
DEFAULT_MODEL = "gemini-2.5-flash"   # more robust for text + planning
FALLBACK_MODEL = "gemini-1.5-flash"

MODEL_NAME = DEFAULT_MODEL
logger.info("ğŸ¤– Using model: %s", MODEL_NAME)

# Validate model availability (optional first call warm-up)
try:
    _ = genai.GenerativeModel(MODEL_NAME)
    logger.info("âœ¨ Model '%s' verified operational.", MODEL_NAME)
except Exception:
    MODEL_NAME = FALLBACK_MODEL
    logger.warning("âš  Primary model unavailable. Switched to fallback: %s", MODEL_NAME)

logger.info("ğŸš€ Model configuration complete. Ready for agent execution.")



# %% --- Observability & Trace Helpers (Day-4 Enhanced)

from typing import Any, Dict, List
from datetime import datetime

TRACE_LOGS: List[Dict[str, Any]] = []
ARTIFACTS: Dict[str, Any] = {}

def trace(step: str, payload: Dict[str, Any] | None = None, preview_limit: int = 400):
    """
    Records trace events for agent reasoning, debugging, and UI monitoring.
    
    Args:
        step (str): The logical step executed (e.g., "LLM_CALL", "SEARCH", "RESPONSE_PARSE").
        payload (dict): Optional associated metadata or result data.
        preview_limit (int): Truncation size for safe logging.
        
    Returns:
        dict: Trace entry added to the log.
    """
    payload = payload or {}
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "step": step,
        "payload": payload
    }
    
    TRACE_LOGS.append(entry)

    # Safe logging without overwhelming console
    safe_preview = json.dumps(payload, default=str)[:preview_limit]
    logger.info(f"ğŸ§ª TRACE [{step}] â€” {safe_preview}")

    return entry


def add_artifact(name: str, data: Any):
    """
    Saves intermediate outputs (tables, files, plans, JSONs) for UI review or agent use.
    """
    ARTIFACTS[name] = data
    logger.info(f"ğŸ“¦ Artifact saved: {name} ({type(data).__name__})")
    return ARTIFACTS[name]


def get_trace_summary(limit: int = 10):
    """
    Returns a lightweight summary useful for UI display.
    """
    return [
        {"step": t["step"], "timestamp": t["timestamp"]}
        for t in TRACE_LOGS[-limit:]
    ]

logger.info("ğŸ“� Trace and Artifact utilities initialized.")



# %% --- Observability & Trace Helpers (Day-4 Enhanced)

import uuid
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Check if logger exists (from Cell 3), otherwise create a basic one
if "logger" not in globals():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AMWO-Trace")

TRACE_LOGS: List[Dict[str, Any]] = []
ARTIFACTS: Dict[str, Any] = {}

def trace(step: str, payload: Optional[Dict[str, Any]] = None, preview_limit: int = 400):
    """
    Records trace events for agent reasoning, debugging, and UI monitoring.
    
    Args:
        step (str): The logical step executed (e.g., "LLM_CALL", "SEARCH", "RESPONSE_PARSE").
        payload (dict): Optional associated metadata or result data.
        preview_limit (int): Truncation size for safe logging.
        
    Returns:
        dict: Trace entry added to the log.
    """
    payload = payload or {}
    
    # Use standard datetime.now() for compatibility
    timestamp = datetime.now().isoformat()
    
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "step": step,
        "payload": payload
    }
    
    TRACE_LOGS.append(entry)

    # Safe logging without overwhelming console
    try:
        # We use 'default=str' to handle objects that aren't natively JSON serializable
        safe_preview = json.dumps(payload, default=str)[:preview_limit]
    except Exception:
        safe_preview = str(payload)[:preview_limit]
        
    logger.info(f"ğŸ§ª TRACE [{step}] â€” {safe_preview}")

    return entry


def add_artifact(name: str, data: Any):
    """
    Saves intermediate outputs (tables, files, plans, JSONs) for UI review or agent use.
    """
    ARTIFACTS[name] = data
    logger.info(f"ğŸ“¦ Artifact saved: {name} ({type(data).__name__})")
    return ARTIFACTS[name]


def get_trace_summary(limit: int = 10):
    """
    Returns a lightweight summary useful for UI display.
    """
    return [
        {"step": t["step"], "timestamp": t["timestamp"]}
        for t in TRACE_LOGS[-limit:]
    ]

# Test the system immediately to confirm it works
trace("system.trace_init", {"status": "ready", "items_in_log": len(TRACE_LOGS)})
logger.info("ğŸ“� Trace and Artifact utilities initialized.")


# %% --- LLM Wrapper (Tools best-practices: retries, traces, structured responses)

from functools import wraps
from typing import Optional, Dict, Any
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def retry_on_exception(max_tries: int = 5, delay: float = 2.0, backoff: float = 2.0):
    """Retry decorator that handles Safety Blocks AND Rate Limits (429)."""
    def deco(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            tries = 0
            cur_delay = delay
            while True:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)

                    # 1. HANDLE SAFETY BLOCKS (Finish Reason 2)
                    if "finish_reason" in error_str and "2" in error_str:
                        trace("llm.safety.hard_block", {"error": error_str})
                        return {
                            "text": " [SYSTEM NOTE: The model refused to answer this specific query due to safety guidelines. Moving to next step.]",
                            "tokens": {},
                            "model": "safety-fallback",
                            "length": 0
                        }

                    # 2. HANDLE RATE LIMITS (429 / Quota Exceeded)
                    if "429" in error_str or "Quota exceeded" in error_str:
                        print(f"â�³ Rate Limit Hit (429). Pausing for 60 seconds to cool down...")
                        time.sleep(60)
                        # We do NOT increment 'tries' here. We just wait and try again forever until it works.
                        continue

                    # 3. STANDARD RETRY (Network errors, timeouts)
                    tries += 1
                    trace("retry.error", {"attempt": tries, "error": error_str})
                    
                    if tries >= max_tries:
                        raise RuntimeError(f"â�Œ Failed after {max_tries} attempts: {e}")
                    
                    time.sleep(cur_delay)
                    cur_delay *= backoff
        return wrapper
    return deco

class LLMClient:
    """Thin abstraction around Google Generative Models."""
    def __init__(self, model_name: str = MODEL_NAME, default_tokens: int = 2048):
        self.model_name = model_name
        self.default_tokens = default_tokens

    @retry_on_exception(max_tries=5) # Increased retries
    def generate(
        self,
        prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.4
    ) -> Dict[str, Any]:
        
        trace("llm.generate.start", {"model": self.model_name})
        model = genai.GenerativeModel(self.model_name)

        # Safety: BLOCK_NONE
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_output_tokens or self.default_tokens,
                    "temperature": temperature
                },
                safety_settings=safety_settings
            )

            # Handle Blocked Responses Gracefully
            text = ""
            try:
                text = response.text.strip()
            except ValueError:
                trace("llm.safety.triggered", {"feedback": str(response.prompt_feedback)})
                return {
                    "text": " [SYSTEM ALERT: Content blocked by Safety Filters.]",
                    "tokens": {},
                    "model": self.model_name,
                    "length": 0
                }

            return {
                "text": text,
                "tokens": getattr(response, "usage_metadata", {}),
                "model": self.model_name,
                "length": len(text)
            }

        except Exception as e:
            trace("llm.generate.error", {"error": str(e)})
            raise

# Re-initialize the shared client
llm = LLMClient()

# Re-wire agents if they exist
if "factory" in globals():
    proposer_agent = factory.create_proposer("proposer")
    solver_agent = factory.create_solver("solver", tools=[globals().get("FALLBACK_TOOL_GOOGLE_SEARCH")] if globals().get("FALLBACK_TOOL_GOOGLE_SEARCH") else [])
    judge_agent = factory.create_judge("judge")
    validator_agent = factory.create_validator("validator")
    orchestrator = Orchestrator(proposer_agent, solver_agent, judge_agent, validator_agent)
    mae_system = MAESystem(orchestrator, short_memory, long_memory)
    print("âœ… Agents and MAE System re-wired with Rate-Limit-Proof Client.")

logger.info("ğŸ¤– LLMClient ready (Patient Mode: Handles 429 Errors)")


# %% â€” Memory: Short-term & Long-term (Improved, Day-3)
import os
import json
import time
from typing import List, Dict, Optional, Any


class ShortTermMemory:
    """Simple in-runtime memory buffer for conversation context."""
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def add(self, role: str, content: Dict[str, Any]) -> Dict[str, Any]:
        record = {"role": role, "content": content, "ts": time.time()}
        self.history.append(record)
        trace("memory.short.add", {"role": role, "records": len(self.history)})
        return record

    def last(self, role: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return most recent message optionally filtered by role."""
        for rec in reversed(self.history):
            if role is None or rec["role"] == role:
                return rec
        return None


class LongTermMemory:
    """Persistent memory across sessions stored locally in JSON."""
    def __init__(self, filename: str = "longterm_memory.json"):
        self.filename = filename
        self.store: List[Dict[str, Any]] = []

        # Load if available, otherwise initialize safely
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    self.store = json.load(f)
            except Exception as e:
                logger.error(f"Corrupted memory file recovered: {e}")
                self.store = []
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(self.store, f)

        trace("memory.long.init", {"file": self.filename, "records": len(self.store)})

    def insert(self, obj: Dict[str, Any]):
        """Insert new memory entry avoiding duplicates."""
        # Avoid storing repeated entries (simple dedupe)
        if any(json.dumps(obj) == json.dumps(e["obj"]) for e in self.store):
            trace("memory.long.skip_duplicate", {"obj": obj})
            return

        entry = {"obj": obj, "ts": time.time()}
        self.store.append(entry)

        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.store, f, indent=2, default=str)

        trace("memory.long.insert", {"records": len(self.store)})

    def query(self, keyword: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Semantic-like search using substring lookup."""
        keyword = keyword.lower()
        hits = [
            rec for rec in reversed(self.store)
            if keyword in json.dumps(rec, ensure_ascii=False).lower()
        ]
        return hits[:top_k]


short_memory = ShortTermMemory()
long_memory = LongTermMemory()
 
trace("memory.init", {"short": True, "long_file": long_memory.filename})



# %% â€” Agent implementations: Proposer, Solver, Judge, Validator (Improved: Day-2, Day-4, Day-5)

import re, json
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Helper to safely extract text from the LLMClient dict response
def get_text(llm_response: Any) -> str:
    if isinstance(llm_response, dict):
        return llm_response.get("text", "")
    return str(llm_response)

# ---------------------- PROPOSER ----------------------
@dataclass
class ProposerAgent:
    name: str
    llm: LLMClient

    def propose(self, topic: str, n: int = 3) -> List[Dict[str, str]]:
        trace("proposer.start", {"topic": topic, "n": n})

        prompt = (
            f"You are a creative question proposer.\n"
            f"Topic: {topic}\n\n"
            f"Generate {n} challenging but solvable short questions.\n"
            f"Rules:\n"
            f"- One question per line\n"
            f"- Short and clear\n"
            f"- Numbered output allowed\n"
        )

        resp = self.llm.generate(prompt)
        output = get_text(resp) # Fix: Extract text from dict
        
        lines = [l.strip() for l in (output or "").splitlines() if l.strip()]

        # Clean question lines
        questions = []
        for line in lines:
            q = re.sub(r'^\d+[\.\)]\s*', '', line)
            if q:
                questions.append(q)

        # Fallback if fewer than n lines
        if len(questions) < n:
            fallback_sentences = [
                s.strip()
                for s in re.split(r'[.?!]\s+', output or "")
                if s.strip()
            ]
            for s in fallback_sentences:
                if s not in questions:
                    questions.append(s)
                    if len(questions) >= n:
                        break

        questions = questions[:n]
        result = [{"id": str(uuid.uuid4()), "text": q} for q in questions]

        # Use short_memory if it exists globally
        if "short_memory" in globals():
            short_memory.add("proposals", result)
            
        trace("proposer.done", {"generated": len(result)})

        return result


# ---------------------- SOLVER ----------------------
@dataclass
class SolverAgent:
    name: str
    llm: LLMClient
    tools: Optional[list] = field(default_factory=list)

    def solve(self, question: str, use_search: bool = False) -> str:
        trace("solver.start", {"q_len": len(question), "use_search": use_search})
        context = ""

        # Optional tool-assisted search
        # We check if the global fallback tool exists before calling it
        search_tool = globals().get("FALLBACK_TOOL_GOOGLE_SEARCH")
        
        if use_search and search_tool:
            try:
                results = search_tool.call(question, top_k=3)
                ctx_lines = [f"- {r['title']}: {r.get('snippet','')}" for r in results]
                context = "\n\nSearch results:\n" + "\n".join(ctx_lines)
            except Exception as e:
                trace("solver.search_error", {"error": str(e)})

        prompt = (
            "You are a precise problem solver. Provide concise reasoning and a direct answer.\n"
            f"Question: {question}\n\n"
            f"{context}\n\n"
            "Final Answer (clear and short):"
        )

        resp = self.llm.generate(prompt)
        answer = get_text(resp) # Fix: Extract text from dict
        
        if "short_memory" in globals():
            short_memory.add("answers", {"question": question, "answer": answer})
            
        trace("solver.done", {"answer_len": len(answer or '')})

        return answer


# ---------------------- JUDGE ----------------------
@dataclass
class JudgeAgent:
    name: str
    llm: LLMClient

    def judge(self, question: str, answer: str) -> Dict[str, Any]:
        trace("judge.start", {"question_len": len(question), "answer_len": len(answer)})

        prompt = (
            "You are a fair evaluator. Assess the answer quality.\n"
            "Return ONLY valid JSON with keys:\n"
            "score: 0-100,\nfeedback: short constructive text,\ndifficulty: integer 1-10.\n\n"
            f"Question: {question}\n\nAnswer: {answer}\n\n"
            "Respond ONLY with JSON:"
        )

        resp = self.llm.generate(prompt)
        output = get_text(resp) # Fix: Extract text from dict

        try:
            json_block = re.search(r'\{.*\}', output, flags=re.DOTALL)
            parsed = json.loads(json_block.group(0)) if json_block else json.loads(output)

            if "short_memory" in globals():
                short_memory.add("evaluations", {"question": question, "eval": parsed})
                
            trace("judge.done", {"score": parsed.get("score")})

            return parsed

        except Exception as e:
            trace("judge.parse_error", {"error": str(e), "raw": (output or "")[:200]})
            return {"score": 50, "feedback": "Auto-fallback evaluation", "difficulty": 5}


# ---------------------- VALIDATOR ----------------------
@dataclass
class ValidatorAgent:
    name: str
    llm: LLMClient

    def validate(self, final_output: str, schema: Optional[dict] = None) -> Dict[str, Any]:
        trace("validator.start", {"output_len": len(final_output or "")})

        instructions = (
            "You are a strict validator reviewing whether output meets format and structural rules.\n"
            "If a schema dictionary is provided, verify all fields exist.\n"
            "Return ONLY JSON with keys:\n"
            "valid: true/false,\nissues: list of issues,\nsummary: short explanation.\n\n"
        )

        prompt = f"{instructions}Final Output:\n{final_output}\n\nJSON response only:"

        resp = self.llm.generate(prompt)
        output = get_text(resp) # Fix: Extract text from dict

        try:
            json_block = re.search(r'\{.*\}', output, flags=re.DOTALL)
            parsed = json.loads(json_block.group(0)) if json_block else json.loads(output)

            trace("validator.done", {"valid": parsed.get("valid")})
            return parsed

        except Exception as e:
            trace("validator.parse_error", {"error": str(e), "raw": (output or "")[:200]})
            ok = bool(final_output and len(final_output) > 20)
            return {"valid": ok, "issues": [] if ok else ["Output too short"], "summary": "Fallback minimal evaluation"}


# %% â€” Orchestrator + Parallel Planner (Improved: Day-2, Day-5 Parallel Thinking)
from typing import Dict, Any, List
import concurrent.futures
import asyncio

class Orchestrator:
    def __init__(
        self,
        proposer: 'ProposerAgent', # Use forward references (strings) just in case
        solver: 'SolverAgent',
        judge: 'JudgeAgent',
        validator: 'ValidatorAgent' = None
    ):
        self.proposer = proposer
        self.solver = solver
        self.judge = judge
        self.validator = validator

    async def run_round_parallel(
        self,
        topic: str,
        questions_per_round: int = 3,
        parallelism: int = 3,
        use_search: bool = False
    ) -> Dict[str, Any]:

        trace("orch.round.start", {"topic": topic, "parallelism": parallelism})

        # Step 1: Generate proposals
        questions = self.proposer.propose(topic, n=questions_per_round)

        loop = asyncio.get_event_loop()
        answers: Dict[str, Dict[str, str]] = {}
        evaluations: Dict[str, Any] = {}

        # ---------- PARALLEL SOLVER ----------
        trace("orch.solve.parallel.start", {"count": len(questions)})

        async def solve_all():
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
                tasks = [
                    loop.run_in_executor(pool, self.solver.solve, q['text'], use_search)
                    for q in questions
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results

        solve_results = await solve_all()

        for q, result in zip(questions, solve_results):
            if isinstance(result, Exception):
                trace("orch.solve.error", {"question": q["text"], "error": str(result)})
                answers[q["id"]] = {"question": q["text"], "answer": "Error during solving"}
            else:
                answers[q["id"]] = {"question": q["text"], "answer": result}

        trace("orch.solve.parallel.done", {"answers_count": len(answers)})

        # ---------- PARALLEL JUDGE ----------
        trace("orch.judge.parallel.start", {"count": len(answers)})

        async def judge_all():
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
                tasks = [
                    loop.run_in_executor(pool, self.judge.judge, qa["question"], qa["answer"])
                    for qa in answers.values()
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results

        judge_results = await judge_all()

        for (qid, qa), result in zip(list(answers.items()), judge_results):
            if isinstance(result, Exception):
                trace("orch.judge.error", {"qid": qid, "error": str(result)})
                evaluations[qid] = {"score": 0, "feedback": "Error evaluating", "difficulty": 1}
            else:
                evaluations[qid] = result

        trace("orch.judge.parallel.done", {"evals_count": len(evaluations)})
        trace("orch.round.complete", {"topic": topic, "total": len(questions)})

        return {"questions": questions, "answers": answers, "evaluations": evaluations}


    # ---------- VALIDATOR WRAPPER ----------
    def run_validator(self, final_output: str, schema: dict = None) -> Dict[str, Any]:
        if not self.validator:
            trace("orch.validate.skipped", {"reason": "no_validator"})
            return {"valid": True, "issues": [], "summary": "Validator not configured"}
        return self.validator.validate(final_output, schema)


# %% â€” MAE Evolution Loop (Improved Automatic Continuous Evolution)
from typing import List, Dict, Any

class MAESystem:
    def __init__(self, orchestrator: 'Orchestrator', short_mem: 'ShortTermMemory', long_mem: 'LongTermMemory'):
        self.orch = orchestrator
        self.short_mem = short_mem
        self.long_mem = long_mem

    async def run(
        self,
        topic: str,
        max_rounds: int = 5,
        questions_per_round: int = 3,
        parallelism: int = 3,
        converge_score: float = 90.0,
        use_search: bool = False
    ) -> List[Dict[str, Any]]:

        trace("mae.run.start", {
            "topic": topic,
            "max_rounds": max_rounds,
            "converge_score": converge_score
        })

        evolution_history: List[Dict[str, Any]] = []

        for round_num in range(1, max_rounds + 1):
            trace("mae.round.begin", {"round": round_num})

            # Execute PSJ pipeline
            result = await self.orch.run_round_parallel(
                topic,
                questions_per_round,
                parallelism,
                use_search
            )
            evolution_history.append(result)

            # Evaluate scoring statistics
            scores = []
            for qid, ev in result["evaluations"].items():
                try:
                    score_value = float(ev.get("score", 0))
                except Exception:
                    score_value = 0.0
                scores.append(score_value)

            avg_score = sum(scores) / len(scores) if scores else 0.0

            logger.info(f"Round {round_num} -- Average Score: {avg_score:.2f}")
            trace("mae.round.score", {"round": round_num, "avg_score": avg_score})

            # Persist learning to long-term memory
            self.long_mem.insert({
                "round": round_num,
                "topic": topic,
                "result": result,
                "avg_score": avg_score
            })

            # --- Convergence condition ---
            if avg_score >= converge_score:
                logger.info(f"Converged at round {round_num} with score {avg_score:.2f}")
                trace("mae.converged", {"round": round_num, "score": avg_score})
                break

            # --- Self-evolution using feedback ---
            # Robustly handle feedback collection
            feedbacks = []
            for ev in result["evaluations"].values():
                fb = ev.get("feedback", "")
                feedbacks.append(str(fb))
            
            aggregated_feedback = " ".join(feedbacks).lower()

            trace("mae.feedback.aggregate", {"len": len(aggregated_feedback)})

            if "too easy" in aggregated_feedback:
                topic += " (increase difficulty)"
            elif "too hard" in aggregated_feedback:
                topic += " (simplify)"
            elif "unclear" in aggregated_feedback:
                topic += " (clarify topic)"
            elif "irrelevant" in aggregated_feedback:
                topic += " (re-focus core concept)"

            trace("mae.topic.update", {"new_topic": topic})

        trace("mae.run.complete", {"rounds_completed": len(evolution_history)})
        return evolution_history


# %% â€” Instantiate agents & systems (wiring) + ADK-native fallback wiring
from typing import Optional, List, Dict, Any

# --- Fix: Tool may not be available if ADK is available and fallback didn't define it
try:
    _ToolType = Tool          # fallback Tool
except NameError:
    _ToolType = Any           # ADK mode or Tool not defined

def _normalize_tools(tools: Optional[List[Any]]) -> List[Any]:
    """Ensure tools is always a list for easier downstream handling."""
    if not tools:
        return []
    return tools


class AgentFactory:
    """
    Factory to create fallback agents (or ADK-wrapped ones if needed).
    """
    def __init__(self, llm_client: 'LLMClient', model_name: str = MODEL_NAME):
        self.llm = llm_client
        self.model_name = model_name

    def create_proposer(self, name: str = "proposer") -> 'ProposerAgent':
        return ProposerAgent(name=name, llm=self.llm)

    def create_solver(self, name: str = "solver", tools: Optional[List[Any]] = None) -> 'SolverAgent':
        return SolverAgent(name=name, llm=self.llm, tools=_normalize_tools(tools))

    def create_judge(self, name: str = "judge") -> 'JudgeAgent':
        return JudgeAgent(name=name, llm=self.llm)

    def create_validator(self, name: str = "validator") -> 'ValidatorAgent':
        return ValidatorAgent(name=name, llm=self.llm)


# ------------------ Instantiate Agents ------------------
factory = AgentFactory(llm_client=llm, model_name=MODEL_NAME)

# Fix: Safely check if the search tool exists in globals before trying to use it
search_tools = []
if "FALLBACK_TOOL_GOOGLE_SEARCH" in globals():
    search_tools = [globals()["FALLBACK_TOOL_GOOGLE_SEARCH"]]

proposer_agent = factory.create_proposer("proposer")
solver_agent = factory.create_solver("solver", tools=search_tools)
judge_agent = factory.create_judge("judge")
validator_agent = factory.create_validator("validator")


# ------------------ Orchestrator + MAE ------------------
orchestrator = Orchestrator(proposer_agent, solver_agent, judge_agent, validator_agent)
mae_system = MAESystem(orchestrator, short_memory, long_memory)


# ------------------ ADK Optional Runner ------------------
adk_runner = None
adk_root_agent = None

# Only try ADK if the flag is true AND the libraries are actually imported
if globals().get("ADK_AVAILABLE", False):
    try:
        from google.adk.models.google_llm import Gemini as ADKGemini
        from google.adk.agents import Agent as ADKAgent
        from google.adk.runners import InMemoryRunner as ADKRunner
        
        adk_model = ADKGemini(model=MODEL_NAME)
        adk_root_agent = ADKAgent(name="amwo_root", model=adk_model, instruction="Root orchestrator agent")
        adk_runner = ADKRunner(agent=adk_root_agent)
        trace("adk.root.ready", {"agent": "amwo_root"})
    except ImportError:
        logger.warning("ADK libraries missing despite ADK_AVAILABLE=True. Ignoring ADK.")
    except Exception as e:
        trace("adk.init.failed", {"error": str(e)})
        logger.warning("ADK runner creation failed, falling back: %s", e)
        adk_runner = None


# ------------------ System Summary ------------------
system = {
    "proposer": proposer_agent,
    "solver": solver_agent,
    "judge": judge_agent,
    "validator": validator_agent,
    "orchestrator": orchestrator,
    "mae": mae_system,
    "adk_runner": adk_runner,
    "adk_available": globals().get("ADK_AVAILABLE", False),
}

trace("system.wired", {"adk_available": system["adk_available"]})
logger.info("Agents & orchestrator instantiated.")

print({
    "proposer": proposer_agent.name,
    "solver": solver_agent.name,
    "judge": judge_agent.name,
    "validator": validator_agent.name,
    "adk_available": system["adk_available"]
})


# %% â€” Run a demo MAE session (Crisis Management Mode - Super Safe Version)
import os
import json
import time
import asyncio

# --- CONFIGURATION: The "Coffee Shop" Crisis ---
TOPIC = """
[SIMULATION: BUSINESS LOGISTICS CASE STUDY]

Develop a specialized Crisis Response Strategy for 'BeanStream' (a premium coffee chain) facing three simultaneous events:
1. LOGISTICS FAILURE: A shipping container with our seasonal beans is stuck in customs due to a paperwork error (missing Form 42-A).
2. MARKETING ERROR: We printed 10,000 holiday cups with a typo ("Happy Holidogs") instead of "Holidays".
3. STAFFING: 20% of our baristas are out sick with the flu during the busiest week of the year.

TASK:
- Round 1: Immediate Damage Control (Next 24 Hours). Do we recall the cups? How do we staff the stores?
- Round 2: Logistics & Supply (Next 7 Days). How do we get the beans released from customs?
- Round 3: Future Proofing. Better inventory management and QA processes.

CONSTRAINTS:
- Budget is tight.
- We cannot close stores.
- Customer satisfaction is the goal.
"""

ROUNDS = 2
PARALLELISM = 3
CONVERGE_SCORE = 85.0

# --- CRITICAL FIX: Define SAVE_DIR ---
SAVE_DIR = "output"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"ğŸš€ Starting AMWO Business Crisis Simulation")
print(f"ğŸ“‰ Scenario: Coffee Shop Logistics")

async def main_demo():
    try:
        # Check if system is wired correctly
        if "mae_system" not in globals() or not mae_system:
            raise RuntimeError("â�Œ MAE System not initialized! Did you run the previous Wiring cell?")

        start_time = time.time()

        # RUN THE AGENTS
        history = await mae_system.run(
            topic=TOPIC,
            max_rounds=ROUNDS,
            questions_per_round=3,
            parallelism=PARALLELISM,
            converge_score=CONVERGE_SCORE,
            use_search=True
        )

        duration = time.time() - start_time
        print(f"\nâœ… Crisis Strategy Complete in {duration:.2f} seconds.")

        # Save the full history
        outfile = os.path.join(SAVE_DIR, "crisis_strategy_history.json")
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)

        print(f"ğŸ“‚ Strategic Plan saved to: {outfile}")

        return history

    except Exception as e:
        print(f"\nâ�Œ Critical Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(main_demo())
        
        # You MUST await the task, otherwise "Run All" finishes 
        # before the agents are done!
        await task 
        
    except RuntimeError:
        # No loop running, start a new one
        asyncio.run(main_demo())


# %% â€” Export artifacts & README (Deployment prep - FIXED VERSION)
import os
import json
import time
import shutil

def export_bundle(run_id: str = "run_latest"):
    """
    Bundles all outputs (traces, artifacts, logs) into a ZIP for download.
    """
    # 1. Setup Export Directory
    export_dir = os.path.join(SAVE_DIR, "export_bundle")
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)
    
    # 2. Save Trace Logs
    trace_file = os.path.join(export_dir, "system_traces.json")
    with open(trace_file, "w", encoding="utf-8") as f:
        # Check if TRACE_LOGS exists globally
        if "TRACE_LOGS" in globals():
            json.dump(globals()["TRACE_LOGS"], f, indent=2, default=str)
        else:
            f.write("[]")
            
    # 3. Save Long-Term Memory
    mem_file = os.path.join(export_dir, "longterm_memory.json")
    if "long_memory" in globals():
        with open(mem_file, "w", encoding="utf-8") as f:
            json.dump(long_memory.store, f, indent=2, default=str)

    # 4. Copy any generated outputs (text/json reports)
    for filename in os.listdir(SAVE_DIR):
        if filename.endswith(".json") or filename.endswith(".txt") or filename.endswith(".md"):
            src = os.path.join(SAVE_DIR, filename)
            dst = os.path.join(export_dir, filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    # 5. Create ZIP Archive
    zip_filename = os.path.join(SAVE_DIR, "amwo_agent_bundle")
    shutil.make_archive(zip_filename, 'zip', export_dir)
    
    print(f"ğŸ“¦ Bundle created successfully!")
    print(f"ğŸ‘‰ Download: {zip_filename}.zip")

# Execute Export
try:
    export_bundle()
except Exception as e:
    print(f"âš ï¸� Export partial failure: {e}")


# %% â€” ADK-native conversion helper (robust version) + optional ADK-run demo
from typing import Any, Dict

def convert_to_adk_manifest(agent_system: Dict[str, Any]):
    """
    Mock function to demonstrate how one might export this agent 
    to a Google Cloud Agent Builder manifest format.
    """
    if not globals().get("ADK_AVAILABLE", False):
        print("â„¹ï¸� ADK not enabled. Skipping manifest generation.")
        return

    manifest = {
        "display_name": "AMWO Crisis Agent",
        "agents": {
            "proposer": {"goal": "Generate scenarios"},
            "solver": {"goal": "Solve problems"},
            "judge": {"goal": "Evaluate solutions"}
        }
    }
    
    # Save dummy manifest
    with open(os.path.join(SAVE_DIR, "adk_manifest_preview.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("âœ… Generated ADK manifest preview.")

# Run conversion check
convert_to_adk_manifest(globals().get("system", {}))


# %% â€” Final README (json + markdown) saved to output (robust)
import os
import time

# --- Setup Paths ---
readme_path = os.path.join(SAVE_DIR, "README.md")

# --- Retrieve runtime stats ---
topic = globals().get("TOPIC", "General Run")
model = globals().get("MODEL_NAME", "Unknown")
traces = len(globals().get("TRACE_LOGS", []))
date_str = time.strftime("%Y-%m-%d %H:%M:%S")

# --- Create Content (No indentation required here) ---
content = f"""# AMWO Capstone Project Report
**Generated by Adaptive Multi-Agent Workflow Orchestrator**

## ğŸ“Š Run Summary
- **Date:** {date_str}
- **Topic:** {topic}
- **Model Engine:** `{model}`
- **Total Operations (Traces):** {traces}

## ğŸ¤– System Architecture
This system utilizes a **Proposer-Solver-Judge (PSJ)** architecture with an evolutionary feedback loop.
1. **Proposer:** Breaks complex topics into sub-problems.
2. **Solver:** Generates solutions using Search tools (if enabled).
3. **Judge:** Scores solutions 0-100 against constraints.
4. **Orchestrator:** Manages parallel execution threads.

## ğŸ“‚ Output Files
- `crisis_strategy_history.json`: The full Q&A conversation logs.
- `long_term_memory.json`: The agent's learned context.
- `system_traces.json`: Debug logs for every API call.

*Generated via Kaggle/Colab Notebook*
"""

# --- Save File ---
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"ğŸ“„ README generated at: {readme_path}")


# %% [markdown]
# ## Notebook Complete ğŸ��
# 
# **Next Steps:**
# 1. Check the **Output** tab (right sidebar in Kaggle, file browser in Colab).
# 2. Download **`amwo_agent_bundle.zip`**.
# 3. Unzip to see your Agents' full strategic plan!

