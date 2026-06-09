# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass, field



class SimpleLogger:
    def __init__(self):
        self.spans = []

    def span(self, name: str):
        return SimpleSpan(name, self)

    def log(self, msg: str, **kwargs):
        entry = {"time": time.time(), "msg": msg, "meta": kwargs}
        self.spans.append(entry)
        print(f"[LOG] {msg} | {kwargs}")


class SimpleSpan:
    def __init__(self, name: str, logger: SimpleLogger):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        self.logger.log(f"start:{self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dur = time.time() - self.start
        self.logger.log(f"end:{self.name}", duration_ms=int(dur * 1000))


@dataclass
class Metrics:
    planned_cases: int = 0
    successful_plans: int = 0
    verification_failures: int = 0
    avg_eta_minutes: float = 0.0

    def update_eta(self, new_eta: float):
        if self.planned_cases == 0:
            self.avg_eta_minutes = new_eta
        else:
            self.avg_eta_minutes = (
                self.avg_eta_minutes * (self.planned_cases - 1) + new_eta
            ) / self.planned_cases


logger = SimpleLogger()
metrics = Metrics()



class SessionService:
    """
    Very simple session + long-term memory.

    session_state: per-case state.
    memory_bank: global facts about clinics, roads, etc.
    """

    def __init__(self):
        self.session_state: Dict[str, Dict[str, Any]] = {}
        self.memory_bank: Dict[str, Any] = {
            "clinic_A": {
                "name": "Clinic A",
                "cold_chain": True,
                "working_hours": "09:00-17:00",
            },
            "road_blockages": [],
        }

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return self.session_state.setdefault(session_id, {})

    def update_session(self, session_id: str, key: str, value: Any):
        sess = self.get_session(session_id)
        sess[key] = value

    def compact_context(self, session_id: str) -> str:
        """
        Simple context compaction: keep only key summaries.
        """
        sess = self.get_session(session_id)
        summary = {
            "last_case": sess.get("case_summary"),
            "last_plan": sess.get("plan_summary"),
        }
        return json.dumps(summary)


session_service = SessionService()



class LLMClient:
    """
    Simple abstraction over an LLM.

    - If GEMINI_API_KEY is set and google-generativeai is available,
      you can wire this to Gemini.
    - Otherwise, we return deterministic stub outputs for demo/eval.
    """

    def __init__(self):
        self.use_stub = True
        self.model = None
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.use_stub = False
            except Exception as e:
                print("Falling back to stub LLM:", e)

    def _stub_structured(self, system_prompt: str, user_message: str, extra_context=None):
        """
        Extremely simple, rule-based stub to make the notebook self-contained.
        """
        # You can tweak this logic to look smarter in your demo.
        if "intake agent" in system_prompt.lower():
            return {
                "name": "Test Patient",
                "symptoms": ["cough", "fever"],
                "onset_hours": 24,
                "urgency": "MED",
                "lat": 12.9,
                "lon": 77.6,
                "language": "en",
                "consent": True,
            }
        if "dispatch planner" in system_prompt.lower():
            return {
                "selected_route_id": "road_van",
                "kit_items": ["paracetamol", "cough_syrup", "oral_rehydration_salts"],
                "priority": 3,
                "notes": "Standard respiratory kit. No cold-chain required.",
            }
        if "compliance officer" in system_prompt.lower():
            return {"ok": True, "reasons": []}
        if "patient facing summary" in system_prompt.lower():
            return {
                "summary": "We will send a van with basic medicines. ETA 45 minutes.",
                "language": extra_context.get("case", {}).get("language", "en"),
            }
        return {"raw_text": "stub response"}

    def structured_chat(
        self,
        system_prompt: str,
        user_message: str,
        extra_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        if self.use_stub or self.model is None:
            return self._stub_structured(system_prompt, user_message, extra_context or {})

        # Gemini path (if enabled)
        ctx_json = json.dumps(extra_context or {})
        full_prompt = f"{system_prompt}\n\nContext JSON:\n{ctx_json}\n\nUser: {user_message}"
        resp = self.model.generate_content(full_prompt)
        txt = resp.text
        try:
            return json.loads(txt)
        except Exception:
            return {"raw_text": txt}


llm = LLMClient()



class InventoryClient:
    """
    Mock Inventory API.
    Pretend this is an OpenAPI tool.
    """

    async def get_stock_for_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        # In real life, call /stock?clinic_id=...
        # Here we return fixed stock.
        return {
            "clinic_id": "clinic_A",
            "available_items": [
                "paracetamol",
                "cough_syrup",
                "oral_rehydration_salts",
                "bandage",
            ],
            "out_of_stock": ["insulin"],
        }


class MapsClient:
    """
    Mock Maps/Distance API.
    """

    async def get_route_candidates(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        lat, lon = case.get("lat", 0), case.get("lon", 0)
        # Simple two candidates
        return [
            {
                "route_id": "road_van",
                "mode": "road",
                "eta_minutes": 45,
                "cost_usd": 5.0,
                "distance_km": 25.0,
            },
            {
                "route_id": "drone",
                "mode": "air",
                "eta_minutes": 25,
                "cost_usd": 10.0,
                "distance_km": 25.0,
            },
        ]


class WeatherClient:
    """
    Mock Weather API.
    """

    async def get_weather_summary(self, case: Dict[str, Any]) -> Dict[str, Any]:
        # For demo, say drone is sometimes not allowed.
        return {
            "condition": "moderate_rain",
            "drone_allowed": False,
        }


inventory_client = InventoryClient()
maps_client = MapsClient()
weather_client = WeatherClient()



class IntakeAgent:
    """
    Intake & triage agent (Gemini-backed or stub).
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract_case(self, message: str) -> Dict[str, Any]:
        system = (
            "You are a medical intake agent for a rural dispatch system. "
            "Return ONLY a valid JSON object with keys: "
            "name, symptoms, onset_hours, urgency, lat, lon, language, consent."
        )
        return self.llm.structured_chat(system_prompt=system, user_message=message)


class PlannerAgent:
    """
    Planner that orchestrates tool calls and drafts a plan.
    """

    def __init__(self, llm_client: LLMClient,
                 inventory: InventoryClient,
                 maps: MapsClient,
                 weather: WeatherClient):
        self.llm = llm_client
        self.inventory = inventory
        self.maps = maps
        self.weather = weather

    async def plan_dispatch(self, case: Dict[str, Any]) -> Dict[str, Any]:
        stock = await self.inventory.get_stock_for_case(case)
        routes = await self.maps.get_route_candidates(case)
        wx = await self.weather.get_weather_summary(case)

        system = (
            "You are a dispatch planner for rural healthcare. "
            "You receive a patient case, stock, route options and weather. "
            "Propose a JSON plan with keys: "
            "selected_route_id, kit_items[list], priority (1-5), notes. "
            "Prefer safe, feasible routes; avoid drone if drone_allowed is false."
        )
        return self.llm.structured_chat(
            system_prompt=system,
            user_message="Draft a dispatch plan.",
            extra_context={"case": case, "stock": stock, "routes": routes, "weather": wx},
        )


class RouterAgent:
    """
    Router that evaluates candidate routes.
    """

    def __init__(self, maps_client: MapsClient):
        self.maps = maps_client

    async def score_routes(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        # In a more advanced version, we could adjust scores, penalties etc.
        return await self.maps.get_route_candidates(case)


class VerifierAgent:
    """
    Policy & safety checker.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def verify(self, case: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        system = (
            "You are a compliance officer. Check if the dispatch plan "
            "violates any basic rules: no expired meds, respect cold-chain, "
            "do not send prescription-only meds without doctor approval. "
            "Return JSON with: ok(bool), reasons(list[str])."
        )
        return self.llm.structured_chat(
            system_prompt=system,
            user_message="Verify plan.",
            extra_context={"case": case, "plan": plan},
        )


class NotifierAgent:
    """
    Generates human readable summaries and supports pause/resume.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def notify(self, case: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        system = (
            "You are a patient facing summary bot. Create a short, clear "
            "message about what will happen, the ETA, and what the patient "
            "should prepare. Return JSON with: summary (string), language."
        )
        return self.llm.structured_chat(
            system_prompt=system,
            user_message="Draft patient-friendly summary.",
            extra_context={"case": case, "plan": plan},
        )


intake_agent = IntakeAgent(llm)
planner_agent = PlannerAgent(llm, inventory_client, maps_client, weather_client)
router_agent = RouterAgent(maps_client)
verifier_agent = VerifierAgent(llm)
notifier_agent = NotifierAgent(llm)



import asyncio


@dataclass
class Orchestrator:
    intake: IntakeAgent
    planner: PlannerAgent
    router: RouterAgent
    verifier: VerifierAgent
    notifier: NotifierAgent
    session_service: SessionService
    logger: SimpleLogger
    metrics: Metrics

    async def run_case(self, session_id: str, user_message: str, auto_approve: bool = True):
        with self.logger.span("intake"):
            case = self.intake.extract_case(user_message)
            self.session_service.update_session(session_id, "case", case)
            self.session_service.update_session(session_id, "case_summary", case)

        with self.logger.span("routes_parallel"):
            routes = await self.router.score_routes(case)
            # you can log or store routes if needed
            self.session_service.update_session(session_id, "routes", routes)

        with self.logger.span("planning"):
            plan = await self.planner.plan_dispatch(case)
            self.session_service.update_session(session_id, "raw_plan", plan)

        with self.logger.span("verification"):
            verdict = self.verifier.verify(case, plan)

        if not verdict.get("ok", False):
            self.logger.log("verification_failed", reasons=verdict.get("reasons"))
            self.metrics.verification_failures += 1
            approved = False
        else:
            approved = True

        # Pause / resume concept: require human approval if not auto_approve
        if not auto_approve:
            self.logger.log("paused_for_approval", session_id=session_id)
            # In a real system, you’d persist state and exit here.
            # For demo, we just simulate approval.
            approved = True

        notif = {}
        if approved:
            with self.logger.span("notification"):
                notif = self.notifier.notify(case, plan)

        # Update metrics
        self.metrics.planned_cases += 1
        # derive ETA from chosen route
        chosen_route_id = plan.get("selected_route_id", "")
        chosen_route = next((r for r in routes if r["route_id"] == chosen_route_id), routes[0])
        self.metrics.successful_plans += 1
        self.metrics.update_eta(chosen_route["eta_minutes"])

        # Save compact summaries
        self.session_service.update_session(session_id, "plan_summary", plan)

        return {
            "case": case,
            "routes": routes,
            "plan": plan,
            "verification": verdict,
            "notification": notif,
            "metrics": self.metrics.__dict__,
            "context_compacted": self.session_service.compact_context(session_id),
        }


orchestrator = Orchestrator(
    intake=intake_agent,
    planner=planner_agent,
    router=router_agent,
    verifier=verifier_agent,
    notifier=notifier_agent,
    session_service=session_service,
    logger=logger,
    metrics=metrics,
)



demo_text = "Patient in village 25km from Clinic A with cough and fever for 1 day."

result = await orchestrator.run_case("session_1", demo_text)

print("\n=== CASE ===")
print(json.dumps(result["case"], indent=2))

print("\n=== PLAN ===")
print(json.dumps(result["plan"], indent=2))

print("\n=== VERIFICATION ===")
print(json.dumps(result["verification"], indent=2))

print("\n=== NOTIFICATION ===")
print(json.dumps(result["notification"], indent=2))

print("\n=== METRICS ===")
print(json.dumps(result["metrics"], indent=2))

print("\n=== COMPACTED CONTEXT ===")
print(result["context_compacted"])



SCENARIOS = [
    {
        "id": "rural_stock_normal_weather",
        "input": "40-year old with cough and mild fever in village 25km from Clinic A.",
        "expect": {
            "policy_ok": True,
        },
    },
    {
        "id": "rural_high_urgency",
        "input": "Elderly diabetic with high fever and vomiting 30km from Clinic A.",
        "expect": {
            "policy_ok": True,
        },
    },
]

async def evaluate_scenarios():
    results = []
    for i, sc in enumerate(SCENARIOS, start=1):
        sid = f"eval_{i}"
        res = await orchestrator.run_case(sid, sc["input"])
        policy_ok = res["verification"].get("ok", False)
        passed = policy_ok == sc["expect"]["policy_ok"]
        results.append(
            {
                "id": sc["id"],
                "passed": passed,
                "policy_ok": policy_ok,
                "eta": res["metrics"]["avg_eta_minutes"],
            }
        )
    return results
eval_results = evaluate_scenarios()
eval_results

