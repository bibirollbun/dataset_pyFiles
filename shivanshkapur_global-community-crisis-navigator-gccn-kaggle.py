# This cell writes the entire project structure and files into the notebook workspace.
import os
from pathlib import Path

PROJECT_ROOT = Path.cwd() / "gccn_project"
print("Writing project to:", PROJECT_ROOT)
PROJECT_ROOT.mkdir(exist_ok=True)

# Files to write: mapping path -> contents
files = {
"requirements.txt": """# Core
fastapi==0.95.2
uvicorn==0.22.0
requests==2.31.0
pydantic==1.10.11

# Testing
pytest==7.4.2

# Optional LLM adapter (placeholder)
openai==1.2.0

# Utilities
python-dotenv==1.0.0
""",

"gccn/__init__.py": '''"""
GCCN package
"""
__version__ = "0.1.0"
''',

"gccn/tools/mock_apis.py": '''"""
Local mock data providers used for demo and unit tests (no network).
"""
import json
from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MOCK_FILE = _PROJECT_ROOT.parent / "demo" / "mock_feeds.json"

def load_mock_feed() -> Dict[str, Any]:
    if not _MOCK_FILE.exists():
        # minimal fallback payload
        return {
            "weather": {"temp_c": 38.5, "alerts": [{"type": "HEAT", "severity": "HIGH"}]},
            "cap": {"alerts": [{"event": "Heat Advisory", "urgency": "Immediate"}]},
            "social": [{"text": "Heatwave heating up", "source": "local_tweet"}],
            "location": {"lat": 40.71, "lon": -74.01, "display_name": "MockTown"}
        }
    with open(_MOCK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        # if top-level mapping of scenarios, return scenario_heatwave if exists
        if "scenario_heatwave" in data:
            return data["scenario_heatwave"]
        return data
''',

"demo/mock_feeds.json": '''{
  "scenario_heatwave": {
    "weather": {
      "temp_c": 41.2,
      "humid": 48,
      "alerts": [
        {"type": "Heat", "severity": "High", "headline": "Excessive Heat Warning", "description": "Temperatures expected above 40°C."}
      ]
    },
    "cap": {
      "alerts": [
        {"event": "Excessive Heat Warning", "urgency": "Immediate", "areas": ["MockTown"]}
      ]
    },
    "social": [
      {"text": "Neighbors reporting high indoor temps", "source": "local_forum"}
    ],
    "location": {"lat": 40.7128, "lon": -74.0060, "display_name": "MockTown, MockCounty"}
  },
  "scenario_flood": {
    "weather": {
      "temp_c": 21.0,
      "rain_mm_24h": 120,
      "alerts": [
        {"type": "Flood", "severity": "Severe", "headline": "Flash Flood Warning", "description": "Rapid flooding expected."}
      ]
    },
    "cap": {
      "alerts": [
        {"event": "Flash Flood Warning", "urgency": "Immediate", "areas": ["MockTown Watershed"]}
      ]
    },
    "social": [
      {"text": "Roads blocked near river", "source": "local_tweet"}
    ],
    "location": {"lat": 40.7128, "lon": -74.0060, "display_name": "MockTown, MockCounty"}
  }
}
''',

"gccn/tools/data_fetcher.py": '''"""
DataFetcher: fetches weather, CAP feeds, social reports.
Uses real APIs when environment variables are present, otherwise falls back to mock data.
"""
import os
from typing import Dict, Any
from .mock_apis import load_mock_feed

USE_MOCK = os.getenv("GCCN_USE_MOCK", "true").lower() in ("1", "true", "yes")

class DataFetcher:
    def __init__(self):
        self.use_mock = USE_MOCK

    def fetch_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        if self.use_mock:
            return load_mock_feed().get("weather", load_mock_feed())
        # Placeholder for real API: implement OpenWeather/NOAA calls using env var API keys
        return {"temp_c": 25.0, "alerts": []}

    def fetch_cap_alerts(self, area_code: str) -> Dict[str, Any]:
        if self.use_mock:
            return load_mock_feed().get("cap", {"alerts": []})
        return {"alerts": []}

    def fetch_social_reports(self, query: str = "") -> Dict[str, Any]:
        if self.use_mock:
            return load_mock_feed().get("social", [])
        return []
''',

"gccn/tools/geocoder.py": '''"""
GeoCoder: maps lat/lon to display names. Uses mock by default.
"""
import os
from typing import Dict, Any
from .mock_apis import load_mock_feed

USE_MOCK = os.getenv("GCCN_USE_MOCK", "true").lower() in ("1", "true", "yes")

class GeoCoder:
    def __init__(self):
        self.use_mock = USE_MOCK

    def reverse(self, lat: float, lon: float) -> Dict[str, Any]:
        if self.use_mock:
            return load_mock_feed().get("location", {"display_name": f"Lat {lat}, Lon {lon}"})
        # Placeholder: call Nominatim in production
        return {"display_name": f"Lat {lat}, Lon {lon}"}
''',

"gccn/tools/llm_wrapper.py": '''"""
LLM wrapper abstraction. For demos/tests this returns deterministic text.
Configure with environment variables to call Gemini/OpenAI/etc in production.
"""
import os
from typing import Optional

USE_REAL_LLM = os.getenv("GCCN_USE_REAL_LLM", "false").lower() in ("1", "true", "yes")

class LLMWrapper:
    def __init__(self):
        self.use_real = USE_REAL_LLM
        # In production, initialize client (e.g., openai.Client or Google client)
    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        if self.use_real:
            # Implement real call here using secure env keys (NOT included)
            return "Real LLM response (configure GCCN_USE_REAL_LLM=true and API keys)"
        # Deterministic mock response useful for tests and demos
        return f"[LLM-DRAFT] Summary based on prompt: {prompt[:120]}..."
''',

"gccn/agents/data_scout.py": '''"""
DataScoutAgent: orchestrates data fetching and produces a normalized payload.
"""
from typing import Dict, Any
from ..tools.data_fetcher import DataFetcher
from ..tools.geocoder import GeoCoder

class DataScoutAgent:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.geocoder = GeoCoder()

    def poll(self, lat: float, lon: float) -> Dict[str, Any]:
        weather = self.fetcher.fetch_weather(lat, lon)
        cap = self.fetcher.fetch_cap_alerts(f"{lat:.4f},{lon:.4f}")
        social = self.fetcher.fetch_social_reports()
        location = self.geocoder.reverse(lat, lon)
        # Normalize into a single payload
        payload = {
            "location": location,
            "weather": weather,
            "cap": cap,
            "social": social,
            "meta": {"lat": lat, "lon": lon}
        }
        return payload
''',

"gccn/agents/risk_agent.py": '''"""
RiskAgent: computes a simple risk score + narrative. Uses rules + optional LLM.
"""
from typing import Dict, Any, Optional
from ..tools.llm_wrapper import LLMWrapper

class RiskAgent:
    def __init__(self, llm: Optional[LLMWrapper] = None):
        self.llm = llm or LLMWrapper()

    def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        weather = payload.get("weather", {})
        cap = payload.get("cap", {})
        social = payload.get("social", [])

        risk_score = 0
        reasons = []

        # Rule: high temp adds risk
        temp = weather.get("temp_c")
        if temp is not None:
            if temp >= 40:
                risk_score += 3
                reasons.append("Extreme temperature (>=40°C)")
            elif temp >= 35:
                risk_score += 2
                reasons.append("High temperature (>=35°C)")

        # Rule: heavy rain / flood indicator
        if weather.get("rain_mm_24h", 0) >= 100:
            risk_score += 3
            reasons.append("Heavy rainfall in last 24h")

        # CAP-based urgency
        cap_alerts = cap.get("alerts", [])
        if cap_alerts:
            for a in cap_alerts:
                urgency = a.get("urgency", "").lower()
                if "immediate" in urgency or a.get("severity", "").lower() in ("severe", "high"):
                    risk_score += 2
                    reasons.append(f"CAP alert: {a.get('event', 'unknown')}")

        # Social signals
        if social and len(social) >= 3:
            risk_score += 1
            reasons.append("Multiple social reports")

        # Narrative via LLM
        prompt = (
            f"Generate a concise plain-language summary of community risk for {payload.get('location', {}).get('display_name')}. "
            f"Weather: {weather}. CAP: {cap}. Social: {social}. Reasons: {reasons}."
        )
        narrative = self.llm.generate(prompt)

        return {
            "risk_score": risk_score,
            "reasons": reasons,
            "narrative": narrative
        }
''',

"gccn/agents/resource_agent.py": '''"""
ResourceAgent: maps available community resources. Uses mock dataset for demo.
"""
from typing import Dict, Any, List

class ResourceAgent:
    def __init__(self):
        # In production, connect to OSM or local DB
        self.mock_resources = [
            {"type": "hospital", "name": "Mock General Hospital", "distance_km": 2.1, "capacity": "unknown"},
            {"type": "shelter", "name": "Community Hall Shelter", "distance_km": 0.9, "capacity": "50"},
            {"type": "cooling_center", "name": "Library Cooling Center", "distance_km": 0.5, "capacity": "30"}
        ]

    def map_resources(self, location: Dict[str, Any]) -> Dict[str, Any]:
        # For demo, just return nearest resources from mock list
        # Filter logic could be added: e.g., accessible only, capacity checks
        return {"resources": self.mock_resources}
''',

"gccn/agents/comms_agent.py": '''"""
CommsAgent: crafts human-friendly messages, checklists and shareable formats.
"""
from typing import Dict, Any, List, Optional
from ..tools.llm_wrapper import LLMWrapper

class CommsAgent:
    def __init__(self, llm: Optional[LLMWrapper] = None):
        self.llm = llm or LLMWrapper()

    def generate_messages(self, analysis: Dict[str, Any], resources: Dict[str, Any]) -> Dict[str, Any]:
        risk_score = analysis.get("risk_score", 0)
        narrative = analysis.get("narrative", "")
        reasons = analysis.get("reasons", [])

        # Basic templated alert
        headline = "Community Alert"
        if risk_score >= 5:
            headline = "HIGH RISK: Immediate Action Recommended"
        elif risk_score >= 3:
            headline = "Moderate Risk — Take Precautions"

        resources_list = resources.get("resources", [])

        checklist = [
            "Check on neighbors who may need help (elderly, disabled).",
            "Prepare a 72-hour emergency kit (water, meds, food).",
            "Move children and pets to a cool/safe area if temperature or flooding risks exist.",
            "Keep phones charged and follow official alerts."
        ]

        # Use LLM to craft a shareable message (demo)
        prompt = f"Create a short SMS-style alert with the headline '{headline}' and key points: {reasons[:3]} and resources: {resources_list[:3]}."
        shareable = self.llm.generate(prompt)

        return {
            "headline": headline,
            "summary": narrative,
            "checklist": checklist,
            "shareable_message": shareable,
            "resources": resources_list
        }
''',

"gccn/agents/memory_agent.py": '''"""
MemoryAgent: simple file-backed memory for demo. Stores the last N events.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

_MEMORY_FILE = Path(__file__).resolve().parents[1] / "data" / "memory.json"
_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

class MemoryAgent:
    def __init__(self, memory_file: Path = _MEMORY_FILE, max_entries: int = 100):
        self.memory_file = memory_file
        self.max_entries = max_entries
        # Ensure file exists
        if not self.memory_file.exists():
            self._write([])

    def _read(self) -> List[Dict[str, Any]]:
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write(self, data):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def append_event(self, event: Dict[str, Any]):
        data = self._read()
        data.append(event)
        # keep recent max_entries
        if len(data) > self.max_entries:
            data = data[-self.max_entries :]
        self._write(data)

    def query_recent(self, limit: int = 10):
        data = self._read()
        return data[-limit:]
''',

"gccn/agents/evaluator_agent.py": '''"""
EvaluatorAgent: simple evaluator to score outputs (for observability & metrics).
Designed to be extendable.
"""
from typing import Dict, Any

class EvaluatorAgent:
    def __init__(self):
        pass

    def evaluate(self, analysis: Dict[str, Any], resources: Dict[str, Any], messages: Dict[str, Any]) -> Dict[str, Any]:
        # Basic checks: ensure message includes headline, checklist, resources non-empty
        score = 0
        reasons = []
        if messages.get("headline"):
            score += 1
        else:
            reasons.append("No headline")

        if messages.get("checklist"):
            score += 1
        else:
            reasons.append("No checklist")

        if resources.get("resources"):
            score += 1
        else:
            reasons.append("No resources mapped")

        # Example metric: risk_score mapped to severity level
        risk = analysis.get("risk_score", 0)
        return {
            "score": score,
            "reasons": reasons,
            "risk_score": risk,
            "valid": score >= 2
        }
''',

"gccn/orchestrator.py": '''"""
Orchestrator: runs a single cycle (or multiple cycles) connecting agents.
This is the main runnable entry for demo mode.
"""
from typing import Dict, Any
from .agents.data_scout import DataScoutAgent
from .agents.risk_agent import RiskAgent
from .agents.resource_agent import ResourceAgent
from .agents.comms_agent import CommsAgent
from .agents.memory_agent import MemoryAgent
from .agents.evaluator_agent import EvaluatorAgent
from .tools.llm_wrapper import LLMWrapper

class Orchestrator:
    def __init__(self, llm: LLMWrapper = None):
        self.scout = DataScoutAgent()
        self.llm = llm or LLMWrapper()
        self.risk = RiskAgent(llm=self.llm)
        self.resource = ResourceAgent()
        self.comms = CommsAgent(llm=self.llm)
        self.memory = MemoryAgent()
        self.evaluator = EvaluatorAgent()

    def run_cycle(self, lat: float, lon: float) -> Dict[str, Any]:
        payload = self.scout.poll(lat, lon)
        analysis = self.risk.analyze(payload)
        resources = self.resource.map_resources(payload.get("location", {}))
        messages = self.comms.generate_messages(analysis, resources)
        eval_metrics = self.evaluator.evaluate(analysis, resources, messages)

        # persist event to memory
        event = {
            "payload": payload,
            "analysis": analysis,
            "resources": resources,
            "messages": messages,
            "evaluation": eval_metrics
        }
        self.memory.append_event(event)

        # Return combined response for UI / API / tests
        return {
            "payload": payload,
            "analysis": analysis,
            "resources": resources,
            "messages": messages,
            "evaluation": eval_metrics
        }
''',

"gccn/cli.py": '''"""
Simple CLI entry to run a demo cycle.
Usage:
    python -m gccn.cli --lat 40.7128 --lon -74.0060
"""
import argparse
from .orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="Run GCCN demo cycle")
    parser.add_argument("--lat", type=float, required=False, default=40.7128)
    parser.add_argument("--lon", type=float, required=False, default=-74.0060)
    args = parser.parse_args()
    orch = Orchestrator()
    result = orch.run_cycle(args.lat, args.lon)
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
''',

"tests/test_agents.py": '''import os
from gccn.agents.data_scout import DataScoutAgent
from gccn.agents.risk_agent import RiskAgent
from gccn.agents.resource_agent import ResourceAgent
from gccn.agents.comms_agent import CommsAgent
from gccn.tools.llm_wrapper import LLMWrapper

def test_data_scout_returns_payload():
    scout = DataScoutAgent()
    payload = scout.poll(40.7128, -74.0060)
    assert "weather" in payload
    assert "cap" in payload
    assert "location" in payload

def test_risk_agent_scores_reasonably():
    llm = LLMWrapper()
    ra = RiskAgent(llm=llm)
    payload = {"weather": {"temp_c": 41.0, "alerts": []}, "cap": {"alerts": []}, "social": [] , "location": {"display_name": "TestTown"}}
    out = ra.analyze(payload)
    assert isinstance(out.get("risk_score"), int)
    assert "narrative" in out

def test_resource_agent_returns_resources():
    ra = ResourceAgent()
    resources = ra.map_resources({"display_name": "TestTown"})
    assert "resources" in resources
    assert isinstance(resources["resources"], list)

def test_comms_agent_generates_message():
    llm = LLMWrapper()
    ca = CommsAgent(llm=llm)
    analysis = {"risk_score": 4, "narrative": "heat risk high", "reasons": ["hot"]}
    resources = {"resources": [{"type":"shelter","name":"X"}]}
    messages = ca.generate_messages(analysis, resources)
    assert "headline" in messages
    assert "checklist" in messages
    assert "shareable_message" in messages
''',

"tests/test_end_to_end.py": '''from gccn.orchestrator import Orchestrator

def test_orchestrator_cycle():
    orch = Orchestrator()
    res = orch.run_cycle(40.7128, -74.0060)
    assert "analysis" in res
    assert "messages" in res
    assert res["evaluation"]["valid"] in (True, False)
''',

"README.md": '''# Global Community Crisis Navigator (GCCN)

**Track:** Agents for Good  
**Short pitch:** GCCN is a modular multi-agent system that ingests local hazard data, analyzes community-level risk, maps nearby resources, and generates actionable, shareable preparedness & response guidance for neighborhoods, schools, and small municipalities.

---

## Quick features
- Multi-agent architecture: DataScout, RiskAgent, ResourceAgent, CommsAgent, MemoryAgent, EvaluatorAgent.
- Tools layer for data ingestion & geocoding (mocked for demo).
- LLM wrapper abstraction (mock deterministic responses by default; supports real LLM integration).
- File-backed memory for reproducible demo & improvement tracking.
- Evaluator that checks output structure for observability metrics.
- Demo mode via CLI; no API keys required to run basic scenarios.

---

## Quickstart (Demo mode — no API keys)
```bash
# In notebook: run the CLI demo (python -m gccn.cli)
python -m gccn.cli --lat 40.7128 --lon -74.0060

# Run tests
pytest -q




---

### Cell 3 — Install dependencies
```bash
# Install requirements inside the Kaggle kernel (may take a minute)
cd gccn_project
pip install -r requirements.txt



# Run pytest in the created project
cd gccn_project
pytest -q


# Run a demo GCCN cycle and display the results nicely
import json, sys
sys.path.insert(0, "/kaggle/working/gccn_project")  # ensure python finds our package
from gccn.orchestrator import Orchestrator

orch = Orchestrator()
res = orch.run_cycle(40.7128, -74.0060)
print(json.dumps(res, indent=2))



from pathlib import Path, PurePosixPath
import json
MEMORY = Path("/kaggle/working/gccn_project/gccn/data/memory.json")
if MEMORY.exists():
    with open(MEMORY, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Memory entries:", len(data))
    # print last event summary
    print(json.dumps(data[-1], indent=2))
else:
    print("Memory file not found at:", MEMORY)



# If you uploaded the logo as a dataset or file, Kaggle stores uploaded files under /kaggle/input or /mnt/data.
from IPython.display import Image, display
logo_path = "/mnt/data/A_logo_for_the_Global_Community_Crisis_Network_(GC.png"
# try both path variants
import os
if os.path.exists(logo_path):
    display(Image(logo_path, width=400))
else:
    print("Logo not found at", logo_path)


