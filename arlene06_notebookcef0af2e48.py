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


# Cell 1 — Install dependencies (run once)
!pip install openai aiohttp nest_asyncio python-dotenv

# (Optional for evidence search using SerpAPI)
# !pip install google-search-results



# Cell 2 — Imports and config
import os, time, json, logging, pickle, uuid, asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

# Load env (if you put keys in a .env file)
from dotenv import load_dotenv
load_dotenv()

# LLM selection: we default to OpenAI but code tolerates missing key (offline dev)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # change if needed

import openai

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("llm_agents")

# Paths
OUTPUT_DIR = "/kaggle/working/agents_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MEMORY_PATH = os.path.join(OUTPUT_DIR, "memory_bank.pkl")
TRACE_PATH = os.path.join(OUTPUT_DIR, "agent_traces.jsonl")



# Cell 3 — Utility functions and simple tools
def generate_id(prefix="s"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def safe_write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def append_trace(record):
    with open(TRACE_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

def calc_bmi(weight_kg:float, height_cm:float):
    if height_cm <= 0:
        return None
    h_m = height_cm/100.0
    return round(weight_kg / (h_m*h_m), 2)



# Cell 4 — Memory and Session services
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}
    def create_session(self, session_id=None, meta=None):
        sid = session_id or generate_id("sess")
        self.sessions[sid] = {"meta": meta or {}, "history": []}
        return sid
    def append(self, session_id, entry):
        if session_id not in self.sessions:
            raise KeyError("session not found")
        self.sessions[session_id]["history"].append({"ts": time.time(), "entry": entry})
    def get(self, session_id):
        return self.sessions.get(session_id)

class MemoryBank:
    def __init__(self, path=MEMORY_PATH):
        self.path = path
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.store = pickle.load(f)
            except Exception:
                self.store = {}
        else:
            self.store = {}
    def set(self, key, value):
        self.store[key] = value
        with open(self.path, "wb") as f:
            pickle.dump(self.store, f)
    def get(self, key):
        return self.store.get(key)
    def all(self):
        return self.store

# instantiate
session_service = InMemorySessionService()
memory_bank = MemoryBank()



# Cell 5 — Base Agent + simple agent implementations
class BaseAgent:
    def __init__(self, name):
        self.name = name
        self.metrics = {"runs":0, "avg_time_s":0.0}
    def _record(self, t_s):
        self.metrics["runs"] += 1
        self.metrics["avg_time_s"] += (t_s - self.metrics["avg_time_s"]) / self.metrics["runs"]
    def run(self, *args, **kwargs):
        raise NotImplementedError

# InputAgent: validates & normalizes input
class InputAgent(BaseAgent):
    def __init__(self):
        super().__init__("InputAgent")
    def run(self, raw_input:Dict[str,Any]):
        start=time.time()
        # normalize keys, defaults
        patient = {}
        # expected keys: age, weight_kg, height_cm, cycle_length_days, last_period_date, symptoms(list)/dict, family_history (dict), medications (list)
        patient['age'] = int(raw_input.get('age', 0))
        patient['weight_kg'] = float(raw_input.get('weight_kg', 0.0))
        patient['height_cm'] = float(raw_input.get('height_cm', 0.0))
        patient['bmi'] = calc_bmi(patient['weight_kg'], patient['height_cm'])
        patient['cycle_length_days'] = raw_input.get('cycle_length_days', None)
        patient['symptoms'] = raw_input.get('symptoms', {})
        patient['family_history'] = raw_input.get('family_history', {})
        patient['notes'] = raw_input.get('notes', "")
        t = time.time()-start
        self._record(t)
        return patient

# MedicalAnalysisAgent: uses LLM to produce per-condition risk assessment
class MedicalAnalysisAgent(BaseAgent):
    def __init__(self, model=OPENAI_MODEL):
        super().__init__("MedicalAnalysisAgent")
        self.model = model
        # conditions to evaluate
        self.conditions = ["PCOS", "Thyroid disorder", "Infertility risk", "Menopause risk"]
    def _make_prompt(self, condition, patient):
        prompt = f"""
You are a clinical-assistant (non-diagnostic). Evaluate the likelihood (Low/Moderate/High) and give a probability estimate (0-1) that the patient has or is at risk for {condition}.
Patient data (JSON):
{json.dumps(patient, indent=2)}

Provide:
1) short label (Low/Moderate/High)
2) probability (0-1, two decimals)
3) 3 concise reasons (medical features from the patient supporting the assessment)
4) 2 suggested next steps for the patient (non-prescriptive, e.g., "consider seeing an endocrinologist")
Return only a JSON object with keys: label, probability, reasons (list), next_steps (list).
Important: include a "source_confidence" field (Low/Medium/High) representing how confident you are given no lab tests.
Do NOT give definitive diagnostic language; always include a safety disclaimer.
"""
        return prompt
    def _call_llm(self, prompt, max_tokens=400):
        start=time.time()
        if not OPENAI_API_KEY:
            # offline fallback: return heuristic stub
            time.sleep(0.4)
            self._record(time.time()-start)
            return {"model":"stub","text":"{\"label\":\"Moderate\",\"probability\":0.45,\"reasons\":[\"Age\",\"BMI\"],\"next_steps\":[\"Check hormones\",\"See doctor\"],\"source_confidence\":\"Low\"}"}
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        text = resp["choices"][0]["message"]["content"]
        self._record(time.time()-start)
        return {"model":self.model,"text":text}
    def run_single(self, condition, patient):
        prompt = self._make_prompt(condition, patient)
        out = self._call_llm(prompt)
        text = out["text"]
        # try parse JSON from response
        try:
            parsed = json.loads(text)
        except Exception:
            # fallback: wrap full text into explanation
            parsed = {"label":"Unknown","probability":0.0,"reasons":[text[:200]],"next_steps":[],"source_confidence":"Low"}
        return parsed
    def run(self, patient, parallel=False):
        results = {}
        if parallel:
            with ThreadPoolExecutor(max_workers=len(self.conditions)) as ex:
                futures = {ex.submit(self.run_single, c, patient): c for c in self.conditions}
                for fut in as_completed(futures):
                    c = futures[fut]
                    try:
                        results[c] = fut.result()
                    except Exception as e:
                        results[c] = {"error":str(e)}
        else:
            for c in self.conditions:
                results[c] = self.run_single(c, patient)
        return results

# EvidenceAgent (optional): calls a search API or returns empty list if disabled
class EvidenceAgent(BaseAgent):
    def __init__(self, serpapi_key=None):
        super().__init__("EvidenceAgent")
        self.serpapi_key = serpapi_key
    def run(self, query, top_k=3):
        start=time.time()
        # if no key, return empty evidence list
        if not self.serpapi_key:
            self._record(time.time()-start)
            return []
        # example using google-search-results package (user must pip install and set key)
        try:
            from serpapi import GoogleSearch
            search = GoogleSearch({"q": query, "api_key": self.serpapi_key})
            res = search.get_dict()
            items = res.get("organic_results", [])[:top_k]
            evidence = [{"title":it.get("title"), "link":it.get("link"), "snippet":it.get("snippet")} for it in items]
        except Exception as e:
            evidence = []
        self._record(time.time()-start)
        return evidence

# ReportAgent: format patient-friendly report using LLM
class ReportAgent(BaseAgent):
    def __init__(self, model=OPENAI_MODEL):
        super().__init__("ReportAgent")
        self.model = model
    def _make_prompt(self, patient, assessments, evidences=None):
        prompt = f"""
You are to produce a concise patient-facing report (max 250 words) that summarizes risk assessments for the following patient.
Patient: {json.dumps(patient, indent=2)}
Assessments per condition: {json.dumps(assessments, indent=2)}
Evidence citations (if any): {json.dumps(evidences or [], indent=2)}

Produce:
- Short title
- 1-2 sentence summary overall
- For each condition: 1-line summary (label + probability + 1-line recommendation)
- 2 general next steps
- Short safety disclaimer

Return only JSON with keys: title, summary, per_condition (dict), next_steps (list), disclaimer.
"""
        return prompt
    def _call_llm(self, prompt):
        start=time.time()
        if not OPENAI_API_KEY:
            time.sleep(0.4)
            self._record(time.time()-start)
            return {"text": json.dumps({"title":"Demo Report","summary":"Demo summary","per_condition":{},"next_steps":[],"disclaimer":"Demo"})}
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0.0
        )
        text = resp["choices"][0]["message"]["content"]
        self._record(time.time()-start)
        return {"text": text}
    def run(self, patient, assessments, evidences=None):
        prompt = self._make_prompt(patient, assessments, evidences)
        out = self._call_llm(prompt)
        text = out["text"]
        try:
            parsed = json.loads(text)
        except Exception:
            # fallback wrap
            parsed = {"title":"Report", "summary": text[:250], "per_condition":assessments, "next_steps":[], "disclaimer":"Please consult a professional."}
        return parsed



# Cell 6 — Orchestration: Session + pipeline runner
# Instantiate agents (set SERPAPI_KEY if you want evidence search)
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
input_agent = InputAgent()
analysis_agent = MedicalAnalysisAgent()
evidence_agent = EvidenceAgent(serpapi_key=SERPAPI_KEY)
report_agent = ReportAgent()

def run_pipeline(raw_input, session_id=None, fetch_evidence=False, parallel=True):
    session_id = session_id or session_service.create_session()
    session_service.append(session_id, {"event":"received_input","data":raw_input})
    # 1. Input agent
    patient = input_agent.run(raw_input)
    session_service.append(session_id, {"event":"normalized_input","data":patient})
    # 2. Analysis (parallel across conditions)
    assessments = analysis_agent.run(patient, parallel=parallel)
    session_service.append(session_id, {"event":"assessments","data":assessments})
    # 3. Evidence fetch (optional)
    evidences = {}
    if fetch_evidence and SERPAPI_KEY:
        for cond, a in assessments.items():
            q = f"{cond} guideline risk factors"
            evidences[cond] = evidence_agent.run(q, top_k=2)
        session_service.append(session_id, {"event":"evidence","data":evidences})
    else:
        evidences = None
    # 4. Report
    report = report_agent.run(patient, assessments, evidences)
    session_service.append(session_id, {"event":"report","data":report})
    # Save to memory (persistent) if the user consents (here we auto-save a short summary)
    mem_key = f"patient_{generate_id()}"
    short_summary = {"age":patient.get("age"), "bmi":patient.get("bmi"), "assessments": {k:{"label":v.get("label"), "probability":v.get("probability")} for k,v in assessments.items()}}
    memory_bank.set(mem_key, short_summary)
    session_service.append(session_id, {"event":"memory_saved","key":mem_key})
    # Observability trace
    trace = {
        "session": session_id,
        "patient_id": mem_key,
        "timestamp": time.time(),
        "agents": {
            "input_agent": input_agent.metrics,
            "analysis_agent": analysis_agent.metrics,
            "evidence_agent": evidence_agent.metrics,
            "report_agent": report_agent.metrics
        }
    }
    append_trace(trace)
    return {"session_id": session_id, "patient_summary_key": mem_key, "report": report, "assessments": assessments}



# Cell 7 — Demo: run a synthetic example (edit values to test)
demo_input = {
    "age": 26,
    "weight_kg": 78,
    "height_cm": 160,
    "cycle_length_days": 40,
    "symptoms": {"irregular_periods": True, "excess_hair": True, "acne": True, "fatigue": False},
    "family_history": {"diabetes": True, "thyroid": False},
    "notes": "Patient reports irregular cycles since late teens; overweight."
}

result = run_pipeline(demo_input, fetch_evidence=False, parallel=True)
print("Session:", result["session_id"])
print("Report (raw):")
print(json.dumps(result["report"], indent=2))
print("\nAssessments (summary):")
for cond, v in result["assessments"].items():
    print(f"- {cond}: {v.get('label')} ({v.get('probability')})")



# Cell 8 — inspect traces and memory
print("MemoryBank keys:", list(memory_bank.all().keys())[:10])
print("Last trace lines (most recent 5):")
with open(TRACE_PATH, "r") as f:
    lines = f.readlines()[-5:]
    for L in lines:
        print(json.loads(L))



# Not runnable in Kaggle kernel persistently, but useful for local demo.
# pip install fastapi uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictPayload(BaseModel):
    raw_input: dict
    fetch_evidence: bool = False

@app.post("/predict")
async def predict(payload: PredictPayload):
    return run_pipeline(payload.raw_input, fetch_evidence=payload.fetch_evidence)

# Run locally: uvicorn.run(app, host="0.0.0.0", port=8000)




---

# 3) Example demo run (what to expect)
If you run the demo cell (Cell 7) with the synthetic input, the pipeline will:

- Normalize input and compute BMI (~30.5).
- The MedicalAnalysisAgent will call the LLM for each condition and return JSON like:

Example (simulated):
```json
{
  "PCOS": {
    "label": "High",
    "probability": 0.78,
    "reasons": ["Irregular cycles (40 days)", "BMI 30.5 (overweight)", "Hirsutism / acne"],
    "next_steps": ["Consider endocrine consult and hormonal tests (LH/FSH/testosterone)", "Lifestyle interventions for weight management"],
    "source_confidence": "Medium"
  },
  "Thyroid disorder": {
    "label": "Moderate",
    "probability": 0.35,
    "reasons": ["Fatigue and family history of diabetes (metabolic risk) — not specific", "No TSH value provided"],
    "next_steps": ["Check TSH/T4 labs", "Discuss symptoms with PCP"],
    "source_confidence": "Low"
  },
  ...
}


