"""
Food Waste Tracker — Multi-Agent Proof-of-Concept
Single-file runnable prototype suitable for a Kaggle/Capstone submission demo.

Features demonstrated (>=3 course concepts):
- Multi-agent system (CaptureAgent, DetectionAgent, ForecastAgent, PlannerAgent)
- Agent powered by an LLM (PlannerAgent uses OpenAI or placeholder)
- Parallel agents (agents run concurrently using multiprocessing)
- Loop agents (continuous worker loops)
- Tools & custom tool emulation (simple 'MCP' message passing via multiprocessing.Queue)
- Sessions & in-memory state (InMemorySessionService)
- Long-running operations with pause/resume control
- Observability: structured logging + Prometheus metrics endpoint
- Agent-to-Agent (A2A) communication via queues
- Basic agent evaluation: detection accuracy logging

Run requirements:
- Python 3.8+
- pip install fastapi uvicorn[standard] pydantic sqlalchemy prometheus-client requests
- Optional: openai (if you want PlannerAgent to call real LLMs)

Run locally:
    uvicorn food_waste_agent_system:app --reload

Endpoints:
- POST /capture  -> upload image (multipart) + weight + meal_id
- GET /metrics   -> Prometheus metrics
- GET /status    -> system & agent status
- POST /control  -> pause/resume agents

This is a prototype: detection is stubbed (random), forecasting is simple, planner uses a local heuristic or OpenAI if API key is provided.

"""

import os
import io
import time
import uuid
import random
import logging
import threading
from multiprocessing import Process, Queue, Event, Manager
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST

# Optional OpenAI import (PlannerAgent can use real LLM if OPENAI_API_KEY is set)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------------- Logging & Observability ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("food-waste-agent")

# Prometheus metrics
METRIC_WASTE_KG = Gauge("food_waste_kg_total", "Total food waste recorded (kg)")
METRIC_RECORDS = Counter("food_waste_records_total", "Number of waste records processed")
METRIC_DET_ACCURACY = Gauge("detection_accuracy", "Detection agent accuracy (simulated)")

# ---------------------- Simple Storage (SQLite via SQLAlchemy could be added). We'll use in-memory list for demo ----------------------
class InMemoryDB:
    def __init__(self):
        self.records = []  # list of dicts
        self.lock = threading.Lock()

    def insert_record(self, rec: Dict[str, Any]):
        with self.lock:
            self.records.append(rec)

    def get_recent(self, minutes=60):
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        with self.lock:
            return [r for r in self.records if r["timestamp"] >= cutoff]

    def all(self):
        with self.lock:
            return list(self.records)

DB = InMemoryDB()

# ---------------------- Session Service (in-memory) ----------------------
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()

    def create_session(self, session_id=None):
        sid = session_id or uuid.uuid4().hex
        with self.lock:
            self.sessions[sid] = {"created": datetime.utcnow(), "events": []}
        return sid

    def append(self, sid, event):
        with self.lock:
            if sid not in self.sessions:
                self.create_session(sid)
            self.sessions[sid]["events"].append((datetime.utcnow(), event))

    def get(self, sid):
        with self.lock:
            return self.sessions.get(sid)

SESSION_SERVICE = InMemorySessionService()

# ---------------------- Message types & A2A (MCP) ----------------------
class CaptureMessage(BaseModel):
    record_id: str
    meal_id: int
    device_id: str
    weight_grams: float
    image_bytes: bytes
    timestamp: datetime

class DetectionResult(BaseModel):
    record_id: str
    leftover_pct: float
    waste_grams: float
    confidence: float
    timestamp: datetime

# ---------------------- Dummy detection model (replace with real model) ----------------------
class DummyDetector:
    def predict_leftover_pct(self, image_bytes: bytes) -> float:
        # Very simple heuristic: random but biased
        return float(max(0.0, min(1.0, random.gauss(0.25, 0.18))))

DETECTOR = DummyDetector()

# ---------------------- Agents ----------------------

class CaptureAgent(Process):
    """Not a long-running process here; capture handled by FastAPI endpoint which pushes to MCP queue."""
    pass

class DetectionAgent(Process):
    def __init__(self, in_queue: Queue, out_queue: Queue, control_event: Event, eval_store: Dict):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.control_event = control_event
        self.eval_store = eval_store

    def run(self):
        logger.info("DetectionAgent started")
        processed = 0
        correct_sim = 0
        while True:
            if self.control_event.is_set():
                logger.info("DetectionAgent paused")
                time.sleep(0.5)
                continue
            try:
                msg: CaptureMessage = self.in_queue.get(timeout=1)
            except Exception:
                time.sleep(0.1)
                continue

            # Run detection (dummy)
            pct = DETECTOR.predict_leftover_pct(msg.image_bytes)
            waste_grams = pct * msg.weight_grams
            result = DetectionResult(
                record_id=msg.record_id,
                leftover_pct=pct * 100.0,
                waste_grams=waste_grams,
                confidence=0.7,
                timestamp=datetime.utcnow(),
            )

            # push to out queue for analytics / storage
            self.out_queue.put(result.json())

            # simple evaluation simulation: assume label exists sometimes
            processed += 1
            if random.random() < 0.4:
                # simulate a ground truth label
                gt_pct = max(0.0, min(1.0, pct + random.gauss(0, 0.15)))
                error = abs(gt_pct - pct)
                if error < 0.2:
                    correct_sim += 1
                # update eval_store atomically
                self.eval_store["processed"] = processed
                self.eval_store["correct"] = correct_sim
                acc = correct_sim / processed
                METRIC_DET_ACCURACY.set(acc)

            METRIC_RECORDS.inc()
            METRIC_WASTE_KG.inc(waste_grams / 1000.0)
            logger.info(f"Detected {result.waste_grams:.1f}g waste for record {result.record_id}")

class StorageAgent(Process):
    def __init__(self, in_queue: Queue, control_event: Event):
        super().__init__()
        self.in_queue = in_queue
        self.control_event = control_event

    def run(self):
        logger.info("StorageAgent started")
        while True:
            if self.control_event.is_set():
                logger.info("StorageAgent paused")
                time.sleep(0.5)
                continue
            try:
                payload = self.in_queue.get(timeout=1)
            except Exception:
                time.sleep(0.1)
                continue
            # payload is JSON string of DetectionResult
            try:
                obj = DetectionResult.parse_raw(payload)
            except Exception as e:
                logger.exception("Bad payload")
                continue
            # store into in-memory DB
            rec = {
                "id": obj.record_id,
                "timestamp": obj.timestamp,
                "leftover_pct": obj.leftover_pct,
                "waste_grams": obj.waste_grams,
            }
            DB.insert_record(rec)
            logger.info(f"Stored record {rec['id']} waste={rec['waste_grams']:.1f}g")

class ForecastAgent(Process):
    """Simple forecasting agent: runs periodically and writes forecast into session store."""
    def __init__(self, interval_seconds: int, control_event: Event, session_service: InMemorySessionService):
        super().__init__()
        self.interval = interval_seconds
        self.control_event = control_event
        self.session_service = session_service

    def run(self):
        logger.info("ForecastAgent started")
        while True:
            if self.control_event.is_set():
                logger.info("ForecastAgent paused")
                time.sleep(0.5)
                continue
            # compute trivial forecast from recent average served/waste
            recent = DB.get_recent(minutes=60*24)
            if recent:
                avg_waste = sum(r["waste_grams"] for r in recent) / len(recent)
            else:
                avg_waste = 50.0
            forecast = {"next_3_days_estimated_waste_grams": [avg_waste * (1 + random.uniform(-0.2, 0.2)) for _ in range(3)]}
            sid = self.session_service.create_session()
            self.session_service.append(sid, {"forecast_generated": forecast, "at": datetime.utcnow().isoformat()})
            logger.info(f"ForecastAgent generated forecast (sid={sid})")
            time.sleep(self.interval)

class PlannerAgent(Process):
    """Agent powered by an LLM (or local heuristic if OpenAI not configured). It consumes latest stats and suggests action items."""
    def __init__(self, control_event: Event, session_service: InMemorySessionService):
        super().__init__()
        self.control_event = control_event
        self.session_service = session_service

    def _query_llm(self, prompt: str) -> str:
        # If OpenAI is available and API key is set, use it; otherwise use a local heuristic generator
        api_key = os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and api_key:
            openai.api_key = api_key
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.exception("OpenAI call failed, falling back")
        # fallback heuristic
        return "Suggest reducing portions for items with consistently >30% waste; offer discounts near end-of-service; collect staff feedback."

    def run(self):
        logger.info("PlannerAgent started")
        while True:
            if self.control_event.is_set():
                logger.info("PlannerAgent paused")
                time.sleep(0.5)
                continue
            # summarize recent waste
            recent = DB.get_recent(minutes=60*24)
            total_kg = sum(r["waste_grams"] for r in recent) / 1000.0
            top = sorted(recent, key=lambda r: r["waste_grams"], reverse=True)[:3]
            summary = {
                "total_kg_last_24h": total_kg,
                "top_waste_samples": top,
            }
            prompt = f"We recorded {total_kg:.2f} kg waste in the last 24h. Top samples: {top}. Provide 3 actionable suggestions for kitchen staff to reduce waste and a brief explanation for each."
            plan = self._query_llm(prompt)
            sid = self.session_service.create_session()
            self.session_service.append(sid, {"plan": plan, "summary": summary, "at": datetime.utcnow().isoformat()})
            logger.info(f"PlannerAgent suggested actions (sid={sid})")
            time.sleep(30)

# ---------------------- Orchestrator to launch agents ----------------------
class Orchestrator:
    def __init__(self):
        manager = Manager()
        self.capture_to_detect_q = manager.Queue()
        self.detect_to_store_q = manager.Queue()

        self.det_control = Event()
        self.store_control = Event()
        self.forecast_control = Event()
        self.planner_control = Event()

        self.eval_store = manager.dict()

        self.detection_agent = DetectionAgent(self.capture_to_detect_q, self.detect_to_store_q, self.det_control, self.eval_store)
        self.storage_agent = StorageAgent(self.detect_to_store_q, self.store_control)
        self.forecast_agent = ForecastAgent(interval_seconds=60, control_event=self.forecast_control, session_service=SESSION_SERVICE)
        self.planner_agent = PlannerAgent(control_event=self.planner_control, session_service=SESSION_SERVICE)

    def start(self):
        logger.info("Starting orchestrator and agents...")
        self.detection_agent.start()
        self.storage_agent.start()
        self.forecast_agent.start()
        self.planner_agent.start()

    def stop(self):
        logger.info("Stopping agents (terminate)")
        for p in [self.detection_agent, self.storage_agent, self.forecast_agent, self.planner_agent]:
            try:
                p.terminate()
            except Exception:
                pass

    def pause_agent(self, agent_name: str):
        if agent_name == "detection":
            self.det_control.set()
        elif agent_name == "storage":
            self.store_control.set()
        elif agent_name == "forecast":
            self.forecast_control.set()
        elif agent_name == "planner":
            self.planner_control.set()

    def resume_agent(self, agent_name: str):
        if agent_name == "detection":
            self.det_control.clear()
        elif agent_name == "storage":
            self.store_control.clear()
        elif agent_name == "forecast":
            self.forecast_control.clear()
        elif agent_name == "planner":
            self.planner_control.clear()


ORCH = Orchestrator()
ORCH.start()

# ---------------------- FastAPI app (Capture endpoint + status) ----------------------
app = FastAPI(title="Food Waste Tracker — Agent POC")

class ControlRequest(BaseModel):
    action: str  # pause | resume
    agent: str

@app.post("/capture")
async def capture_endpoint(meal_id: int = Form(...), device_id: str = Form("unknown"), weight_grams: float = Form(...), image: UploadFile = File(...)):
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image")
    rec_id = uuid.uuid4().hex
    msg = CaptureMessage(
        record_id=rec_id,
        meal_id=meal_id,
        device_id=device_id,
        weight_grams=weight_grams,
        image_bytes=content,
        timestamp=datetime.utcnow(),
    )
    # push into MCP queue for detection
    ORCH.capture_to_detect_q.put(msg)
    SESSION_SERVICE.append(rec_id, {"captured": {"meal_id": meal_id, "weight_grams": weight_grams}})
    logger.info(f"Capture received: {rec_id} meal={meal_id} weight={weight_grams}g")
    return JSONResponse({"record_id": rec_id, "status": "queued"})

@app.get("/status")
async def status():
    return {
        "agent_pids": {
            "detection": ORCH.detection_agent.pid,
            "storage": ORCH.storage_agent.pid,
            "forecast": ORCH.forecast_agent.pid,
            "planner": ORCH.planner_agent.pid,
        },
        "db_records": len(DB.all()),
        "eval_store": dict(ORCH.eval_store),
    }

@app.post("/control")
async def control(req: ControlRequest):
    action = req.action.lower()
    agent = req.agent.lower()
    if action == "pause":
        ORCH.pause_agent(agent)
        return {"status": "paused", "agent": agent}
    elif action == "resume":
        ORCH.resume_agent(agent)
        return {"status": "resumed", "agent": agent}
    else:
        raise HTTPException(status_code=400, detail="action must be 'pause' or 'resume'")

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)

@app.get("/records")
async def list_records():
    recs = DB.all()
    # convert datetimes to iso
    for r in recs:
        if isinstance(r["timestamp"], datetime):
            r["timestamp"] = r["timestamp"].isoformat()
    return {"count": len(recs), "records": recs}

@app.get("/session/{sid}")
async def get_session(sid: str):
    s = SESSION_SERVICE.get(sid)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return s

# ---------------------- Graceful shutdown ----------------------
import atexit

def _cleanup():
    try:
        ORCH.stop()
    except Exception:
        pass

atexit.register(_cleanup)

# ---------------------- Simple agent evaluation endpoint ----------------------
@app.get("/eval")
async def eval_stats():
    st = dict(ORCH.eval_store)
    processed = st.get("processed", 0)
    correct = st.get("correct", 0)
    acc = (correct / processed) if processed else None
    return {"processed": processed, "correct": correct, "accuracy": acc}

# ---------------------- Example: health-check and README text ----------------------
@app.get("/")
async def home():
    return {"service": "food-waste-agent-poc", "version": "0.1"}

# EOF


