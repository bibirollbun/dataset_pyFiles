import asyncio
import logging
import time
import random
import json
from typing import Any, Dict, Optional, List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("aqi_guardian")

Base = declarative_base()

class MemoryRecord(Base):
    __tablename__ = 'memory_records'
    id = Column(Integer, primary_key=True)
    user_id = Column(String(128), index=True)
    key = Column(String(128), index=True)
    value = Column(Text)

gine = create_engine('sqlite:///aqi_guardian_memory.db', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

class AQIFetcherTool:
    async def fetch(self, location: str) -> Dict[str, Any]:
        await asyncio.sleep(random.uniform(0.1, 0.4))
        aqi_val = random.randint(40, 320) if 'delhi' not in location.lower() else random.randint(150, 350)
        return {
            'location': location,
            'aqi': aqi_val,
            'pm25': round(random.uniform(10, 250), 1),
            'pm10': round(random.uniform(20, 300), 1),
            'o3': round(random.uniform(5, 120), 1),
            'no2': round(random.uniform(5, 150), 1),
            'source': 'simulated',
            'ts': time.time()
        }

class WeatherTool:
    async def fetch(self, location: str) -> Dict[str, Any]:
        await asyncio.sleep(random.uniform(0.05, 0.2))
        return {
            'location': location,
            'temp_c': round(random.uniform(5, 35), 1),
            'wind_kph': round(random.uniform(0, 20), 1),
            'humidity': random.randint(20, 95),
            'ts': time.time()
        }

class GoogleSearchTool:
    async def query(self, q: str) -> List[str]:
        await asyncio.sleep(0.2)
        return [f"Result for {q} - {i}" for i in range(1,4)]

class LLMClient:
    def __init__(self):
        pass
    async def generate(self, prompt: str, max_tokens: int = 200) -> str:
        lines = prompt.split("\n")
        core = ' '.join(lines[-4:])
        return f"Summary: {core[:200]}"

class InMemorySessionService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    def get(self, session_id: str) -> Dict[str, Any]:
        with self.lock:
            return self.sessions.setdefault(session_id, {})
    def set(self, session_id: str, key: str, value: Any):
        with self.lock:
            s = self.sessions.setdefault(session_id, {})
            s[key] = value

class MemoryBank:
    def __init__(self):
        self.db = SessionLocal
    def save(self, user_id: str, key: str, value: Any):
        db = self.db()
        rec = MemoryRecord(user_id=user_id, key=key, value=json.dumps(value))
        db.add(rec)
        db.commit()
        db.close()
    def query_recent(self, user_id: str, limit: int = 10) -> List[Dict[str,Any]]:
        db = self.db()
        rows = db.query(MemoryRecord).filter(MemoryRecord.user_id==user_id).order_by(MemoryRecord.id.desc()).limit(limit).all()
        out = []
        for r in rows:
            out.append({'key': r.key, 'value': json.loads(r.value)})
        db.close()
        return out

class Trace:
    def __init__(self):
        self.steps = []
    def add(self, agent: str, msg: str):
        self.steps.append({'agent': agent, 'msg': msg, 'ts': time.time()})
    def to_dict(self):
        return {'steps': self.steps}

class DataCollectorAgent:
    def __init__(self, aqi_tool: AQIFetcherTool, weather_tool: WeatherTool):
        self.aqi_tool = aqi_tool
        self.weather_tool = weather_tool
    async def collect(self, location: str, trace: Trace) -> Dict[str, Any]:
        trace.add("collector", "start")
        t1 = asyncio.create_task(self.aqi_tool.fetch(location))
        t2 = asyncio.create_task(self.weather_tool.fetch(location))
        aqi_res, weather_res = await asyncio.gather(t1, t2)
        trace.add("collector", "done")
        return {'aqi': aqi_res, 'weather': weather_res}

class AnalyticsAgent:
    async def analyze(self, collected: Dict[str, Any], trace: Trace) -> Dict[str, Any]:
        trace.add("analytics", "start")
        await asyncio.sleep(0.2)
        aqi = collected['aqi']['aqi']
        pm25 = collected['aqi']['pm25']
        wind = collected['weather']['wind_kph']
        trend = -1 if wind > 8 else (1 if pm25 > 100 else 0)
        forecast = []
        base = aqi
        for h in range(1,7):
            base = int(base + trend * random.randint(0, 15))
            forecast.append({'hour': h, 'pred_aqi': max(10, base)})
        trace.add("analytics", "done")
        return {
            'current_aqi': aqi,
            'pm25': pm25,
            'wind_kph': wind,
            'forecast_hours': forecast,
            'explain': f"Trend depends on wind {wind} and pm25 {pm25}"
        }

class AdvisorAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    async def advise(self, analysis: Dict[str, Any], prefs: Dict[str,Any], trace: Trace) -> Dict[str,Any]:
        trace.add("advisor", "start")
        prompt = f"Analysis: {json.dumps(analysis)} User: {json.dumps(prefs)}"
        resp = await self.llm.generate(prompt)
        out_score = max(0, 100 - int(min(200, analysis['current_aqi'])))
        alert_flag = analysis['current_aqi'] > 200
        trace.add("advisor", "done")
        return {'summary': resp, 'outdoor_score': out_score, 'alert': alert_flag}

class ControllerAgent:
    def __init__(self, collector: DataCollectorAgent, analytics: AnalyticsAgent, advisor: AdvisorAgent, session: InMemorySessionService, memory: MemoryBank):
        self.collector = collector
        self.analytics = analytics
        self.advisor = advisor
        self.session = session
        self.memory = memory
    async def run(self, user_id: str, location: str, prefs: Dict[str,Any], trace: Trace) -> Dict[str,Any]:
        trace.add("controller", "start")
        collected = await self.collector.collect(location, trace)
        self.session.set(user_id, 'last_collected', collected)
        analysis = await self.analytics.analyze(collected, trace)
        advice = await self.advisor.advise(analysis, prefs, trace)
        self.memory.save(user_id, f"report_{int(time.time())}", {'location': location, 'analysis': analysis, 'advice': advice})
        trace.add("controller", "end")
        return {
            'user_id': user_id,
            'location': location,
            'analysis': analysis,
            'advice': advice,
            'trace': trace.to_dict()
        }

app = FastAPI(title="AQI Guardian")
session_service = InMemorySessionService()
memory_bank = MemoryBank()
aqi_tool = AQIFetcherTool()
weather_tool = WeatherTool()
collector = DataCollectorAgent(aqi_tool, weather_tool)
analytics = AnalyticsAgent()
llm_client = LLMClient()
advisor = AdvisorAgent(llm_client)
controller = ControllerAgent(collector, analytics, advisor, session_service, memory_bank)

class RunRequest(BaseModel):
    user_id: str
    location: str
    preferences: Optional[Dict[str,Any]] = {}

@app.post('/run_report')
async def run_report(req: RunRequest):
    trace = Trace()
    try:
        report = await controller.run(req.user_id, req.location, req.preferences, trace)
        return JSONResponse(content=report)
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

@app.get('/memory/{user_id}')
async def get_memory(user_id: str):
    res = memory_bank.query_recent(user_id, limit=10)
    return JSONResponse(content={'memory': res})



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

