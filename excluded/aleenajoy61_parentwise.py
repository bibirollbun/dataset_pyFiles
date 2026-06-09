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


!pip install google-generativeai pandas chromadb pypdf langchain langchain-community


!pip install -q google-generativeai
!pip install -q google-generativeai langchain chromadb langchain-community pypdf



import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

import pandas as pd
import json
import os
import time
import logging
from typing import List, Dict, Any
from pprint import pprint

# Logging for observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
GOOGLE_API_KEY = None
try:
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini key loaded from Kaggle Secrets.")
except Exception as e:
    print("No Kaggle secret found or error loading key. Fallback mode available.")
    GOOGLE_API_KEY = None

# Toggle to allow demo without API key (uses Mock LLM for core flows)
NO_KEY_FALLBACK = GOOGLE_API_KEY is None



genai.configure(api_key=GOOGLE_API_KEY)   # IMPORTANT: Leave blank for judging

MODEL = "gemini-1.5-flash"
EMBED_MODEL = "models/embedding-001"


# Observability primitives
class Tracer:
    def __init__(self):
        self.events = []
    def trace(self, step:str, details:dict=None):
        self.events.append({"ts": time.time(), "step": step, "details": details or {}})
    def dump(self):
        return self.events

class Metrics:
    def __init__(self):
        self.c = {}
    def incr(self, k, v=1):
        self.c[k] = self.c.get(k,0) + v
    def export(self):
        return dict(self.c)

tracer = Tracer()
metrics = Metrics()



# Mock resource dataset for NearbyFinder demo (you can extend)
MOCK_RESOURCES = {
    "hospitals": [{"name":"Sunrise Pediatric Clinic","lat":12.9718,"lng":77.5946,"rating":4.6}],
    "daycares": [{"name":"Little Stars Daycare","lat":12.9720,"lng":77.5950,"rating":4.8}],
    "coaches": [{"name":"Glide Pro Skating","lat":12.9700,"lng":77.5965,"rating":4.9}],
    "parks": [{"name":"Lakeview Park","lat":12.9730,"lng":77.5940,"rating":4.6}]
}

from math import radians, sin, cos, sqrt, asin
def haversine(lat1,lon1,lat2,lon2):
    R=6371.0
    dlat=radians(lat2-lat1); dlon=radians(lon2-lon1)
    a=sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*asin(sqrt(a))*R

class NearbyFinder:
    def __init__(self, resources):
        self.resources = resources
    def find(self, resource_type, location=(12.9718,77.5946), radius_km=5.0, top_n=5):
        lat0,lng0 = location
        items = self.resources.get(resource_type,[])
        enriched=[]
        for it in items:
            d = round(haversine(lat0,lng0,it["lat"],it["lng"]),2)
            it2=dict(it); it2["distance_km"]=d; enriched.append(it2)
        enriched=[i for i in enriched if i["distance_km"]<=radius_km]
        enriched.sort(key=lambda x:(x["distance_km"],-x.get("rating",0)))
        tracer.trace("nearby_find", {"resource":resource_type, "count": len(enriched)})
        metrics.incr("nearby_calls")
        return enriched[:top_n]

class MockSearch:
    def __init__(self, resources):
        self.resources = resources
    def search(self, q, location=None, limit=5):
        ql=q.lower()
        key="coaches"
        if "hospital" in ql or "clinic" in ql: key="hospitals"
        if "daycare" in ql: key="daycares"
        res = self.resources.get(key,[])[:limit]
        tracer.trace("mock_search", {"query": q, "results": len(res)})
        metrics.incr("search_calls")
        return [{"title": r["name"], "rating": r.get("rating"), "distance_km": round(haversine(location[0],location[1],r["lat"],r["lng"]),2)} for r in res]

class PathGenerator:
    def generate(self, child_profile, goal):
        age = child_profile.get("age", 5)
        sessions = 3 if age<=5 else (4 if age<=12 else 5)
        duration = 20 if age<=5 else (40 if age<=12 else 60)
        schedule=[]
        week=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        for i in range(sessions):
            schedule.append({"day":week[i%7],"activity":f"{goal} practice","mins":duration})
        tracer.trace("path_generated", {"goal":goal, "sessions": sessions})
        metrics.incr("path_calls")
        return {"child": child_profile.get("name","child"), "goal":goal, "sessions_per_week":sessions, "session_length":duration, "schedule": schedule}



import uuid
from datetime import datetime

class MemoryBank:
    def __init__(self):
        self.data={"children":{}, "family":{}}
    def add_child(self, cid, profile):
        self.data["children"][cid]=profile
        tracer.trace("memory_add_child", {"id":cid})
    def update_child(self, cid, patch):
        self.data["children"].setdefault(cid,{})
        self.data["children"][cid].update(patch)
        tracer.trace("memory_update_child", {"id":cid, "patch":patch})
    def get_child(self, cid):
        return self.data["children"].get(cid)
    def list_children(self):
        return list(self.data["children"].keys())

class InMemorySessionService:
    def __init__(self):
        self.sessions={}
    def create(self, meta=None):
        sid=str(uuid.uuid4()); self.sessions[sid]={"created":datetime.now().isoformat(),"context":meta or {}}
        tracer.trace("session_create", {"sid": sid})
        return sid
    def get(self,sid):
        return self.sessions.get(sid)
    def update(self,sid,k,v):
        self.sessions.setdefault(sid,{"context":{}}); self.sessions[sid]["context"][k]=v
        tracer.trace("session_update", {"sid":sid,"k":k})

# instantiate
memory = MemoryBank()
sessions = InMemorySessionService()

# seed example
memory.add_child("ria", {"name":"Ria","age":8,"interests":["skating","drawing"],"strengths":["balance"],"weaknesses":["patience"],"performance":{"math":"B"}})



class MockLLM:
    def generate(self, prompt, max_tokens=512):
        # deterministic short response for demos; structured to mimic Thought/Action/Answer
        if "recommend a sport" in prompt.lower():
            return {"text":"Thought: consider age and allergies.\nAction: generate_plan:skating\nAnswer: Badminton or Skating are suitable based on indoor availability."}
        return {"text":"Thought: analyze\nAction: none\nAnswer: Provide gentle parenting steps."}

# Gemini wrapper (calls genai.generate_text or similar)
class GeminiLLM:
    def __init__(self, model="gemini-1.5-flash"):
        self.model = model
    def generate(self, prompt, max_tokens=512):
        # Using Google Generative SDK
        resp = genai.generate_text(model=self.model, prompt=prompt, max_output_tokens=max_tokens)
        # adapt to our expected return shape
        text = resp.text if hasattr(resp, "text") else str(resp)
        return {"text": text}

# select LLM
LLM = MockLLM() if NO_KEY_FALLBACK else GeminiLLM()



# ReAct engine
import re
class ReActEngine:
    def __init__(self, llm, tools, tracer=None, metrics=None):
        self.llm=llm; self.tools=tools; self.tracer=tracer; self.metrics=metrics
    def _parse(self, txt):
        # tolerant parse for blocks like "Thought: ...", "Action: ...", "Answer: ..."
        parsed={"Thought":"","Action":"","Answer":""}
        for line in txt.splitlines():
            if line.lower().startswith("thought:"): parsed["Thought"] += line.split(":",1)[1].strip()+" "
            elif line.lower().startswith("action:"): parsed["Action"] += line.split(":",1)[1].strip()+" "
            elif line.lower().startswith("answer:"): parsed["Answer"] += line.split(":",1)[1].strip()+" "
        return parsed
    def run(self, prompt, context=None, max_steps=3):
        context = context or {}
        observation=""
        for step in range(max_steps):
            full_prompt = f"Context: {context}\nObservation: {observation}\nUser Query: {prompt}\nRespond with Thought:, Action:, Answer:"
            self.tracer and self.tracer.trace("llm_call",{"step":step})
            self.metrics and self.metrics.incr("llm_calls")
            resp = self.llm.generate(full_prompt)
            parsed = self._parse(resp.get("text",""))
            self.tracer and self.tracer.trace("llm_parsed", parsed)
            action = parsed.get("Action","").strip()
            if action:
                # handle simple actions
                if action.startswith("find_nearby"):
                    parts = action.split(":")
                    res_type = parts[1] if len(parts)>1 else "coaches"
                    res = self.tools["nearby"].find(res_type, context.get("parent_location",(12.9718,77.5946)))
                    observation = f"Found {len(res)} {res_type}."
                elif action.startswith("search:"):
                    q = action.split(":",1)[1]
                    res = self.tools["search"].search(q, location=context.get("parent_location"))
                    observation = f"Search returned {len(res)} results."
                elif action.startswith("generate_plan:"):
                    goal = action.split(":",1)[1]
                    plan = self.tools["path"].generate(context.get("child_profile",{}), goal)
                    observation = f"Plan generated with {plan['sessions_per_week']} sessions."
                else:
                    observation = "Action unrecognized."
                self.tracer and self.tracer.trace("tool_exec", {"action":action, "obs":observation})
                self.metrics and self.metrics.incr("tool_actions")
                # continue loop to let LLM use observation
            else:
                ans = parsed.get("Answer","").strip()
                if ans:
                    self.tracer and self.tracer.trace("final_answer", {"answer": ans})
                    return {"answer": ans, "trace": self.tracer.dump() if self.tracer else None}
        # fallback
        return {"answer": parsed.get("Answer","I'm not sure; please clarify."), "trace": self.tracer.dump() if self.tracer else None}



# Tools instances
nearby = NearbyFinder(MOCK_RESOURCES)
search_tool = MockSearch(MOCK_RESOURCES)
path_gen = PathGenerator()

# ReAct engine instance
react_engine = ReActEngine(LLM, tools={"nearby":nearby,"search":search_tool,"path":path_gen}, tracer=tracer, metrics=metrics)

# Agent implementations
class BaseAgent:
    def __init__(self, name, memory=None, tools=None, engine=None):
        self.name=name; self.memory=memory; self.tools=tools or {}; self.engine=engine
    def handle(self, query, ctx):
        raise NotImplementedError

class DailyAgent(BaseAgent):
    def handle(self, query, ctx):
        if any(w in query.lower() for w in ["hospital","clinic","daycare","park","nearby","coach","tutor"]):
            return {"type":"nearby","results": self.tools["nearby"].find("hospitals", ctx.get("parent_location"))}
        return self.engine.run(query, context=ctx)

class LifeAgent(BaseAgent):
    def handle(self, query, ctx):
        compact = {k: ctx.get("child_profile",{}).get(k) for k in ("name","age","interests","strengths","weaknesses")}
        return self.engine.run(query, context={"child_profile": compact, "parent_location": ctx.get("parent_location")})

class ResearchAgent(BaseAgent):
    def handle(self, query, ctx):
        return {"type":"search","results": self.tools["search"].search(query, location=ctx.get("parent_location"))}

class ProgressAgent(BaseAgent):
    def handle(self, query, ctx):
        child_id = ctx.get("child_id")
        c = self.memory.get_child(child_id) if self.memory else None
        perf = c.get("performance",{}) if c else {}
        if any(g in ["C","D","F"] for g in perf.values()):
            return {"replan":True,"suggestion":"increase tutoring"}
        return {"replan":False,"suggestion":"on track"}

class RouterAgent:
    def __init__(self, agents):
        self.agents = agents
    def handle(self, query, ctx):
        ql=query.lower()
        if any(w in ql for w in ["school","college","degree","career","subject"]):
            return self.agents["life"].handle(query, ctx)
        if any(w in ql for w in ["hospital","clinic","daycare","park","coach","tutor"]):
            return self.agents["daily"].handle(query, ctx)
        if any(w in ql for w in ["progress","review","milestone","replan"]):
            return self.agents["progress"].handle(query, ctx)
        if any(w in ql for w in ["find","where","search","nearby"]):
            return self.agents["research"].handle(query, ctx)
        return self.agents["life"].handle(query, ctx)

# instantiate agents map
agents_map = {
    "daily": DailyAgent("daily", memory=memory, tools={"nearby":nearby,"search":search_tool}, engine=react_engine),
    "life": LifeAgent("life", memory=memory, tools={"path":path_gen,"search":search_tool}, engine=react_engine),
    "research": ResearchAgent("research", memory=memory, tools={"search":search_tool}),
    "progress": ProgressAgent("progress", memory=memory)
}
router = RouterAgent(agents_map)



def safety_filter(text):
    lower = text.lower()
    risky = ["seizure","stop breathing","choking","suicide","self-harm","blood loss"]
    if any(r in lower for r in risky):
        return "This sounds like a possible medical emergency. Please seek immediate medical help or call emergency services."
    return text



def parentwise_query(user_query, session_ctx):
    tracer.trace("parent_query_received", {"query": user_query})
    # Run router to get an agent response
    res = router.handle(user_query, session_ctx)
    # If response is engine output
    if isinstance(res, dict) and "answer" in res:
        out = res["answer"]
    elif isinstance(res, dict) and res.get("type")=="search":
        out = json.dumps(res["results"], indent=2, ensure_ascii=False)
    elif isinstance(res, dict) and res.get("type")=="nearby":
        out = json.dumps(res["results"], indent=2, ensure_ascii=False)
    else:
        # if the agent returned an engine-run object (ReAct)
        out = res if isinstance(res, str) else str(res)
    out = safety_filter(out)
    # Save to memory history
    memory.update_child(session_ctx.get("child_id","ria"), {"last_query": user_query})
    tracer.trace("final_output", {"output": out})
    return out



# Demo 1: Find nearby hospitals and daycares
ctx = {"parent_location": (12.9718,77.5946), "child_profile": memory.get_child("ria"), "child_id":"ria"}
print("Demo 1 - Nearby search:")
print(parentwise_query("Find nearby hospitals and daycares for my newborn", ctx)[:1000])
print("\n--- TRACE SAMPLE ---")
pprint(tracer.dump()[-6:])



# Demo 2: Life decision - recommend a sport + generate plan
print("\nDemo 2 - Recommend sport + plan:")
print(parentwise_query("Recommend a sport for my 10-year-old and build a training plan", {"parent_location":(12.9718,77.5946), "child_profile":{"name":"TestKid","age":10}, "child_id":"testkid"}))



# Demo 3: Behavior guidance (ReAct reasoning fallback)
print("\nDemo 3 - Behavior guidance:")
print(parentwise_query("My 5-year-old refuses to sleep unless I stay with him. What should I do?", {"parent_location":(12.9718,77.5946), "child_profile":memory.get_child("ria"), "child_id":"ria"}) )



# RAG building (only run if you uploaded PDFs)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

pdf_folder = "/kaggle/input/parentwise-resources"
if os.path.exists(pdf_folder) and len(os.listdir(pdf_folder))>0 and not NO_KEY_FALLBACK:
    docs=[]
    for f in os.listdir(pdf_folder):
        if f.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, f))
            pages = loader.load()
            docs.extend(pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    print("PDF chunks created:", len(chunks))
    # embed chunks directly using genai.embed_content
    def embed_text(text):
        r = genai.embed_content(model=EMBED_MODEL, content=text, task_type="retrieval_document")
        # r may be a dict with 'embedding' or nested; adapt accordingly
        return r.get("embedding") if isinstance(r, dict) else r
    texts = [c.page_content for c in chunks]
    embeddings = [embed_text(t) for t in texts]
    # Build Chroma DB from raw embeddings (Chroma can accept embeddings)
    from langchain_community.vectorstores import Chroma
    vectordb = Chroma.from_embeddings(embeddings=embeddings, metadatas=[c.metadata for c in chunks], ids=[f"c{i}" for i in range(len(embeddings))], persist_directory="./parentwise_vectordb")
    print("Vector DB ready.")
else:
    print("Skipping RAG build (no PDFs found or no API key).")



# Basic assertions
assert "ria" in memory.list_children(), "Seed child 'ria' must exist"
print("Memory children:", memory.list_children())
print("Metrics snapshot:", metrics.export())


