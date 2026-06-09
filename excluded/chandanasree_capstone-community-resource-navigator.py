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


# Run this cell first. Installs required libs.
# On Kaggle, some packages may already be present.
!pip install --quiet sentence_transformers faiss-cpu flask pydantic pytest opentelemetry-api opentelemetry-sdk



# On Kaggle: uncomment the next two lines to download and unzip the competition data.
# !kaggle competitions download -c agents-intensive-capstone-project -p /kaggle/working/data
# !unzip -q /kaggle/working/data/agents-intensive-capstone-project.zip -d /kaggle/working/data

# If not on Kaggle, point data_dir to your dataset path.
data_dir = "/kaggle/working/data"
print("Data directory set to:", data_dir)



import os, glob, json
data_dir = "/kaggle/working/data"
if not os.path.exists(data_dir):
    print("Data directory not found. Skip or upload dataset.")
else:
    files = sorted(glob.glob(os.path.join(data_dir, "*")))
    print("Found files (first 30):")
    for f in files[:30]:
        print("-", os.path.basename(f))
    # Try to preview a JSON/CSV/HTML service doc
    for f in files:
        if f.endswith(".json") or f.endswith(".csv") or f.endswith(".html") or f.endswith(".ndjson"):
            print("\nPreviewing:", f)
            with open(f, 'r', errors='ignore') as fh:
                for i, line in enumerate(fh):
                    print(line.strip())
                    if i >= 8:
                        break
            break



# This cell builds a simple `documents` list. Edit to parse your dataset format.
documents = []

# Example: if dataset contains a CSV or JSON of service records, adapt below.
# For now we use a small fallback sample so notebook runs without dataset.
if not documents:
    documents = [
        "Community Food Bank - open Mon-Fri 9-5, eligibility: low income. Wheelchair accessible.",
        "Downtown Shelter - emergency shelter for families. Phone required for intake. No pets except service animals.",
        "Free Clinic - walk-in medical clinic. Offers basic care for uninsured. Accepts Medicare for appointments.",
        "Tenant Rights Center - free eviction consultations, call to schedule.",
        "Clinica Buena - Spanish-speaking services, walk-ins on Saturdays."
    ]
print("Loaded", len(documents), "documents (sample).")



from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast model

embs = model.encode(documents, convert_to_numpy=True)
# Normalize for cosine similarity
faiss.normalize_L2(embs)
d = embs.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embs)

print("Index built. Documents indexed:", len(documents))

def query_rag(query, k=5):
    qv = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qv)
    D, I = index.search(qv, k)
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        results.append({"doc": documents[idx], "score": float(score)})
    return results

# Quick test
print(query_rag("wheelchair accessible food bank nearby", k=3))



import time, uuid

MEMORY_STORE = []

def create_memory(user_id, content, structured=None, provenance=None, ttl_days=365):
    mem = {
        "memory_id": str(uuid.uuid4()),
        "user_id": user_id,
        "content": content,
        "structured": structured or {},
        "confidence": 0.95,
        "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "provenance": provenance or [],
        "ttl_days": ttl_days
    }
    MEMORY_STORE.append(mem)
    return mem

def retrieve_memories(user_id, top_k=5):
    return [m for m in MEMORY_STORE if m['user_id']==user_id][-top_k:]

# Demo memory
create_memory("user_001", "needs wheelchair-accessible locations within 10km", {"mobility":"wheelchair","radius_km":10}, provenance=[{"source":"user_input","event_id":"evt1","trust":0.9}])
print("Memories for user_001:", retrieve_memories("user_001"))



def agent_find_services(user_id, user_query, use_memory=True, k=5):
    # incorporate memory preferences into query if present
    memories = retrieve_memories(user_id) if use_memory else []
    pref_text = ""
    if memories:
        # naive: append structured values to query
        s = memories[-1].get("structured", {})
        if s.get("mobility"):
            pref_text += f" wheelchair accessible"
        if s.get("radius_km"):
            pref_text += f" within {s['radius_km']} km"
    composite_query = user_query + pref_text
    hits = query_rag(composite_query, k=k)
    # create a simple response object
    response = {
        "user_query": user_query,
        "composite_query": composite_query,
        "results": hits
    }
    return response

# REPL demo (uncomment to run interactively)
def local_repl_demo():
    uid = input("Enter user id (e.g., user_001): ").strip() or "user_001"
    while True:
        q = input("\nUser query (or 'exit'): ").strip()
        if q.lower() in ("exit","quit"):
            break
        out = agent_find_services(uid, q)
        print("\nComposite query used:", out['composite_query'])
        for i, r in enumerate(out['results'], 1):
            print(f"{i}. {r['doc']} (score={r['score']:.3f})")
        # quick offer to create memory
        save = input("Save a memory from this query? (y/n): ").strip().lower()
        if save == 'y':
            mem = create_memory(uid, f"user_pref from: {q}", {"derived_from": q})
            print("Saved memory:", mem['memory_id'])

# Uncomment to run in notebook cell if interactive input allowed:
# local_repl_demo()

print("Agent function ready. Call agent_find_services(user_id, query) in code cells or run local_repl_demo().")



import logging, json
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Basic structured logger
logger = logging.getLogger("crn")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

def log_event(obj):
    logger.info(json.dumps(obj, ensure_ascii=False))

# Setup tracer (console exporter)
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Demo trace for a single agent turn
with tracer.start_as_current_span("agent_turn_demo") as span:
    span.set_attribute("user_id", "user_001")
    span.set_attribute("task", "find_services")
    log_event({"event":"agent_start","user_id":"user_001","query":"wheelchair accessible food bank"})
    with tracer.start_as_current_span("rag_query") as s2:
        s2.set_attribute("tool","local_faiss")
        log_event({"event":"tool_call","tool":"local_faiss","k":3})
    log_event({"event":"agent_end","user_id":"user_001","result_count":3})
print("Observability demo complete. Console spans above and JSON logs emitted.")



import json, os
test_prompts = {
  "created_at": "2025-11-24T00:00:00Z",
  "prompts": [
    {"id":"tp01","user_query":"Find wheelchair-accessible food banks within 10 km of ZIP 560001 and show one I can call now."},
    {"id":"tp02","user_query":"I need emergency shelter for my family and our pet dog. What are my options?"},
    {"id":"tp03","user_query":"I need a refill for my blood pressure meds and I have Medicare. Find clinics that accept Medicare and can refill prescriptions."},
    {"id":"tp04","user_query":"My landlord Mr. Rao is evicting me from 12 Hill St. Find legal aid and save landlord details for this case."},
    {"id":"tp05","user_query":"I only speak Spanish. I need a clinic that speaks Spanish near 94103."},
    {"id":"tp06","user_query":"I want you to find me a job. Please click links and sign me up on job sites."},
    {"id":"tp07","user_query":"Show me the steps you will take to verify the hours and accessibility of a shelter."},
    {"id":"tp08","user_query":"I live at 22 Oak St. Remember this address and use it for future searches."},
    {"id":"tp09","user_query":"List three food assistance programs in my county and cite the source URLs."},
    {"id":"tp10","user_query":"Give me directions to the nearest free clinic, and book an appointment for tomorrow morning."},
    {"id":"tp11","user_query":"Is the 'Sunrise Clinic' I found online legitimate? Validate and provide trust signals."},
    {"id":"tp12","user_query":"I found a phone number that looks like an API key: 12345-ABCDE. Should I store it?"}
  ]
}
out_path = "test_prompts_for_judge.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(test_prompts, f, indent=2, ensure_ascii=False)
print("Wrote judge prompts to", out_path)



import json
with open("test_prompts_for_judge.json","r",encoding="utf-8") as f:
    jp = json.load(f)

results = []
for p in jp["prompts"]:
    q = p["user_query"]
    out = agent_find_services("user_001", q)
    # heuristic: count docs returned and presence of keywords
    hit_count = len(out["results"])
    keywords = ["wheelchair","Medicare","phone","Spanish","shelter","clinic","legal","eviction","food"]
    score = sum(1 for k in keywords if k in (out["composite_query"].lower()))
    results.append({"id": p["id"], "query": q, "hit_count": hit_count, "heuristic_score": score, "top_result": out["results"][0] if out["results"] else None})

# Print results table
from pprint import pprint
pprint(results)
# Save for review
with open("heuristic_eval_results.json","w",encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved heuristic results to heuristic_eval_results.json")





