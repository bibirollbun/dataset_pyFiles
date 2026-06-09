# ============================================================
# TrustNet AI — Dynamic LLM Trust Scoring (Gemini-Driven)
# ============================================================

# 1) Imports & setup
import os
import json
import asyncio
import re
from urllib.parse import urlparse
import nest_asyncio
nest_asyncio.apply()

# ADK imports
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search

print("✅ ADK imports loaded.")

# 2) Config & secrets
MODEL_NAME = "gemini-2.5-flash-lite"
MIN_HIGH_TRUST_SCORE = 85

# Check for API key (for real search) — fallback if missing
try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ GOOGLE_API_KEY loaded from Kaggle Secrets.")
except:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        print("✅ GOOGLE_API_KEY loaded from environment.")
    else:
        GOOGLE_API_KEY = None
        print("⚠️ GOOGLE_API_KEY not found. Notebook will use fallback search results.")

# ============================================================
# Utility Functions
# ============================================================

def extract_text_from_event_like(obj):
    if obj is None: return ""
    if isinstance(obj, str): return obj
    if isinstance(obj, list): return "\n".join([extract_text_from_event_like(i) for i in obj])
    if isinstance(obj, dict):
        for key in ("text", "content", "message", "output", "response"):
            if key in obj: return extract_text_from_event_like(obj[key])
        for v in obj.values():
            t = extract_text_from_event_like(v)
            if t: return t
        return ""
    if hasattr(obj, "content") and getattr(obj.content, "parts", None):
        return "\n".join([p.text for p in obj.content.parts if hasattr(p, "text")])
    if hasattr(obj, "text"): return obj.text
    return str(obj)

def extract_json_from_text(text):
    if not text: return None
    try:
        return json.loads(text)
    except: pass
    m = re.search(r"(\{(?:.|\n)*\})", text)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r"(\[(?:.|\n)*\])", text)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    return None

def extract_domain(url):
    try:
        if not url: return "unknown"
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."): domain = domain[4:]
        return domain or "unknown"
    except:
        return "unknown"

# ============================================================
# Extract search results (with fallback)
# ============================================================

def extract_search_items_from_adk_output(raw_response, max_results=6, fallback=True):
    items = []
    text = extract_text_from_event_like(raw_response)

    # Attempt to parse JSON
    parsed = extract_json_from_text(text)
    if parsed:
        if isinstance(parsed, dict) and "search_results" in parsed:
            return parsed["search_results"][:max_results]
        if isinstance(parsed, list):
            return parsed[:max_results]

    # fallback mode
    if fallback:
        print("⚠️ Using fallback search results")
        fallback_results = [
            {
                "title": "U.S. Passport Renewal Guidelines (Fallback)",
                "url": "https://travel.state.gov/content/travel/en/passports/renew.html",
                "snippet": "You can renew your U.S. passport either online or by mail, following eligibility requirements and DS-82 form instructions."
            },
            {
                "title": "Passport Renewal Process Overview",
                "url": "https://www.usa.gov/renew-passport",
                "snippet": "Renew your passport with the appropriate forms, fees, and supporting documents. Online or mail-in methods are available."
            }
        ]
        return fallback_results[:max_results]

    # last resort
    return [{"title": "", "url": "", "snippet": text[:2000]}]

# ============================================================
# Build Agents
# ============================================================

research_agent = Agent(
    name="ResearchAgent",
    model=Gemini(model=MODEL_NAME),
    instruction=(
        "Use google_search tool to find 3–6 relevant results. "
        "Return JSON: { 'search_results': [ { 'title': '', 'url': '', 'snippet': '' }, ... ] }."
    ),
    tools=[google_search] if GOOGLE_API_KEY else []  # only enable tool if key exists
)

trust_agent = Agent(
    name="TrustScorer",
    model=Gemini(model=MODEL_NAME),
    instruction=(
        "Score trustworthiness (0–100). Return EXACT JSON:\n"
        "{ score:int, classification:'high_trust'|'low_trust', confidence:float, explanation:string }"
    )
)

synth_agent = Agent(
    name="SynthesizerAgent",
    model=Gemini(model=MODEL_NAME),
    instruction=(
        "Given a query and allowed sources, produce a 2–5 sentence answer using ONLY those sources. "
        "Include a 'Sources Used' block."
    )
)

# ============================================================
# Orchestrator
# ============================================================

async def orchestrate_query_dynamic(query, min_high_trust_score=MIN_HIGH_TRUST_SCORE, max_results=6):

    # ---- 1. Search ----
    runner_search = InMemoryRunner(agent=research_agent)
    raw = await runner_search.run_debug(query) if GOOGLE_API_KEY else ""
    search_results = extract_search_items_from_adk_output(raw, max_results=max_results, fallback=True)

    if not search_results:
        return {
            "query": query,
            "synth_text": "No search results.",
            "evaluated_sources": [],
            "allowed_sources": []
        }

    # ---- 2. Trust scoring ----
    evaluated_sources = []
    for r in search_results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        domain = extract_domain(url)

        trust_prompt = f"""
Query: {query}
Source Domain: {domain}
Title: {title}
Snippet: {snippet[:1200]}

Return EXACT JSON with:
- score
- classification
- confidence
- explanation
"""

        runner_trust = InMemoryRunner(agent=trust_agent)
        trust_raw = await runner_trust.run_debug(trust_prompt)
        trust_text = extract_text_from_event_like(trust_raw)
        trust_json = extract_json_from_text(trust_text)

        if not trust_json:
            score = 80 if domain.endswith((".gov", ".edu")) else 70
            trust_json = {
                "score": score,
                "classification": "high_trust" if score >= min_high_trust_score else "low_trust",
                "confidence": 0.7,
                "explanation": "Fallback applied (invalid JSON from LLM)."
            }

        evaluated_sources.append({
            "title": title,
            "url": url,
            "domain": domain,
            "snippet": snippet,
            "score": int(trust_json["score"]),
            "classification": trust_json["classification"],
            "confidence": float(trust_json["confidence"]),
            "explanation": trust_json["explanation"],
            "trust_raw_text": trust_text
        })

    # ---- 3. Filter allowed sources ----
    allowed = [s for s in evaluated_sources if s["score"] >= min_high_trust_score]
    if not allowed:
        allowed = [s for s in evaluated_sources if s["score"] >= 70]
    if not allowed:
        allowed = sorted(evaluated_sources, key=lambda x: x["score"], reverse=True)[:2]

    allowed_sources_for_prompt = [{
        "title": s["title"],
        "url": s["url"],
        "domain": s["domain"],
        "snippet": s["snippet"][:1000],
        "score": s["score"],
        "classification": s["classification"],
        "confidence": s["confidence"],
        "explanation": s["explanation"]
    } for s in allowed]

    # ---- 4. Synthesize final answer ----
    runner_synth = InMemoryRunner(agent=synth_agent)
    synth_prompt = f"""
Query: {query}

Allowed sources (JSON):
{json.dumps(allowed_sources_for_prompt, indent=2)}

Write a 2–5 sentence answer using ONLY these sources.
Include a 'Sources Used' block.
"""

    synth_raw = await runner_synth.run_debug(synth_prompt)
    synth_text = extract_text_from_event_like(synth_raw)

    return {
        "query": query,
        "synth_text": synth_text,
        "evaluated_sources": evaluated_sources,
        "allowed_sources": allowed_sources_for_prompt
    }

# ============================================================
# Demo runner
# ============================================================

async def run_demo():
    query = "What are the official rules for renewing a U.S. passport?"
    result = await orchestrate_query_dynamic(query)

    print("\n=== USER QUERY ===")
    print(result["query"])
    print("\n=== FINAL ANSWER ===")
    print(result["synth_text"])

    print("\n=== ALLOWED SOURCES ===")
    for s in result["allowed_sources"]:
        print(f"- {s['title']} | {s['domain']} | score={s['score']}")

    print("\n=== EVALUATED SOURCES ===")
    for s in result["evaluated_sources"]:
        print(f"- {s['domain']} | score={s['score']} | {s['classification']}")

    return result

# ============================================================
# Run & Export CSV
# ============================================================

result = await run_demo()

# ============================================================
# Run & Export Everything in One CSV
# ============================================================

import pandas as pd

# Prepare data for CSV
rows = []

for s in result["evaluated_sources"]:
    rows.append({
        "query": result["query"],
        "final_answer": result["synth_text"],
        "source_title": s.get("title", ""),
        "source_url": s.get("url", ""),
        "source_domain": s.get("domain", ""),
        "source_snippet": s.get("snippet", ""),
        "score": s.get("score", 0),
        "classification": s.get("classification", ""),
        "confidence": s.get("confidence", 0.0),
        "explanation": s.get("explanation", "")
    })

# Convert to DataFrame
df_combined = pd.DataFrame(rows)

# Save single CSV
df_combined.to_csv("/kaggle/working/submission.csv", index=False)

print("File: submission.csv")

