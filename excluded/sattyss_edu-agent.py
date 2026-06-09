# Cell 00 - imports & environment setup
import os, json, time, uuid, sqlite3, logging, asyncio, re
from pathlib import Path
from typing import List, Dict, Any

# Data folders
WORK_DIR = Path("/kaggle/working/edu_agents")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("edu_agents")



# Cell 01 - read Kaggle secret (optional - only needed for real Gemini calls)
from kaggle_secrets import UserSecretsClient
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    logger.info("✅ GOOGLE_API_KEY loaded from Kaggle secrets.")
except Exception as e:
    logger.warning("GOOGLE_API_KEY not found - running in offline (stub) mode. Error: %s", e)



# Cell 02 - ADK / GenAI imports and retry config
from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner, Runner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Retry options for stable API calls
retry_config = types.HttpRetryOptions(attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429,500,503,504])

def make_model(model_name="gemini-2.5-flash-lite"):
    """Return a configured Gemini model wrapper (ADK)."""
    return Gemini(model=model_name, retry_options=retry_config)



# Cell 03 - persistent sqlite DB for transcripts, flashcards, logs, metrics
DB_PATH = WORK_DIR / "agents_memory.db"
conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
cur = conn.cursor()

cur.execute('''CREATE TABLE IF NOT EXISTS transcripts (id TEXT PRIMARY KEY, title TEXT, text TEXT, created_ts INTEGER)''')
cur.execute('''CREATE TABLE IF NOT EXISTS flashcards (id TEXT PRIMARY KEY, transcript_id TEXT, card_json TEXT, created_ts INTEGER)''')
cur.execute('''CREATE TABLE IF NOT EXISTS agent_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, transcript_id TEXT, agent_name TEXT, action TEXT, payload TEXT, created_ts INTEGER)''')
cur.execute('''CREATE TABLE IF NOT EXISTS metrics (name TEXT PRIMARY KEY, value REAL)''')
cur.execute('''CREATE TABLE IF NOT EXISTS vector_mem (id TEXT PRIMARY KEY, transcript_id TEXT, chunk_index INTEGER, embedding BLOB, text TEXT, created_ts INTEGER)''')
conn.commit()

def log_agent(tid, agent, action, payload):
    cur.execute("INSERT INTO agent_logs (transcript_id, agent_name, action, payload, created_ts) VALUES (?, ?, ?, ?, ?)",
                (tid, agent, action, json.dumps(payload), int(time.time())))
    conn.commit()

def metric_inc(name: str, inc: float = 1.0):
    cur.execute("INSERT OR REPLACE INTO metrics (name, value) VALUES (?, COALESCE((SELECT value FROM metrics WHERE name=?),0) + ?)",
                (name, name, inc))
    conn.commit()



# Cell 04 - transcript save/load helpers
def save_transcript(title: str, text: str) -> str:
    tid = str(uuid.uuid4())
    cur.execute("INSERT INTO transcripts (id, title, text, created_ts) VALUES (?, ?, ?, ?)",
                (tid, title, text, int(time.time())))
    conn.commit()
    logger.info("Saved transcript %s", tid)
    return tid

def load_transcript(tid: str) -> Dict[str, Any]:
    cur.execute("SELECT id, title, text, created_ts FROM transcripts WHERE id=?", (tid,))
    row = cur.fetchone()
    if not row:
        raise KeyError("Transcript not found")
    return {"id": row[0], "title": row[1], "text": row[2], "created_ts": row[3]}



# Cell 05 - create a sample transcript (or load your uploaded transcript file path)
sample = """
Introduction To Generative aI
meet Emma a graphic designer working on
a new project one day her colleague
mentions a tool that helps create
designs images and text using AI
intrigued Emma wonders how AI can create
something from scratch her curiosity
grows and she decides to dive deeper
into this new technology called
generative AI generative AI refers to a
What Is Generative AI?
type of artificial intelligence designed
to create new content such as text
images music and videos unlike
traditional AI which analyzes or
categorizes data generative AI produces
original content based on patterns
learned from vast data sets essentially
it generates new unique material these
models are often trained on large
amounts of data and use sophisticated
algorithms to mimic human creativity
tools like chat GPT or DOL e can create
art write essays or simulate
conversations by generating output based
on user prompts generative AI has a wide
range of applications content creation
Generative aI Applications
tools like gp4 generate text blog posts
stories and essays from simple prompts
Art and Design AI models such as doly
generates Unique Images and Designs
based on text descriptions transforming
creativity and art music and audio AI
can compose music or replicate voices
offering new possibilities for music
Physicians and audio
Engineers Healthcare generative AI
simulates disease progression or creates
synthetic Medical Data helping doctors
gain faster insights for research let's
take image generation as an example to
How Generative AI Works?
explain how generative AI
Works data collection and learning AI
models like Dolly are trained on large
data sets of images paired with text
descriptions these data sets teach the
model to recognize different objects
colors St Styles and how to associate
text with corresponding images the more
data the AI learns from the better it
can generate accurate and diverse images
based on user prompts neural networks
and
Transformers when Emma inputs a prompt
like a cat wearing sunglasses the
Transformer model processes the text
recognizing words like cat and
sunglasses and links them to images at
learn from during training Transformers
help the AI decide how to combine these
elements into a coherent image tokens
and context the text input such as a cat
wearing sunglasses is split into smaller
parts called tokens the AI processes
each token and understands their
relationship for instance it knows the
sunglasses should be placed on the cat
creating a contextually accurate image
feedback mechanism generative AI models
improve through feedback after
generating an image users provide
feedback on the accuracy or quality of
the output if Emma's generated image
shows the sunglasses floating beside the
cat she can mark it as incorrect the
model uses this feedback to improve
future image
Generations reinforcement learning
reinforcement learning further enhances
the ai's ability the model is rewarded
when it generates accurate images and
corrected when it makes mistakes for
example when Emma describes a Sunset and
the AI produces a vibrant Sunset image
it receives positive reinfor ment over
time this method refines the model's
ability to generate better images data
science and AI models data scientists
curate the training data and Define the
parameters that help the AI generate
accurate images the more varied the data
set the more versatile the AI becomes in
generating diverse types of content
Advanced models use billions of
parameters which are settings that guide
the AI in processing data and generating
outputs generating original content once
trained the model can generate original
images for example Emma might describe a
futuristic cityscape and the AI would
produce a unique image based on what it
learned the generated image isn't just a
copy of P data but an entirely New
Creation showcasing the ai's ability to
combine learn patterns and creativity
"""
p = WORK_DIR / "sample_transcript.txt"
p.write_text(sample, encoding="utf-8")
transcript_text = p.read_text(encoding="utf-8")
transcript_id = save_transcript("sample_transcript.txt", transcript_text)
logger.info("Ingested sample transcript id=%s", transcript_id)



# Cell 06 - chunk helper + safe LLM output parsing + stub lookup
def chunk_text(text: str, chunk_size:int=1500) -> List[str]:
    text = text.strip()
    if not text: return []
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def safe_parse_json_from_text(raw: str):
    """Strip code fences and try JSON parse, return None on failure."""
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        return None

# Offline safe stub for definition lookup (used when google_search or external dictionary is unavailable)
def lookup_definition_stub(term: str):
    return {"term": term, "definition": f"(stub) concise definition for {term}", "source":"local-stub"}



# Cell 07 - session state tools (used by agents that need to store small key-value state)
def save_userinfo(tool_context: ToolContext, user_name: str, country: str):
    tool_context.state["user:name"] = user_name
    tool_context.state["user:country"] = country
    return {"status":"ok"}

def retrieve_userinfo(tool_context: ToolContext):
    return {"status":"ok", "user_name": tool_context.state.get("user:name", None), "country": tool_context.state.get("user:country", None)}



# Cell 08 (updated) - improved agents + small helper to fetch definitions via ADK tools
from google.adk.tools import google_search
from google.adk.runners import InMemoryRunner

# --- Jargon extractor (explicit and conservative)
jargon_agent = Agent(
    name="JargonResearchAgent",
    model=make_model(),
    instruction=(
        "Extract up to 25 domain-specific technical terms or short phrases from the input transcript text. "
        "Output MUST be valid JSON: a JSON array of strings, e.g. [\"term1\",\"term2\",...]. "
        "Avoid generic stopwords and return concise terms (no long sentences)."
    ),
    tools=[],   # keep empty or add google_search if you want live lookups inside this agent
    output_key="jargon_terms"
)

# --- Definition agent (uses google_search tool when available)
definition_agent = Agent(
    name="DefinitionAgent",
    model=make_model(),
    tools=[google_search],   # ADK tool — agent may call this if runtime allows
    instruction=(
        "You are a glossary generator. Input: a JSON array of short terms (e.g. [\"term1\",\"term2\"]).\n\n"
        "For each term return an object with keys: "
        "'term' (original term), "
        "'definition' (1-2 sentence student-friendly concise definition), "
        "'source' (short string: authoritative source or url).\n\n"
        "Rules:\n"
        "- Return ONLY valid JSON (no commentary).\n"
        "- For each term, if you can find a short authoritative definition via tool lookup, include it and provide the source (domain or URL).\n"
        "- If the agent cannot confidently find a concise definition, set 'definition' to \"No concise definition found.\" and set 'source' to 'none'.\n\n"
        "Output example:\n"
        '[{\"term\":\"RAG\",\"definition\":\"Retrieval-augmented generation is...\",\"source\":\"https://en.wikipedia.org/...\"}, ...]'
    ),
    output_key="definitions"
)

# --- Chunk summarizer and downstream agents (unchanged, but re-declare explicit tooling)
chunk_summarizer = Agent(
    name="ChunkSummarizer",
    model=make_model(),
    instruction="Summarize the provided chunk into 3-6 concise bullet points in markdown.",
    output_key="chunk_summary"
)

aggregator_explicit = Agent(
    name="SummaryAggregatorExplicit",
    model=make_model(),
    instruction=(
        "You are an aggregator. Given multiple chunk summaries (plain text) combine them into a single structured summary "
        "with a short 'Key points' list. Return ONLY the summary text (no JSON)."
    ),
    output_key="final_summary_explicit"
)

critic_explicit = Agent(
    name="CriticAgentExplicit",
    model=make_model(),
    instruction=(
        "You are a constructive critic. Given a summary, return EXACTLY 'APPROVED' if it's high-quality; otherwise "
        "return 2-3 specific actionable suggestions. Return only the critique."
    ),
    output_key="critique_explicit"
)

refiner_explicit = Agent(
    name="RefinerAgentExplicit",
    model=make_model(),
    instruction=(
        "Given a current summary and a critique, if critique == 'APPROVED' return the current summary. Otherwise rewrite "
        "the summary to incorporate the critique and return only the refined summary."
    ),
    output_key="refined_summary_explicit"
)

flashcard_explicit = Agent(
    name="FlashcardAgentExplicit",
    model=make_model(),
    instruction=(
        "Given a final summary, produce a JSON array of 10 flashcards: "
        "[{\"q\":\"\",\"a\":\"\",\"tags\":[],\"difficulty\":\"easy|medium|hard\"}, ...]. Return only JSON."
    ),
    output_key="flashcards_explicit"
)

logger.info("Agents defined (LLM-backed, definition agent uses google_search when available).")

# --- Helper: run the DefinitionAgent via ADK runner (returns list[dict])
# Replace the previous fetch_definitions_with_adk with this corrected version
async def fetch_definitions_with_adk(terms: List[str]):
    """
    Run definition_agent with the provided terms using InMemoryRunner.
    Returns: list of dicts [{"term":..., "definition":..., "source":...}, ...]
    Falls back to safe stubs when parsing fails or the agent returns invalid output.
    """
    if not isinstance(terms, list):
        terms = [terms]

    payload = json.dumps(terms, ensure_ascii=False)
    runner = InMemoryRunner(agent=definition_agent)

    # NOTE: run_debug does NOT accept a 'timeout' kwarg in this ADK version.
    events = await runner.run_debug(payload)

    # get last event text
    last = events[-1] if events else None
    raw = ""
    if last and last.content and last.content.parts:
        raw = last.content.parts[0].text or ""
    parsed = safe_parse_json_from_text(raw)

    if parsed is None:
        # fallback to per-term stubs (keeps shape consistent)
        return [lookup_definition_stub(t) for t in terms]

    # ensure parsed is a list of dicts (normalize)
    normalized = []
    if isinstance(parsed, list):
        for i, item in enumerate(parsed):
            if isinstance(item, dict):
                normalized.append({
                    "term": item.get("term") or (terms[i] if i < len(terms) else ""),
                    "definition": item.get("definition") or "No concise definition found.",
                    "source": item.get("source") or ""
                })
            elif isinstance(item, str):
                normalized.append({"term": item, "definition": "No concise definition found.", "source": ""})
            else:
                # unknown shape -> stub
                t = terms[i] if i < len(terms) else ""
                normalized.append(lookup_definition_stub(t))
    else:
        # unexpected shape -> return stubs
        return [lookup_definition_stub(t) for t in terms]

    return normalized




# Cell 09 - Orchestrator (async) - runs the whole pipeline; offline stubs if real LLM is unavailable
import nest_asyncio
nest_asyncio.apply()  # safe for Kaggle/Jupyter nested event loop

async def run_pipeline(transcript_id: str, use_offline_stubs: bool = True):
    t = load_transcript(transcript_id)
    text = t["text"]

    # 0) start
    log_agent(transcript_id, "orchestrator", "start_pipeline", {"use_offline_stubs": use_offline_stubs})

    # 1) Jargon extraction
    log_agent(transcript_id, "orchestrator", "start_jargon", {})
    if use_offline_stubs:
        # conservative heuristic extraction for offline mode
        tokens = set(re.findall(r"Term:\s*([A-Za-z0-9 _\-]+)", text))
        tokens |= set(re.findall(r"\b[A-Z][a-zA-Z]{6,}\b", text))
        terms = list(tokens)[:25]
        jargon_defs = [lookup_definition_stub(tok) for tok in terms]
        log_agent(transcript_id, "JargonResearchAgent", "completed_stub", {"n": len(jargon_defs)})
    else:
        # run the explicit ADK jargon agent to extract terms (expects valid JSON array of strings)
        try:
            runner = InMemoryRunner(agent=jargon_agent)
            events = await runner.run_debug(text)
            last = events[-1]
            raw = last.content.parts[0].text if last.content and last.content.parts else ""
            extracted = safe_parse_json_from_text(raw) or []
            # normalize to list of strings
            terms = [s for s in extracted if isinstance(s, str)]
            jargon_defs = [{"term": s, "definition": None, "source": None} for s in terms]
            log_agent(transcript_id, "JargonResearchAgent", "completed", {"n": len(jargon_defs)})
        except Exception as e:
            logger.exception("Jargon agent failed, falling back to stub extraction: %s", e)
            tokens = set(re.findall(r"Term:\s*([A-Za-z0-9 _\-]+)", text))
            tokens |= set(re.findall(r"\b[A-Z][a-zA-Z]{6,}\b", text))
            terms = list(tokens)[:25]
            jargon_defs = [lookup_definition_stub(tok) for tok in terms]
            log_agent(transcript_id, "JargonResearchAgent", "completed_with_fallback_stub", {"n": len(jargon_defs), "error": str(e)})

    # 1b) Definitions lookup (use DefinitionAgent when online) — with small DB cache & graceful fallback
    log_agent(transcript_id, "DefinitionAgent", "start_lookup", {"n_terms": len(terms)})

    # ensure cache table exists
    cur.execute('''
        CREATE TABLE IF NOT EXISTS definitions_cache (
            term TEXT PRIMARY KEY,
            def_json TEXT,
            updated_ts INTEGER
        )
    ''')
    conn.commit()

    defs = []  # final list of {"term","definition","source"}

    # helper: get from cache
    def get_cached_definition(term: str):
        row = cur.execute("SELECT def_json FROM definitions_cache WHERE term=?", (term,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    # helper: save to cache
    def save_cached_definition(term: str, obj: Dict[str,Any]):
        try:
            cur.execute("INSERT OR REPLACE INTO definitions_cache (term, def_json, updated_ts) VALUES (?, ?, ?)",
                        (term, json.dumps(obj, ensure_ascii=False), int(time.time())))
            conn.commit()
        except Exception as e:
            logger.warning("Failed to save definition cache for %s: %s", term, e)

    # decide whether to attempt live lookup: require not-offline and GOOGLE_API_KEY present
    can_try_live = (not use_offline_stubs) and (os.environ.get("GOOGLE_API_KEY") is not None)

    if not can_try_live:
        # offline path or key missing: use cache if available, else stubs
        cache_hits = 0
        for t in terms:
            cached = get_cached_definition(t)
            if cached:
                defs.append(cached)
                cache_hits += 1
            else:
                stub = lookup_definition_stub(t)
                defs.append(stub)
        log_agent(transcript_id, "DefinitionAgent", "completed_stub_or_cache", {"n": len(defs), "cache_hits": cache_hits})
    else:
        # attempt live ADK-driven lookup for all terms; if it fails, fall back per-term to cache/stub
        try:
            live_defs = await fetch_definitions_with_adk(terms)
            # normalize and merge with cache as needed
            live_count = 0
            cache_hits = 0
            for idx, t in enumerate(terms):
                item = None
                if isinstance(live_defs, list) and idx < len(live_defs) and isinstance(live_defs[idx], dict):
                    item = {
                        "term": live_defs[idx].get("term") or t,
                        "definition": live_defs[idx].get("definition") or "No concise definition found.",
                        "source": live_defs[idx].get("source") or ""
                    }
                else:
                    # fallback to try cache, then stub
                    cached = get_cached_definition(t)
                    if cached:
                        item = cached
                        cache_hits += 1
                    else:
                        item = lookup_definition_stub(t)
                defs.append(item)
                # save to cache if item looks like a live lookup (has a non-stub source)
                if item.get("source") and item.get("source") != "local-stub":
                    save_cached_definition(item["term"], item)
                    live_count += 1
            log_agent(transcript_id, "DefinitionAgent", "completed_live", {"n": len(defs), "live_hits": live_count, "cache_hits": cache_hits})
        except Exception as e:
            # ADK or network call failed — gracefully fall back to cache/stub per-term
            logger.exception("DefinitionAgent live lookup failed, falling back to cache/stubs: %s", e)
            cache_hits = 0
            for t in terms:
                cached = get_cached_definition(t)
                if cached:
                    defs.append(cached)
                    cache_hits += 1
                else:
                    defs.append(lookup_definition_stub(t))
            log_agent(transcript_id, "DefinitionAgent", "failed_fallback_stub", {"n": len(defs), "cache_hits": cache_hits, "error": str(e)})

    # persist jargon definitions artifact
    open(WORK_DIR / f"jargon_{transcript_id}.json","w",encoding="utf-8").write(json.dumps(defs, indent=2, ensure_ascii=False))
    log_agent(transcript_id, "orchestrator", "saved_jargon_defs", {"path": str(WORK_DIR / f"jargon_{transcript_id}.json")})

    # 2) Chunk summarization (sequential; could be parallelized)
    chunk_summaries = []
    for i, c in enumerate(chunk_text(text, chunk_size=1200)):
        log_agent(transcript_id, "ChunkSummarizer", "start_chunk", {"index": i})
        if use_offline_stubs:
            s = "- Key point A\n- Key point B\n"
        else:
            try:
                runner = InMemoryRunner(agent=chunk_summarizer)
                events = await runner.run_debug(c)
                last = events[-1]
                s = last.content.parts[0].text if last.content and last.content.parts else ""
            except Exception as e:
                logger.exception("Chunk summarizer failed for chunk %s: %s", i, e)
                s = "- Key point A\n- Key point B\n"
        chunk_summaries.append(s)
        log_agent(transcript_id, "ChunkSummarizer", "completed_chunk", {"index": i, "len": len(s.split())})

    # 3) Aggregation (use explicit agent)
    log_agent(transcript_id, "SummaryAggregator", "start", {"n_chunks": len(chunk_summaries)})
    combined_input = "\n\n".join(chunk_summaries)
    if use_offline_stubs:
        final_summary = "## Final Summary\n\n" + "\n\n".join(chunk_summaries)
    else:
        try:
            runner = InMemoryRunner(agent=aggregator_explicit)
            events = await runner.run_debug(combined_input)
            last = events[-1]
            final_summary = last.content.parts[0].text if last.content and last.content.parts else "\n\n".join(chunk_summaries)
        except Exception as e:
            logger.exception("Aggregator failed, falling back to concatenated chunks: %s", e)
            final_summary = "## Final Summary\n\n" + "\n\n".join(chunk_summaries)
    log_agent(transcript_id, "SummaryAggregator", "completed", {"len": len(final_summary.split())})

    # 4) Critique + refine loop (1 iteration max for safety)
    log_agent(transcript_id, "Critic", "start", {})
    if use_offline_stubs:
        critique = "APPROVED"
    else:
        try:
            runner = InMemoryRunner(agent=critic_explicit)
            events = await runner.run_debug(final_summary)
            last = events[-1]
            critique = last.content.parts[0].text.strip() if last.content and last.content.parts else ""
        except Exception as e:
            logger.exception("Critic agent failed: %s", e)
            critique = "APPROVED"
    log_agent(transcript_id, "Critic", "completed", {"critique": critique})

    if critique == "APPROVED":
        refined = final_summary
    else:
        if use_offline_stubs:
            refined = final_summary
        else:
            try:
                runner = InMemoryRunner(agent=refiner_explicit)
                events = await runner.run_debug(final_summary + "\n\nCritique:\n" + critique)
                last = events[-1]
                refined = last.content.parts[0].text if last.content and last.content.parts else final_summary
            except Exception as e:
                logger.exception("Refiner failed: %s", e)
                refined = final_summary
    log_agent(transcript_id, "Refiner", "completed", {"len": len(refined.split())})

    # 5) Flashcards
    log_agent(transcript_id, "Flashcard", "start", {})
    if use_offline_stubs:
        flashcards = [{"q":"What is amortized analysis?","a":"(stub) amortized analysis definition","tags":["algorithms"], "difficulty":"medium"}]
    else:
        try:
            runner = InMemoryRunner(agent=flashcard_explicit)
            events = await runner.run_debug(refined)
            last = events[-1]
            raw_fc = last.content.parts[0].text if last.content and last.content.parts else "[]"
            raw_fc_clean = raw_fc.replace("```json","").replace("```","").strip()
            try:
                flashcards = json.loads(raw_fc_clean)
            except Exception:
                flashcards = []
                logger.warning("Flashcard parse failed.")
        except Exception as e:
            logger.exception("Flashcard agent failed: %s", e)
            flashcards = []

    # Save artifacts to disk and DB
    open(WORK_DIR / f"summary_{transcript_id}.md","w",encoding="utf-8").write(refined)
    open(WORK_DIR / f"jargon_defs_{transcript_id}.json","w",encoding="utf-8").write(json.dumps(defs, indent=2, ensure_ascii=False))
    open(WORK_DIR / f"flashcards_{transcript_id}.json","w",encoding="utf-8").write(json.dumps(flashcards, indent=2, ensure_ascii=False))

    cur.execute("INSERT OR REPLACE INTO flashcards (id, transcript_id, card_json, created_ts) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), transcript_id, json.dumps(flashcards), int(time.time())))
    conn.commit()

    # simple evaluation metrics
    orig_tokens = len(text.split())
    summary_tokens = len(refined.split())
    compression_ratio = summary_tokens / max(1, orig_tokens)
    metric_inc("doc_processed", 1)
    metric_inc("avg_compression", compression_ratio)
    eval_result = {"original_tokens": orig_tokens, "summary_tokens": summary_tokens, "compression_ratio": compression_ratio}
    log_agent(transcript_id, "EvaluationAgent", "completed", eval_result)

    log_agent(transcript_id, "orchestrator", "completed_pipeline", {"eval": eval_result})
    return {"summary": refined, "jargon": defs, "flashcards": flashcards, "eval": eval_result}



# Cell 10 - wrapper to run async pipeline from top-level sync cell
use_offline = False   # set False to call real Gemini (requires GOOGLE_API_KEY & quota)

# If you're in a Jupyter/Kaggle kernel, nest_asyncio is already applied earlier
loop = asyncio.get_event_loop()
if loop.is_running():
    # schedule task and wait (works with nest_asyncio)
    task = asyncio.ensure_future(run_pipeline(transcript_id, use_offline_stubs=use_offline))
    loop.run_until_complete(task)
    res = task.result()
else:
    res = loop.run_until_complete(run_pipeline(transcript_id, use_offline_stubs=use_offline))

logger.info("Pipeline completed. Eval: %s", res["eval"])
print("Artifacts written to", WORK_DIR)



# Cell 11 - safe flashcard heuristics (same safe version we used in evaluation)
import numpy as np

def to_str(x):
    if isinstance(x, str): return x.lower()
    return str(x).lower()

def flashcard_quality(flashcards: List[Any], jargon_terms: List[Any]) -> Dict[str, Any]:
    if not isinstance(flashcards, list): return {"parse_ok": False, "count": 0}
    # normalize jargon
    jargon_clean=[]
    for j in jargon_terms:
        if isinstance(j, str):
            jargon_clean.append(j.lower())
        elif isinstance(j, dict):
            val = j.get("term") or j.get("name")
            if isinstance(val, str): jargon_clean.append(val.lower())
    count=len(flashcards)
    if count==0:
        return {"parse_ok": True,"count":0,"avg_q_len":0,"avg_a_len":0,"jargon_coverage":0,"unique_tags":0,"difficulty_counts":{"easy":0,"medium":0,"hard":0,"unknown":0}}
    q_lens=[]; a_lens=[]
    for fc in flashcards:
        q = fc.get("q") if isinstance(fc, dict) else ""
        a = fc.get("a") if isinstance(fc, dict) else ""
        q_lens.append(len(str(q).split()))
        a_lens.append(len(str(a).split()))
    avg_q = float(np.mean(q_lens))
    avg_a = float(np.mean(a_lens))
    found=0
    for term in jargon_clean:
        for fc in flashcards:
            q = to_str(fc.get("q",""))
            a = to_str(fc.get("a",""))
            if term in q or term in a:
                found += 1; break
    jargon_coverage = round(found / max(1, len(jargon_clean)), 3)
    tags=[]
    for fc in flashcards:
        t = fc.get("tags", [])
        if isinstance(t, list):
            for it in t: tags.append(str(it))
        else:
            tags.append(str(t))
    unique_tags = len(set(tags))
    diff_counts={"easy":0,"medium":0,"hard":0,"unknown":0}
    for fc in flashcards:
        d = to_str(fc.get("difficulty"))
        if d in diff_counts: diff_counts[d]+=1
        else: diff_counts["unknown"]+=1
    return {"parse_ok":True,"count":count,"avg_q_len":round(avg_q,2),"avg_a_len":round(avg_a,2),"jargon_coverage":jargon_coverage,"unique_tags":unique_tags,"difficulty_counts":diff_counts}

# quick run of heuristics (if artifacts exist)
summary_path = WORK_DIR / f"summary_{transcript_id}.md"
flash_path = WORK_DIR / f"flashcards_{transcript_id}.json"
jargon_path = WORK_DIR / f"jargon_{transcript_id}.json"

flashcards = json.loads(flash_path.read_text()) if flash_path.exists() else []
jargon = json.loads(jargon_path.read_text()) if jargon_path.exists() else []

fc_metrics = flashcard_quality(flashcards, jargon)
logger.info("Flashcard metrics: %s", fc_metrics)



# Cell 12 - minimal rouge check (use your evaluation notebook for full metrics)
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
orig = cur.execute("SELECT text FROM transcripts WHERE id=?", (transcript_id,)).fetchone()[0]
hyp = (WORK_DIR / f"summary_{transcript_id}.md").read_text() if (WORK_DIR / f"summary_{transcript_id}.md").exists() else ""
if hyp:
    scores = scorer.score(orig, hyp)
    print({k: round(v.fmeasure,4) for k,v in scores.items()})
else:
    print("No summary to score.")



%%bash
set -e
APPDIR=/kaggle/working/edu_agents_app
mkdir -p $APPDIR/templates
cat > $APPDIR/app.py <<'PY'
# app.py - EduAgents artifacts viewer (reads from /kaggle/working/edu_agents)
import json, sqlite3, os
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file, abort
import markdown as md

# === CONFIG: change WORK_DIR if your artifacts are elsewhere ===
WORK_DIR = Path("/kaggle/working/edu_agents")
DB_PATH = WORK_DIR / "agents_memory.db"

SUMMARY_GLOB = "summary_*.md"
JARGON_GLOB = "jargon*.json"
FLASH_GLOB = "flashcards*.json"

app = Flask(__name__, template_folder="templates", static_folder="static")

def find_file(glob_pattern):
    files = list(WORK_DIR.glob(glob_pattern))
    return files[0] if files else None

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def load_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def get_summary_path():
    return find_file(SUMMARY_GLOB)

def get_jargon_path():
    return find_file(JARGON_GLOB)

def get_flashcards_path():
    return find_file(FLASH_GLOB)

def list_files():
    return [{"name": p.name, "path": str(p), "size_kb": round(p.stat().st_size/1024,2)} for p in sorted(WORK_DIR.iterdir(), key=lambda x: x.name)]

def get_db_conn():
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def index():
    summary_path = get_summary_path()
    jargon_path = get_jargon_path()
    flash_path = get_flashcards_path()
    files = list_files()
    return render_template("index.html",
                           summary_exists=bool(summary_path),
                           jargon_exists=bool(jargon_path),
                           flash_exists=bool(flash_path),
                           files=files)

@app.route("/api/summary")
def api_summary():
    p = get_summary_path()
    if not p:
        return jsonify({"ok": False, "html": "", "raw": ""})
    raw = load_text(p)
    html = md.markdown(raw, extensions=["fenced_code","tables"])
    return jsonify({"ok": True, "html": html, "raw": raw, "filename": p.name})

@app.route("/api/jargon")
def api_jargon():
    p = get_jargon_path()
    if not p:
        return jsonify({"items": [], "total": 0})
    data = load_json(p)
    q = request.args.get("q", "").strip().lower()
    page = max(1, int(request.args.get("page", 1)))
    per_page = max(5, min(100, int(request.args.get("per_page", 20))))
    filtered = []
    for item in data:
        term = (item.get("term","") if isinstance(item, dict) else str(item)).lower()
        definition = (item.get("definition","") if isinstance(item, dict) else "").lower()
        if not q or q in term or q in definition:
            filtered.append(item)
    total = len(filtered)
    start = (page-1)*per_page
    return jsonify({"items": filtered[start:start+per_page], "total": total, "page": page, "per_page": per_page})

@app.route("/api/flashcards")
def api_flashcards():
    p = get_flashcards_path()
    if not p:
        return jsonify({"items": [], "total": 0})
    data = load_json(p)
    q = request.args.get("q","").strip().lower()
    tag = request.args.get("tag","").strip().lower()
    difficulty = request.args.get("difficulty","").strip().lower()
    page = max(1, int(request.args.get("page",1)))
    per_page = max(5, min(100, int(request.args.get("per_page",20))))
    filtered = []
    for item in data:
        qmatch = not q or (q in (item.get("q","").lower()) or q in (item.get("a","").lower()))
        tagmatch = not tag or tag in " ".join(item.get("tags",[])).lower()
        diffmatch = not difficulty or difficulty == item.get("difficulty","").lower()
        if qmatch and tagmatch and diffmatch:
            filtered.append(item)
    total = len(filtered)
    start = (page-1)*per_page
    return jsonify({"items": filtered[start:start+per_page], "total": total, "page": page, "per_page": per_page})

@app.route("/files")
def files():
    return jsonify(list_files())

@app.route("/download/<path:name>")
def download(name):
    p = WORK_DIR / name
    if not p.exists():
        abort(404)
    return send_file(str(p), as_attachment=True, download_name=p.name)

@app.route("/api/transcripts")
def api_transcripts():
    conn = get_db_conn()
    if not conn:
        return jsonify({"ok": False, "rows": []})
    rows = conn.execute("SELECT id, title, created_ts FROM transcripts ORDER BY created_ts DESC LIMIT 100").fetchall()
    items = [dict(r) for r in rows]
    return jsonify({"ok": True, "rows": items})

@app.route("/transcript/<tid>")
def view_transcript(tid):
    conn = get_db_conn()
    if not conn:
        abort(404)
    row = conn.execute("SELECT id,title,text,created_ts FROM transcripts WHERE id=?", (tid,)).fetchone()
    if not row:
        abort(404)
    return render_template("transcript.html", row=dict(row))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
PY

echo "wrote $APPDIR/app.py"



%%bash
set -e
APPDIR=/kaggle/working/edu_agents_app
cat > $APPDIR/templates/layout.html <<'HTML'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>Edu Agents - Artifacts UI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { padding-top: 70px; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace; }
      pre { white-space: pre-wrap; word-break: break-word; }
    </style>
  </head>
  <body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary fixed-top">
      <div class="container-fluid">
        <a class="navbar-brand" href="/">EduAgents Artifacts</a>
        <div class="d-flex">
          <a class="btn btn-outline-light btn-sm me-2" href="/files">Files JSON</a>
        </div>
      </div>
    </nav>
    <main class="container">
      {% block body %}{% endblock %}
    </main>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
HTML

cat > $APPDIR/templates/index.html <<'HTML'
{% extends "layout.html" %}
{% block body %}
<div class="row mb-3">
  <div class="col-md-8">
    <h3>Artifacts Viewer</h3>
    <p class="text-muted">Showing artifacts from <code class="mono">/kaggle/working/edu_agents</code></p>
  </div>
</div>

<ul class="nav nav-tabs mb-3" id="tabs">
  <li class="nav-item"><a class="nav-link active" href="#" data-tab="summary">Summary</a></li>
  <li class="nav-item"><a class="nav-link" href="#" data-tab="jargon">Jargon</a></li>
  <li class="nav-item"><a class="nav-link" href="#" data-tab="flashcards">Flashcards</a></li>
  <li class="nav-item"><a class="nav-link" href="#" data-tab="files">Files</a></li>
  <li class="nav-item"><a class="nav-link" href="#" data-tab="db">DB / Transcripts</a></li>
</ul>

<div id="tab-summary" class="tab-pane active">
  <div id="summary-container" class="card p-3">Loading summary…</div>
</div>

<div id="tab-jargon" class="tab-pane" style="display:none;">
  <div class="mb-2">
    <input id="jsearch" class="form-control" placeholder="Search jargon (term or definition)">
  </div>
  <div id="jargon-list" class="row gy-3"></div>
  <nav><ul id="jargon-pages" class="pagination mt-3"></ul></nav>
</div>

<div id="tab-flashcards" class="tab-pane" style="display:none;">
  <div class="row mb-2">
    <div class="col-md-5"><input id="fsearch" class="form-control" placeholder="Search question or answer"></div>
    <div class="col-md-3"><input id="ftag" class="form-control" placeholder="Filter tag"></div>
    <div class="col-md-2">
      <select id="fdiff" class="form-control">
        <option value="">All difficulty</option>
        <option value="easy">easy</option>
        <option value="medium">medium</option>
        <option value="hard">hard</option>
      </select>
    </div>
    <div class="col-md-2 text-end"><button id="fdo" class="btn btn-primary">Filter</button></div>
  </div>
  <div id="flashcards-list" class="row gy-3"></div>
  <nav><ul id="flash-pages" class="pagination mt-3"></ul></nav>
</div>

<div id="tab-files" class="tab-pane" style="display:none;">
  <h5>Files in work dir</h5>
  <div id="files-list"></div>
</div>

<div id="tab-db" class="tab-pane" style="display:none;">
  <h5>Transcripts table (DB)</h5>
  <div id="transcripts-list" class="mt-2"></div>
</div>

<script>
const perPage = 12;
// tab switching
document.querySelectorAll('#tabs a').forEach(a=>{
  a.addEventListener('click', (ev)=>{
    ev.preventDefault();
    document.querySelectorAll('#tabs a').forEach(x=>x.classList.remove('active'));
    a.classList.add('active');
    const tab = a.dataset.tab;
    document.querySelectorAll('.tab-pane').forEach(p=>p.style.display='none');
    document.getElementById('tab-'+tab).style.display='block';
    if(tab==='summary') loadSummary();
    if(tab==='jargon') loadJargon();
    if(tab==='flashcards') loadFlashcards();
    if(tab==='files') loadFiles();
    if(tab==='db') loadTranscripts();
  });
});

// Summary
async function loadSummary() {
  const c = document.getElementById('summary-container');
  c.innerHTML = "Loading …";
  const r = await fetch('/api/summary'); const j = await r.json();
  if(!j.ok){ c.innerHTML = "<div class='alert alert-secondary'>No summary file found</div>"; return; }
  c.innerHTML = `<div>${j.html}</div><hr/><small class="text-muted">Filename: ${j.filename}</small>`;
}

// Jargon
let jpage = 1;
async function loadJargon(){
  const q = document.getElementById('jsearch').value||"";
  const r = await fetch(`/api/jargon?q=${encodeURIComponent(q)}&page=${jpage}&per_page=12`);
  const j = await r.json();
  const container = document.getElementById('jargon-list');
  container.innerHTML = "";
  if(j.items.length===0){ container.innerHTML = "<div class='alert alert-secondary'>No jargon found</div>"; renderJargonPages(0,1); return; }
  for(const it of j.items){
    const col = document.createElement('div'); col.className='col-md-6';
    col.innerHTML = `<div class="card"><div class="card-body"><h5>${escape(it.term)}</h5><p>${escape(it.definition)}</p><p class="mb-0"><small class="text-muted">Source: ${escape(it.source||'N/A')}</small></p></div></div>`;
    container.appendChild(col);
  }
  renderJargonPages(j.total, j.page);
}
function renderJargonPages(total, page){
  const pages = Math.max(1, Math.ceil(total/12));
  const ul = document.getElementById('jargon-pages'); ul.innerHTML="";
  for(let i=1;i<=pages;i++){
    const li = document.createElement('li'); li.className=`page-item ${i===page?'active':''}`;
    li.innerHTML = `<a href="#" class="page-link">${i}</a>`;
    li.querySelector('a').addEventListener('click', (ev)=>{ ev.preventDefault(); jpage=i; loadJargon();});
    ul.appendChild(li);
  }
}
document.getElementById('jsearch').addEventListener('keyup', (ev)=>{ if(ev.key==='Enter'){ jpage=1; loadJargon(); }});

// Flashcards
let fpage=1;
async function loadFlashcards(){
  const q = document.getElementById('fsearch').value||"";
  const tag = document.getElementById('ftag').value||"";
  const diff = document.getElementById('fdiff').value||"";
  const r = await fetch(`/api/flashcards?q=${encodeURIComponent(q)}&tag=${encodeURIComponent(tag)}&difficulty=${encodeURIComponent(diff)}&page=${fpage}&per_page=12`);
  const j = await r.json();
  const container = document.getElementById('flashcards-list'); container.innerHTML="";
  if(j.items.length===0){ container.innerHTML="<div class='alert alert-secondary'>No flashcards found</div>"; renderFlashPages(0,1); return; }
  for(const it of j.items){
    const col = document.createElement('div'); col.className='col-md-6';
    col.innerHTML = `<div class="card"><div class="card-body"><h5>${escape(it.q)}</h5><p>${escape(it.a)}</p><p class="mb-0"><small class="text-muted">Tags: ${escape((it.tags||[]).join(', '))} • Difficulty: ${escape(it.difficulty||'')}</small></p></div></div>`;
    container.appendChild(col);
  }
  renderFlashPages(j.total, j.page);
}
function renderFlashPages(total, page){
  const pages = Math.max(1, Math.ceil(total/12));
  const ul = document.getElementById('flash-pages'); ul.innerHTML="";
  for(let i=1;i<=pages;i++){
    const li = document.createElement('li'); li.className=`page-item ${i===page?'active':''}`;
    li.innerHTML = `<a href="#" class="page-link">${i}</a>`;
    li.querySelector('a').addEventListener('click', (ev)=>{ ev.preventDefault(); fpage=i; loadFlashcards();});
    ul.appendChild(li);
  }
}
document.getElementById('fdo').addEventListener('click', ()=>{ fpage=1; loadFlashcards();});

// Files
async function loadFiles(){
  const r = await fetch('/files'); const j = await r.json();
  const c = document.getElementById('files-list'); c.innerHTML="";
  for(const f of j){
    const el = document.createElement('div'); el.className='mb-2';
    el.innerHTML = `<div class="d-flex justify-content-between align-items-center"><div><strong>${escape(f.name)}</strong> <small class="text-muted">(${f.size_kb} KB)</small></div><div><a class="btn btn-sm btn-outline-primary me-2" href="/download/${encodeURIComponent(f.name)}">Download</a></div></div>`;
    c.appendChild(el);
  }
}

// DB transcripts
async function loadTranscripts(){
  const r = await fetch('/api/transcripts'); const j = await r.json();
  const c = document.getElementById('transcripts-list'); c.innerHTML="";
  if(!j.ok){ c.innerHTML="<div class='alert alert-secondary'>No DB found</div>"; return;}
  for(const row of j.rows){
    const d = new Date(row.created_ts*1000);
    const el = document.createElement('div'); el.className='mb-2';
    el.innerHTML = `<div><a href="/transcript/${row.id}">${escape(row.title||row.id)}</a> <small class="text-muted">${d.toLocaleString()}</small></div>`;
    c.appendChild(el);
  }
}

// escape helper
function escape(s){ if(!s) return ""; return s.toString().replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;"); }

// initial load
loadSummary();
</script>
HTML

cat > $APPDIR/templates/transcript.html <<'HTML'
{% extends "layout.html" %}
{% block body %}
  <h3>{{ row.title }}</h3>
  <pre>{{ row.text }}</pre>
{% endblock %}
HTML

echo "wrote templates to $APPDIR/templates"



from pathlib import Path
p = Path("/kaggle/working/edu_agents")
print("Files in /kaggle/working/edu_agents:")
for f in sorted(p.iterdir()):
    print(f.name, f.stat().st_size, "bytes")



ls -alt



ls -alt edu_agents_app



# === Safe, self-contained UI generator (no .format() problems) ===
import json
from pathlib import Path
from IPython.display import HTML, display

WORK_DIR = Path("/kaggle/working/edu_agents")
APP_DIR  = Path("/kaggle/working/edu_agents_app")
OUT_HTML = APP_DIR / "output.html"
APP_DIR.mkdir(parents=True, exist_ok=True)

# find artifacts
def find_first(*patterns):
    for pat in patterns:
        files = sorted(WORK_DIR.glob(pat))
        if files:
            return files[0]
    return None

summary_p = find_first("summary_*.md", "*.md")
jargon_p  = find_first("jargon*.json")
flash_p   = find_first("flashcards*.json", "flashcards_*.json")

summary_text = summary_p.read_text(encoding="utf-8") if summary_p else ""
jargon_data  = json.loads(jargon_p.read_text(encoding="utf-8")) if jargon_p else []
flashcards   = json.loads(flash_p.read_text(encoding="utf-8")) if flash_p else []

summary_html = summary_text.replace("\n", "<br/>")
file_list_html = "".join(
    f"<li><code>{p.name}</code> — {round(p.stat().st_size/1024,2)} KB</li>"
    for p in sorted(WORK_DIR.iterdir())
)

jargon_json = json.dumps(jargon_data, ensure_ascii=False)
flash_json  = json.dumps(flashcards, ensure_ascii=False)

# HTML template with unique tokens (safe)
template = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Edu_Agents Viewer</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
<style>
body { padding:18px; background:#f8f9fa; }
pre  { white-space:pre-wrap; }
.card { border-radius:10px; }
</style>
</head>
<body>
<div class="container">
  <h3>EduAgents Artifacts Viewer</h3>
  <p><small class="text-muted">Work dir: __WORKDIR__</small></p>

  <div class="row">
    <div class="col-md-6">
      <h5>Summary</h5>
      __SUMMARY_BLOCK__
    </div>

    <div class="col-md-6">
      <h5>Files</h5>
      <ul>__FILE_LIST__</ul>
    </div>
  </div>

  <hr/>

  <div class="row">
    <div class="col-md-6">
      <h5>Jargon / Glossary</h5>
      <div class="input-group mb-2">
        <input id="j-q" class="form-control" placeholder="Search term or definition">
        <button class="btn btn-primary" onclick="loadJargon(1)">Search</button>
      </div>
      <div id="jargon-list"></div>
      <nav><ul id="j-pages" class="pagination mt-2"></ul></nav>
    </div>

    <div class="col-md-6">
      <h5>Flashcards</h5>
      <div class="input-group mb-2">
        <input id="f-q" class="form-control" placeholder="Search question/answer">
        <button class="btn btn-primary" onclick="loadFlash(1)">Search</button>
      </div>
      <div id="flash-list"></div>
      <nav><ul id="f-pages" class="pagination mt-2"></ul></nav>
    </div>
  </div>
</div>

<script>
// Data
const JARGON = __JARGON__;
const FLASH  = __FLASH__;
const PER_PAGE = 8;

function escapeHtml(s){
    if(!s) return "";
    return s.toString()
            .replace(/&/g,"&amp;")
            .replace(/</g,"&lt;")
            .replace(/>/g,"&gt;");
}

function renderPages(id,total,cur,handler){
    const pages = Math.max(1, Math.ceil(total/PER_PAGE));
    const ul = document.getElementById(id);
    ul.innerHTML = '';
    for(let i=1;i<=pages;i++){
        const li = document.createElement('li');
        li.className = 'page-item ' + (i===cur ? 'active' : '');
        li.innerHTML = '<a class="page-link" href="#">' + i + '</a>';
        li.firstChild.onclick = (e)=>{ e.preventDefault(); handler(i); };
        ul.appendChild(li);
    }
}

function renderJargon(items,page){
    const c=document.getElementById('jargon-list'); c.innerHTML='';
    if(items.length===0){
        c.innerHTML='<div class="alert alert-secondary">No jargon entries found</div>';
        renderPages('j-pages',0,1,loadJargon); return;
    }
    const start=(page-1)*PER_PAGE;
    const chunk=items.slice(start,start+PER_PAGE);
    for(const it of chunk){
        const term=escapeHtml(it.term||'');
        const def =escapeHtml(it.definition||'');
        const src =escapeHtml(it.source||'');
        c.innerHTML += `<div class="mb-2"><div class="card p-2"><b>${term}</b><div>${def}</div><div class="mt-1"><small class="text-muted">Source: ${src}</small></div></div></div>`;
    }
    renderPages('j-pages', items.length, page, loadJargon);
}

function loadJargon(page){
    page = page || 1;
    const q=(document.getElementById('j-q').value||'').toLowerCase().trim();
    const items = JARGON.filter(it=>{
        const t=(it.term||'').toLowerCase();
        const d=(it.definition||'').toLowerCase();
        if(!q) return true;
        return t.includes(q) || d.includes(q);
    });
    renderJargon(items,page);
}

function renderFlash(items,page){
    const c=document.getElementById('flash-list'); c.innerHTML='';
    if(items.length===0){
        c.innerHTML='<div class="alert alert-secondary">No flashcards found</div>';
        renderPages('f-pages',0,1,loadFlash); return;
    }
    const start=(page-1)*PER_PAGE;
    const chunk=items.slice(start,start+PER_PAGE);
    for(const it of chunk){
        const q=escapeHtml(it.q||'');
        const a=escapeHtml(it.a||'');
        const tags=(it.tags||[]).join(', ');
        const diff=escapeHtml(it.difficulty||'');
        c.innerHTML += `<div class="mb-2"><div class="card p-2"><h6>${q}</h6><div>${a}</div><div class="mt-1"><small class="text-muted">Tags: ${tags} • Difficulty: ${diff}</small></div></div></div>`;
    }
    renderPages('f-pages', items.length, page, loadFlash);
}

function loadFlash(page){
    page = page || 1;
    const q=(document.getElementById('f-q').value||'').toLowerCase().trim();
    const items = FLASH.filter(it=>{
        const Q=(it.q||'').toLowerCase();
        const A=(it.a||'').toLowerCase();
        const TAGS=(it.tags||[]).join(' ').toLowerCase();
        if(!q) return true;
        return Q.includes(q) || A.includes(q) || TAGS.includes(q);
    });
    renderFlash(items,page);
}

document.addEventListener('DOMContentLoaded', ()=>{
    loadJargon(1);
    loadFlash(1);
});
</script>
</body>
</html>
"""

# Build summary block
if summary_text:
    summary_block = "<div class='card p-3'>" + summary_html + "</div>"
else:
    summary_block = "<div class='alert alert-secondary'>No summary available</div>"

# Replace tokens safely
html = template.replace("__WORKDIR__", str(WORK_DIR))
html = html.replace("__SUMMARY_BLOCK__", summary_block)
html = html.replace("__FILE_LIST__", file_list_html)
html = html.replace("__JARGON__", jargon_json)
html = html.replace("__FLASH__", flash_json)

# Save and show link
OUT_HTML.write_text(html, encoding="utf-8")
print("Saved UI to:", OUT_HTML)
display(HTML(f"<a href='/files{OUT_HTML}' target='_blank'>Open EduAgents UI (new tab)</a>"))



# 3) Render HTML content directly (guaranteed to show if file exists)
from IPython.display import HTML, display
p = Path("/kaggle/working/edu_agents_app/output.html")
if p.exists():
    display(HTML(p.read_text(encoding="utf-8")))
else:
    print("output.html not found at", p)





