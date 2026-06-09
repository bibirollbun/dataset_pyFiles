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
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )




import pandas as pd
import numpy as np
import json, os, time
from scipy import stats
from datetime import datetime

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


DATA_PATH = "/kaggle/input/sample/kpi_sample.csv"
SESSION_PATH = "/kaggle/working/session.json"
LOG_PATH = "/kaggle/working/logs.jsonl"
MAX_TOP_SLICES = 3
Z_THRESHOLD = 2.5

print("âœ… Configuration variables initialized.")


#Tool: load_and_summarize_kpis

def load_and_summarize_kpis(data_path, metric='revenue', date_col='date', timeframe=None, filters=None, time_grain='W'):

    """
        Loads CSV, filters, aggregates by period, computes delta/pct_change,
        z-score anomaly detection, and top slices (by product/region heuristics).
        Returns dict with agg, latest, prev, delta, pct_change, anomalies,
        top_slices, is_anomalous, anomaly_score.
    """
    df = pd.read_csv(data_path, parse_dates=[date_col], low_memory=False)

    #Filtering
    if filters:
        for k, v in filters.items():
            if k not in df.columns:
                continue
            df = df[df[k] == v]

    #Time Filtering
    if timeframe:
        start, end = pd.to_datetime(timeframe[0]), pd.to_datetime(timeframe[1])
        df = df[(df[date_col] >= start) & (df[date_col] <= end)]

    #Handle metric casing issues
    if metric not in df.columns:
        cand = [c for c in df.columns if c.lower() == metric.lower()]
        if cand:
            metric = cand[0]

    #Period Grouping
    df = df.copy()
    if time_grain == 'W':
        df['period'] = df[date_col].dt.to_period('W').apply(lambda p: p.start_time)
    elif time_grain == 'M':
        df['period'] = df[date_col].dt.to_period('M').apply(lambda p: p.start_time)
    else:
        df['period'] = df[date_col].dt.date

    agg = df.groupby('period', as_index=False)[metric].sum().sort_values('period')

    # Latest/rev/delta
    latest = agg.iloc[-1].to_dict() if len(agg) >= 1 else None
    prev = agg.iloc[-2].to_dict() if len(agg) >= 2 else None
    delta = latest[metric] - prev[metric] if (latest and prev) else None
    pct_change = (delta / prev[metric] * 100) if (prev and prev[metric] != 0) else None

    #z-score anomaly detection
    anomaly_score = None
    anomalies = []
    is_anomalous = False
    if len(agg) >= 3:
        vals = agg[metric].astype(float)
        z = (vals - vals.mean()) / (vals.std(ddof=0) + 1e-9)
        agg['z'] = z
        anomalies = agg[agg['z'].abs() >= Z_THRESHOLD].to_dict(orient='records')
        anomaly_score = float(z.iloc[-1])
        is_anomalous = abs(anomaly_score) >= Z_THRESHOLD

    #Top slices
    top_slices =[]
    for col in ['product', 'region', 'channel']:
        if col in df.columns:
            top = df.groupby([col])[metric].sum().reset_index().sort_values(metric, ascending=False).head(MAX_TOP_SLICES)
            for _, r in top.iterrows():
                top_slices.append({"slice_filter": {col: r[col]}, "value": float(r[metric]), "slice_col": col})

    #Deduplicate slices
    seen = set()
    uniq_slices = []
    for s in top_slices:
        key = tuple(sorted(s["slice_filter"].items()))
        if key not in seen:
            seen.add(key)
            uniq_slices.append(s)

    return{
        "agg": agg,
        "latest": latest,
        "prev": prev,
        "delta": delta,
        "pct_change": pct_change,
        "anomalies": anomalies,
        "anomaly_score": anomaly_score,
        "is_anomalous": bool(is_anomalous),
        "top_slices": uniq_slices[:MAX_TOP_SLICES]
    }

print("âœ… Custom tool 'load_and_summarize_kpis' loaded successfully.")



# Session & Logging Helpers

def load_session(session_id='default'):
    if os.path.exists(SESSION_PATH):
        try:
            return json.load(open(SESSION_PATH))
        except:
            return {"session_id": session_id, "prior_anomalies": [], "thumbs_up": 0, "thumbs_down": 0}
    return {"session_id": session_id, "prior_anomalies": [], "thumbs_up": 0, "thumbs_down": 0}

def save_session(session):
    json.dump(session, open(SESSION_PATH, 'w'), indent=2)

def append_log(record):
    record['ts'] = datetime.utcnow().isoformat() + "Z"
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + "\n")

print("âœ… Session and logging helpers loaded successfully.")


#Triage Loop

import time

def triage_loop(metric, timeframe, base_filters=None, data_path=DATA_PATH, max_depth=3, max_candidates=5):
    base_filters = base_filters or {}
    candidates = []
    stack = [{"filters": base_filters, "depth": 0}]
    visited = set()

    while stack:
        item = stack.pop(0)
        filters = item["filters"]
        depth = item["depth"]

        # Avoid repeating filters
        key = tuple(sorted(filters.items()))
        if key in visited:
            continue
        visited.add(key)

        start = time.time()
        res = load_and_summarize_kpis(data_path, metric=metric, timeframe=timeframe, filters=filters)
        elapsed = time.time() - start

        # Log the step
        append_log({
            "session_id": "default",
            "call": "load_and_summarize_kpis",
            "metric": metric,
            "filters": filters,
            "depth": depth,
            "is_anomalous": res["is_anomalous"],
            "anomaly_score": res["anomaly_score"],
            "runtime_s": elapsed,
            "top_slices_count": len(res["top_slices"])
        })

        # Collect anomalous results
        if res["is_anomalous"]:
            candidates.append({
                "filters": filters,
                "detail": res,
                "depth": depth,
                "score": abs(res["anomaly_score"] or 0)
            })

            # Add children if not hitting max depth
            if depth < max_depth - 1:
                for s in res["top_slices"]:
                    new_filters = dict(filters)
                    new_filters.update(s["slice_filter"])
                    stack.append({"filters": new_filters, "depth": depth + 1})

        if len(candidates) >= max_candidates:
            break

    return sorted(candidates, key=lambda x: x["score"], reverse=True)

print("âœ… Triage loop loaded successfully.")


# Demo Run
print("ğŸš€ Starting demo run...")

print("Checking data file:", os.path.exists(DATA_PATH))

# Auto-detect timeframe
demo_timeframe = None
try:
    df_tmp = pd.read_csv(DATA_PATH, parse_dates=[0], nrows=1000)
    date_col_guess = df_tmp.columns[0]
    demo_timeframe = [
        df_tmp[date_col_guess].min().strftime("%Y-%m-%d"),
        df_tmp[date_col_guess].max().strftime("%Y-%m-%d")
    ]
    print("ğŸ“… Auto timeframe detected:", demo_timeframe)
except:
    print("âš ï¸� Could not auto-detect timeframe. Using last 30 days.")
    demo_timeframe = [
        (datetime.utcnow() - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        datetime.utcnow().strftime("%Y-%m-%d")
    ]

demo_metric = "revenue"
print(f"ğŸ”� Running summarization for metric: {demo_metric}")

summ = load_and_summarize_kpis(DATA_PATH, metric=demo_metric, timeframe=demo_timeframe, filters={}, time_grain='W')

print("ğŸ“Š Summary keys:", list(summ.keys()))
print("ğŸ”� Is anomalous?", summ['is_anomalous'], "| Score:", summ['anomaly_score'])
print("âœ¨ Top slices:", summ['top_slices'])

print("\nğŸ”� Running triage loop (max_depth=3)...")
cands = triage_loop(metric=demo_metric, timeframe=demo_timeframe, base_filters={}, data_path=DATA_PATH)

print(f"ğŸ�� Triage complete. Candidates found: {len(cands)}")

for i, c in enumerate(cands[:5]):
    print(f"\nğŸ”¸ Candidate {i+1} | depth={c['depth']} | score={c['score']}")
    print("   Filters:", c['filters'])
    if c['detail'] and c['detail'].get('pct_change') is not None:
        print("   Change:", c['detail']['delta'], "(pct:", c['detail']['pct_change'], ")")

print("\nâœ… Demo run completed successfully.")


# === Final Gemini wrapper (modified, ready for Insights Agent) ===
import os
import json
import time
import traceback

print("ğŸ”§ Initializing final Gemini wrapper (modified)...")

# sentinel prefix used for explicit mock responses
MOCK_PREFIX = "__MOCK_JSON_RESPONSE__:"

# sensible default model for your environment (change if you later prefer another)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")
GEMINI_AVAILABLE = False
GEMINI_CLIENT = None
INIT_ERROR = None

# Try to initialize the genai client if key present
if "GOOGLE_API_KEY" not in os.environ:
    print("âš ï¸� GOOGLE_API_KEY not found in environment. Run secret-loader cell first.")
else:
    try:
        from google import genai
        # local import of types is OK; we also import when needed later
        api_key = os.environ["GOOGLE_API_KEY"]
        print("ğŸ”� GOOGLE_API_KEY found in env. Creating genai client...")
        GEMINI_CLIENT = genai.Client(api_key=api_key)
        GEMINI_AVAILABLE = True
        print(f"âœ… Gemini client initialized. Default model: {GEMINI_MODEL}")
    except Exception as e:
        INIT_ERROR = e
        GEMINI_AVAILABLE = False
        print("â�— Gemini client initialization FAILED.")
        traceback.print_exc()
        print("â�¡ï¸� Wrapper will return mock responses until init succeeds.")


def _extract_text_from_response(resp):
    """
    Extract text from a variety of SDK response shapes.

    - Returns a string (the textual model output) or None if nothing extractable.
    - Tries to dig into nested shapes like resp.output[0].content[0].text or resp.candidates[*].content[*].text
    """
    if resp is None:
        return None

    # Fast path: if it's already a string
    try:
        if isinstance(resp, str) and resp.strip():
            return resp
    except Exception:
        pass

    # If it's a dict-like object, check common keys first
    try:
        if isinstance(resp, dict):
            for k in ("text", "output", "response", "candidates", "content", "result"):
                if k in resp and resp[k]:
                    val = resp[k]
                    if isinstance(val, str):
                        return val
                    # list-like value
                    if isinstance(val, (list, tuple)) and len(val) > 0:
                        first = val[0]
                        if isinstance(first, str):
                            return first
                        if isinstance(first, dict):
                            # try common nested patterns
                            if "text" in first and first["text"]:
                                return first["text"]
                            if "content" in first:
                                c = first["content"]
                                if isinstance(c, str):
                                    return c
                                if isinstance(c, (list, tuple)) and len(c) > 0:
                                    fc = c[0]
                                    if isinstance(fc, dict) and "text" in fc and fc["text"]:
                                        return fc["text"]
                                    if isinstance(fc, str):
                                        return fc
                            try:
                                return json.dumps(first)
                            except Exception:
                                return str(first)
                    # dict-like
                    if isinstance(val, dict):
                        try:
                            return json.dumps(val)
                        except Exception:
                            return str(val)
    except Exception:
        pass

    # If object has attributes, inspect them
    for attr in ("text", "output", "result", "response", "candidates", "content"):
        try:
            if hasattr(resp, attr):
                val = getattr(resp, attr)
                # if string
                if isinstance(val, str) and val.strip():
                    return val
                # if list-like, try to extract nested text
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    first = val[0]
                    # dict-like first entry
                    if isinstance(first, dict):
                        if "text" in first and first["text"]:
                            return first["text"]
                        if "content" in first:
                            c = first["content"]
                            if isinstance(c, str) and c.strip():
                                return c
                            if isinstance(c, (list, tuple)) and len(c) > 0:
                                fc = c[0]
                                if isinstance(fc, dict) and "text" in fc and fc["text"]:
                                    return fc["text"]
                                if isinstance(fc, str) and fc.strip():
                                    return fc
                        try:
                            return json.dumps(first)
                        except Exception:
                            return str(first)
                    # if string first element
                    if isinstance(first, str) and first.strip():
                        return first
                # if dict-like attribute
                if isinstance(val, dict):
                    try:
                        return json.dumps(val)
                    except Exception:
                        return str(val)
        except Exception:
            continue

    # fallback to str()
    try:
        return str(resp)
    except Exception:
        return None


def gemini_generate(prompt, use_mock=False, max_output_tokens=512, retries=0, retry_delay=0.6, mock_kind="generic", return_raw=False):
    """
    Robust wrapper to call Gemini (or return mock).

    Returns:
      - by default: text (str) extracted from the SDK response (or MOCK_PREFIX + json)
      - if return_raw=True: returns tuple (text, raw_response_object)

    Behavior:
      - If use_mock=True, returns a deterministic mock prefixed with MOCK_PREFIX
      - If not initialized, returns a diagnostic mock prefixed with MOCK_PREFIX
      - Attempts several SDK call patterns with safe extraction
      - On failure, returns a diagnostic mock with failure summary
    """
    # explicit mock branch
    if use_mock:
        print("ğŸ¤– gemini_generate: explicit mock requested.")
        if mock_kind == "insights":
            mock_obj = {"title": "Mock Insight", "narrative": "This is a mock insight.", "follow_ups": []}
        elif mock_kind == "interpreter":
            mock_obj = {
                "metric": "revenue",
                "timeframe": ["2025-01-01", "2025-01-31"],
                "time_grain": "week",
                "filters": {"region": "APAC"},
                "intent": "anomaly",
            }
        else:
            mock_obj = {"mock": True}
        out = MOCK_PREFIX + " " + json.dumps(mock_obj)
        print("ğŸ”� Returning mock output.")
        return (out, None) if return_raw else out

    # not initialized -> diagnostic mock
    if not GEMINI_AVAILABLE or GEMINI_CLIENT is None:
        print("âš ï¸� Gemini client not initialized. Returning diagnostic mock.")
        mock = {
            "title": "Mock fallback",
            "narrative": "Gemini client not available. Re-run init with API key.",
            "init_error": str(INIT_ERROR)[:300],
        }
        out = MOCK_PREFIX + " " + json.dumps(mock)
        return (out, None) if return_raw else out

    client = GEMINI_CLIENT
    model = GEMINI_MODEL
    last_exc = None
    raw_resp = None

    for attempt in range(retries + 1):
        try:
            print(f"ğŸ“¡ gemini_generate attempt {attempt+1}/{retries+1} (model={model}, max_tokens={max_output_tokens})")

            # Preferred: client.models.generate_content(model=..., contents=..., config=types.GenerateContentConfig(...))
            try:
                if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                    print("  - Trying: client.models.generate_content(model=..., contents=..., config=...)")
                    from google.genai import types as _types  # local import ensures availability
                    cfg = _types.GenerateContentConfig(max_output_tokens=max_output_tokens)
                    resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
                    raw_resp = resp
                    out = _extract_text_from_response(resp)
                    print("  âœ… Pattern: generate_content(model,contents,config) succeeded.")
                    return (out, raw_resp) if return_raw else (out if out is not None else str(resp))
            except Exception as e:
                print("  - Pattern generate_content error:", repr(e))
                last_exc = e

            # Fallback: client.models.generate(...)
            try:
                if hasattr(client, "models") and hasattr(client.models, "generate"):
                    print("  - Trying: client.models.generate(model=..., messages=...)")
                    resp = client.models.generate(model=model, messages=[{"role": "user", "content": prompt}], max_output_tokens=max_output_tokens)
                    raw_resp = resp
                    out = _extract_text_from_response(resp)
                    print("  âœ… Pattern: models.generate succeeded.")
                    return (out, raw_resp) if return_raw else (out if out is not None else str(resp))
            except Exception as e:
                print("  - Pattern models.generate error:", repr(e))
                last_exc = e

            # Fallback: client.generate(...)
            try:
                if hasattr(client, "generate"):
                    print("  - Trying: client.generate(model=..., prompt=...)")
                    resp = client.generate(model=model, prompt=prompt, max_output_tokens=max_output_tokens)
                    raw_resp = resp
                    out = _extract_text_from_response(resp)
                    print("  âœ… Pattern: client.generate succeeded.")
                    return (out, raw_resp) if return_raw else (out if out is not None else str(resp))
            except Exception as e:
                print("  - Pattern client.generate error:", repr(e))
                last_exc = e

            # Fallback: client.responses.create(...)
            try:
                if hasattr(client, "responses") and hasattr(client.responses, "create"):
                    print("  - Trying: client.responses.create(model=..., input=...)")
                    resp = client.responses.create(model=model, input=prompt, max_output_tokens=max_output_tokens)
                    raw_resp = resp
                    out = _extract_text_from_response(resp)
                    print("  âœ… Pattern: responses.create succeeded.")
                    return (out, raw_resp) if return_raw else (out if out is not None else str(resp))
            except Exception as e:
                print("  - Pattern responses.create error:", repr(e))
                last_exc = e

            # none matched
            raise last_exc or RuntimeError("No compatible SDK call pattern matched.")

        except Exception as e:
            last_exc = e
            print(f"â�— Attempt {attempt+1} failed with error:", repr(e))
            traceback.print_exc()
            if attempt < retries:
                print(f"â†» Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            else:
                print("â�¡ï¸� All attempts exhausted. Returning mock fallback with error summary.")
                fallback = {"title": "Mock fallback", "narrative": f"Gemini call failed: {str(last_exc)[:300]}", "follow_ups": []}
                out = MOCK_PREFIX + " " + json.dumps(fallback)
                return (out, raw_resp) if return_raw else out

# final diagnostics
print("ğŸ”� gemini_generate defined:", "gemini_generate" in globals())
print("ğŸ”� GEMINI_AVAILABLE flag:", GEMINI_AVAILABLE)
print("ğŸ”� GEMINI_CLIENT present:", GEMINI_CLIENT is not None)
if not GEMINI_AVAILABLE:
    print("â„¹ï¸� INIT_ERROR (summary):", repr(INIT_ERROR)[:300])
print("âœ… Gemini wrapper ready. Use gemini_generate(prompt, use_mock=False, return_raw=False) from Insights Agent.")



# === Gemini Query Interpreter Agent (improved extractor + diagnostics) ===
import json
import re
import traceback
import pandas as pd
import os
import textwrap
import time

print("Loading Gemini Query Interpreter Agent (improved)...")

# === Mock sentinel alignment ===
# Try to pick up mock prefix from wrapper if available in the same runtime; otherwise fall back.
LEGACY_MOCK_PREFIX = "MOCK_JSON_RESPONSE:"
WRAPPER_MOCK_PREFIX = globals().get("MOCK_PREFIX", "__MOCK_JSON_RESPONSE__:")

# System prompt for the interpreter
INTERPRETER_SYSTEM_PROMPT = """
You are an analyst assistant whose ONLY job is to parse a single user KPI query into strictly valid JSON.
Output MUST be valid JSON and nothing else.

Required JSON schema:
{
  "metric": <string, e.g. "revenue" or "churn">,
  "timeframe": [ "YYYY-MM-DD", "YYYY-MM-DD" ],
  "time_grain": <"day" | "week" | "month">,
  "filters": { "<col>": "<value>", ... },
  "intent": <"trend" | "anomaly" | "compare" | "slice">
}

Rules:
- Convert vague ranges like "last week", "Q3", "last month" into explicit ISO date strings if possible (use session timeframe if provided).
- If filters are not present, return {}.
- If unsure about dates, default to last 4 weeks.
- DO NOT output any explanatory textâ€”only the JSON object.
"""

INTERPRETER_INSTRUCTION = """
Please return only the JSON following the schema above.
User query:
"""

# -------------------------
# Helpers
# -------------------------
def _default_last_4_weeks():
    end = pd.to_datetime("today")
    start = end - pd.Timedelta(days=28)
    return [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]


def _simple_parse_fallback(user_q, session=None):
    print("ğŸ”„ _simple_parse_fallback active.")
    session = session or {}

    metric = "revenue" if re.search(r"revenue|sales", user_q, re.I) else (
        "churn" if re.search(r"churn", user_q, re.I) else session.get("current_metric", "revenue")
    )

    if re.search(r"last week", user_q, re.I):
        end = pd.to_datetime(session.get("timeframe", _default_last_4_weeks())[1])
        start = end - pd.Timedelta(days=7)
        timeframe = [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]
        time_grain = "week"
    elif re.search(r"last month", user_q, re.I):
        end = pd.to_datetime(session.get("timeframe", _default_last_4_weeks())[1])
        start = end - pd.Timedelta(days=30)
        timeframe = [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]
        time_grain = "month"
    else:
        timeframe = session.get("timeframe", _default_last_4_weeks())
        time_grain = "week"

    filters = {}
    m = re.search(r"apac|amer|emea", user_q, re.I)
    if m:
        filters["region"] = m.group(0).upper()

    m2 = re.search(r"product\s*[:= ]\s*([A-C])", user_q, re.I)
    if m2:
        filters["product"] = m2.group(1).upper()

    intent = "anomaly" if re.search(r"anomal|drop|why|change", user_q, re.I) else "trend"

    return {"metric": metric, "timeframe": timeframe, "time_grain": time_grain, "filters": filters, "intent": intent}


def _local_extract_text_from_raw(raw):
    """
    Local fallback extractor to pull text/JSON from various raw shapes
    (used if wrapper's _extract_text_from_response is not available).
    """
    try:
        # common: string
        if isinstance(raw, str):
            return raw

        # if wrapper returned an object, try to inspect a few attributes
        for attr in ("text", "output", "result", "response", "candidates", "content"):
            if hasattr(raw, attr):
                val = getattr(raw, attr)
                # if string
                if isinstance(val, str) and val.strip():
                    return val
                # if list-like, try first element
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    first = val[0]
                    if isinstance(first, str) and first.strip():
                        return first
                    if isinstance(first, dict) and "text" in first:
                        return first["text"]
                    # try attributes on object
                    for k in ("text", "content", "message"):
                        if hasattr(first, k) and getattr(first, k):
                            return getattr(first, k)
                    try:
                        return json.dumps(first)
                    except Exception:
                        return str(first)
                # if dict-like
                if isinstance(val, dict):
                    try:
                        return json.dumps(val)
                    except Exception:
                        return str(val)

        # last resort: str(raw)
        try:
            s = str(raw)
            # try to find JSON substring
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = s[start:end+1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    return s
            return s
        except Exception:
            return None
    except Exception:
        return None


def _find_all_balanced_json_substrings(s: str):
    """
    Scan the string for balanced {...} substrings and return them in appearance order.
    This helps find JSON embedded inside SDK debug text.
    """
    subs = []
    if not s:
        return subs
    start_idx = None
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start_idx is not None:
                subs.append(s[start_idx:i + 1])
                start_idx = None
    return subs


def _try_parse(candidate_text: str):
    try:
        return json.loads(candidate_text)
    except Exception:
        return None


# -------------------------
# Interpreter
# -------------------------
def interpret_query(user_query: str, session: dict = None, use_mock: bool = None, max_tokens: int = 512):
    """
    Parse a single user KPI query into JSON using Gemini (preferred) with a robust fallback pipeline.
    Returns dict: {metric, timeframe, time_grain, filters, intent}
    Behavior:
      - Tries to call gemini_generate(prompt, use_mock=..., return_raw=True) when available.
      - Prefers extracting from the raw SDK response when returned.
      - Cleans typical markdown fences and SDK wrapper text.
      - Attempts parsing; if parse fails, retries once with a larger token budget (1024).
      - If still failing, falls back to _simple_parse_fallback.
      - Normalizes missing fields from session/defaults.
    """
    print("\n==============================")
    print("ğŸ”� interpret_query() STARTED")
    print("==============================")
    print("ğŸ“¥ Received query:", user_query)

    session = session or {}
    print("ğŸ“� Session snapshot:", session)

    # Build prompt
    prompt = INTERPRETER_SYSTEM_PROMPT + "\n" + INTERPRETER_INSTRUCTION + "\n" + user_query.strip()
    print("ğŸ§± Prompt built successfully (len=%d)." % len(prompt))

    # Decide mock usage
    wrapper_defined = "gemini_generate" in globals()
    gemini_ready = wrapper_defined and globals().get("GEMINI_AVAILABLE", False) and globals().get("GEMINI_CLIENT") is not None
    if use_mock is None:
        use_mock = not wrapper_defined or not gemini_ready
    print(f"ğŸš¦ use_mock={use_mock} (wrapper_defined={wrapper_defined}, gemini_ready={gemini_ready})")

    # Helper: clean raw text from SDK/model (strip fences, wrappers)
    def _clean_raw_text(s_raw: str) -> str:
        s = str(s_raw or "")
        # Remove common triple-backtick fences (```json ... ```), keep JSON inside
        s = s.replace("```json", "```").replace("```", "")
        # Remove stray leading/trailing whitespace/newlines
        s = s.strip()
        # If SDK prints "content=Content(...)" we still return the whole string for extractor to hunt braces
        return s

    # Call Gemini (or mock) to get raw output string/object â€” prefer return_raw for better extraction
    raw_text_candidate = None
    raw_resp = None
    try:
        if wrapper_defined:
            print("ğŸ¤– Calling gemini_generate(prompt, use_mock=%s, max_output_tokens=%d, return_raw=True) ..." % (use_mock, max_tokens))
            maybe = None
            try:
                maybe = gemini_generate(prompt, use_mock=use_mock, max_output_tokens=max_tokens, return_raw=True)
            except TypeError:
                # wrapper might not support return_raw; fall back to older signature
                maybe = gemini_generate(prompt, use_mock=use_mock, max_output_tokens=max_tokens)
            # wrapper may return (text, raw_resp) or just text
            if isinstance(maybe, tuple) and len(maybe) == 2:
                raw_text_candidate, raw_resp = maybe
            else:
                raw_text_candidate, raw_resp = maybe, None
            print("ğŸ“¬ gemini_generate returned (text_type=%s, raw_resp_type=%s)" % (type(raw_text_candidate).__name__, type(raw_resp).__name__))
        else:
            print("âš ï¸� gemini_generate not defined in globals â€” will use fallback.")
            raw_text_candidate, raw_resp = None, None
    except NameError as ne:
        print("âš ï¸� gemini_generate not present:", ne)
        raw_text_candidate, raw_resp = None, None
    except Exception as e:
        print("â�— gemini_generate raised:", repr(e))
        raw_text_candidate, raw_resp = None, None

    # If no real raw, and wrapper exists but returned None, request mock from wrapper
    if (raw_text_candidate is None or str(raw_text_candidate).strip() == "") and wrapper_defined:
        try:
            print("â�¡ï¸� Requesting gemini_generate mock fallback (return_raw=True)...")
            maybe = None
            try:
                maybe = gemini_generate(prompt, use_mock=True, max_output_tokens=64, return_raw=True)
            except TypeError:
                maybe = gemini_generate(prompt, use_mock=True, max_output_tokens=64)
            if isinstance(maybe, tuple) and len(maybe) == 2:
                raw_text_candidate, raw_resp = maybe
            else:
                raw_text_candidate, raw_resp = maybe, raw_resp
        except Exception as e:
            print("â�— gemini fallback mock failed:", e)
            raw_text_candidate, raw_resp = raw_text_candidate, raw_resp

    # Build a canonical string to search for JSON â€” prefer the explicit text, otherwise inspect raw_resp then raw_text_candidate then str(raw)
    raw_text = ""
    if raw_text_candidate:
        raw_text = _clean_raw_text(raw_text_candidate)
    elif raw_resp is not None:
        print("â„¹ï¸� Attempting to extract text from raw_resp attributes...")
        extracted = _local_extract_text_from_raw(raw_resp)
        raw_text = _clean_raw_text(extracted or "")
    else:
        raw_text = _clean_raw_text(raw_text_candidate or "")

    if not raw_text:
        # fallback to stringifying whatever we got
        raw_text = _clean_raw_text(str(raw_resp or raw_text_candidate or ""))

    print("ğŸ“¥ Raw model output preview (first 500 chars):")
    print(raw_text[:500])

    # Extract largest balanced JSON substring if possible
    parsed = None
    candidate = None

    s_trim = raw_text.strip()

    # detect mocks via either prefix
    if s_trim.startswith(LEGACY_MOCK_PREFIX) or s_trim.startswith(WRAPPER_MOCK_PREFIX):
        prefix = LEGACY_MOCK_PREFIX if s_trim.startswith(LEGACY_MOCK_PREFIX) else WRAPPER_MOCK_PREFIX
        candidate = s_trim.split(prefix, 1)[1].strip()
        print("ğŸ§¾ Detected mock prefix; extracted candidate snippet.")
        parsed = _try_parse(candidate)
    else:
        # try to find all balanced {...} substrings and attempt parse (prefer largest valid)
        candidates = _find_all_balanced_json_substrings(raw_text)
        if candidates:
            # try largest-first (more likely contains full object)
            candidates = sorted(candidates, key=len, reverse=True)
            for cand in candidates:
                parsed_try = _try_parse(cand)
                if parsed_try is not None:
                    parsed = parsed_try
                    candidate = cand
                    print(f"ğŸª„ Parsed JSON candidate from balanced-substring (len={len(cand)}).")
                    break

    # If parsing failed, attempt one retry with larger token budget (only if gemini available & not using mock)
    # Also use raw_resp to detect truncation (finish_reason == MAX_TOKENS)
    should_retry = parsed is None
    if raw_resp is not None:
        try:
            # SDK shapes differ; attempt to detect finish_reason in common spots
            fin = None
            # if raw_resp has attribute 'candidates', inspect first candidate
            if hasattr(raw_resp, "candidates"):
                candlist = getattr(raw_resp, "candidates")
                if isinstance(candlist, (list, tuple)) and len(candlist) > 0:
                    first = candlist[0]
                    # try attribute or dict-like
                    fin = getattr(first, "finish_reason", None) or (first.get("finish_reason") if isinstance(first, dict) else None)
            else:
                # try to parse any substring indicating MAX_TOKENS
                rep = str(raw_resp)
                if "MAX_TOKENS" in rep or "finish_reason" in rep:
                    fin = "MAX_TOKENS"
            if fin and ("MAX_TOKENS" in str(fin) or "max_tokens" in str(fin).lower()):
                print("âš ï¸� Detected model truncation (MAX_TOKENS) in raw_resp â€” will retry with larger token budget.")
                should_retry = True
        except Exception:
            pass

    if parsed is None and not use_mock and wrapper_defined and should_retry:
        print("â�— JSON parsing failed or incomplete; attempting one retry with larger token budget (1024).")
        try:
            maybe_retry = None
            try:
                maybe_retry = gemini_generate(prompt, use_mock=False, max_output_tokens=1024, return_raw=True)
            except TypeError:
                maybe_retry = gemini_generate(prompt, use_mock=False, max_output_tokens=1024)
            if isinstance(maybe_retry, tuple) and len(maybe_retry) == 2:
                raw_retry_candidate, raw_retry_resp = maybe_retry
            else:
                raw_retry_candidate, raw_retry_resp = maybe_retry, None

            raw_retry_text = _clean_raw_text(raw_retry_candidate or _local_extract_text_from_raw(raw_retry_resp) or "")
            print("ğŸ“¥ Retry raw preview (first 500 chars):")
            print(raw_retry_text[:500])

            # try balanced extraction on retry
            candidates_retry = _find_all_balanced_json_substrings(raw_retry_text)
            if candidates_retry:
                candidates_retry = sorted(candidates_retry, key=len, reverse=True)
                for cand in candidates_retry:
                    parsed_try = _try_parse(cand)
                    if parsed_try is not None:
                        parsed = parsed_try
                        candidate = cand
                        print("âœ… Retry parse succeeded (balanced-substring).")
                        break

            # final simple parse attempt (maybe full string is JSON)
            if parsed is None:
                parsed_try = _try_parse(raw_retry_text)
                if parsed_try is not None:
                    parsed = parsed_try
                    candidate = raw_retry_text
                    print("âœ… Retry parse succeeded on full retry text.")
        except Exception as e:
            print("â�— Retry attempt failed:", repr(e))

    # Last fallback: if still not parsed, use simple rule-based parser
    if parsed is None:
        print("âš ï¸� Falling back to simple rule-based parser.")
        parsed = _simple_parse_fallback(user_query, session)
        print("ğŸ“Œ Fallback parsed result:", parsed)
    else:
        print("ğŸ“¦ JSON parsed successfully from model output.")

    # Normalize missing fields (use session defaults)
    if not parsed.get("filters"):
        parsed["filters"] = session.get("active_filters", {})
        print("ğŸ”§ Filters normalized.")

    if not parsed.get("timeframe"):
        parsed["timeframe"] = session.get("timeframe", None) or _default_last_4_weeks()
        print("ğŸ”§ Timeframe normalized.")

    if not parsed.get("metric"):
        parsed["metric"] = session.get("current_metric", "revenue")
        print("ğŸ”§ Metric normalized.")

    if not parsed.get("time_grain"):
        parsed["time_grain"] = parsed.get("time_grain") or "week"
        print("ğŸ”§ Time grain normalized.")

    if not parsed.get("intent"):
        parsed["intent"] = parsed.get("intent") or "anomaly"
        print("ğŸ”§ Intent normalized.")

    print("âœ… FINAL PARSED JSON:", parsed)
    print("ğŸ�‰ interpret_query() COMPLETED")

    return parsed


# === Quick tests (they attempt real Gemini when available) ===
print("\nğŸš€ Running quick interpret_query tests (prefer real Gemini)...")

print("\nğŸ§ª Test 1 (APAC revenue):")
try:
    res1 = interpret_query("Why did revenue drop last week in APAC?", session={}, use_mock=False, max_tokens=512)
    print("=>", res1)
except Exception as e:
    print("Test1 error:", e)
    traceback.print_exc()

print("\nğŸ§ª Test 2 (EMEA churn):")
try:
    res2 = interpret_query("Any anomalies in churn last month in EMEA?", session={}, use_mock=False, max_tokens=512)
    print("=>", res2)
except Exception as e:
    print("Test2 error:", e)
    traceback.print_exc()

print("\nğŸ�‰ Gemini Interpreter (improved) setup COMPLETE.")



# === Insights-from-Interpreter Agent ===
# Takes the interpreter's JSON output (metric, timeframe, time_grain, filters, intent)
# and produces human-friendly sentences (title, narrative, follow_ups).
# It will call the gemini wrapper for a high-quality narrative when available,
# and otherwise fall back to a deterministic local generator (safe, no invented data).
#

import json
import textwrap
import time
import traceback
from typing import Optional

print("ğŸ“¦ Loading Insights-from-Interpreter Agent...")

# Try to pick up wrapper mock prefix if available; keep legacy sentinel too
WRAPPER_MOCK_PREFIX = globals().get("MOCK_PREFIX", "__MOCK_JSON_RESPONSE__:")
LEGACY_MOCK_PREFIX = "MOCK_JSON_RESPONSE:"

# -------------------------
# Helpers
# -------------------------
def _compact_context_from_interpreted(interp: dict) -> str:
    """Turn the interpreter dict into a short context string for the model."""
    metric = interp.get("metric", "metric")
    tf = interp.get("timeframe", None)
    grain = interp.get("time_grain", None)
    filters = interp.get("filters", {})
    intent = interp.get("intent", "trend")

    parts = [f"metric={metric}"]
    if tf:
        parts.append(f"timeframe={tf}")
    if grain:
        parts.append(f"time_grain={grain}")
    if filters:
        fparts = ",".join([f"{k}={v}" for k, v in filters.items()])
        parts.append(f"filters={fparts}")
    parts.append(f"intent={intent}")
    return "; ".join(parts)

def _build_local_insight_from_interpreted(interp: dict):
    """
    Deterministic, safe fallback that composes a short title, a narrative
    framed as hypotheses (not factual claims) and 2-3 recommended follow-ups.
    This function deliberately does NOT invent numeric facts.
    """
    metric = interp.get("metric", "the KPI")
    timeframe = interp.get("timeframe", ["<start>","<end>"])
    time_grain = interp.get("time_grain", "period")
    filters = interp.get("filters", {})
    intent = interp.get("intent", "trend")

    # Title: short, <= 8 words
    title_parts = [metric.capitalize()]
    if filters:
        # include first filter key/value for context
        k, v = list(filters.items())[0]
        title_parts.append(f"in {v}")
    title_parts.append("Analysis")
    title = " ".join(title_parts)[:80]

    # Narrative: 2-4 sentences, phrased as hypotheses & constraints
    sentences = []
    sentences.append(f"This is a hypothesis-driven note for {metric} over {timeframe[0]} to {timeframe[1]}.")
    if filters:
        fsummary = ", ".join([f"{k}={v}" for k, v in filters.items()])
        sentences.append(f"Focus is scoped to {fsummary}, so findings will be specific to that slice.")
    if intent and intent.lower() in ("anomaly", "compare"):
        sentences.append(
            f"Suggested initial assumption: look for sudden shifts, data gaps, or segmentation effects rather than stable seasonal trends."
        )
    else:
        sentences.append(
            f"Suggested initial assumption: examine week-over-week or month-over-month trends and seasonality as plausible drivers."
        )
    # short guidance sentence
    sentences.append("Do not assume root cause without checking data quality, sample size, and major campaign or product changes.")

    narrative = " ".join(sentences)

    # Follow-ups: 2-3 clear next steps
    follow_ups = []
    # step 1: data quality
    follow_ups.append("Check data ingestion and pipeline health for the selected timeframe and filters.")
    # step 2: drilldown
    if filters:
        # recommend widening or slicing differently
        follow_ups.append(f"Compare the same metric without the {list(filters.keys())[0]} filter to see if the effect is localized.")
    else:
        follow_ups.append("Drill down by region/product to identify high-contribution slices (top 3).")
    # step 3: operational checks
    follow_ups.append("Review recent product releases, marketing campaigns, or pricing changes that overlap the timeframe.")

    return {"title": title, "narrative": narrative, "follow_ups": follow_ups}

def _find_all_balanced_json_substrings(s: str):
    subs = []
    if not s:
        return subs
    start_idx = None
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start_idx is not None:
                subs.append(s[start_idx:i + 1])
                start_idx = None
    return subs

def _try_parse(candidate_text: str):
    try:
        return json.loads(candidate_text)
    except Exception:
        return None

def _clean_text_for_extraction(s_raw):
    s = str(s_raw or "")
    s = s.replace("```json", "```").replace("```", "")
    return s.strip()

# -------------------------
# Main function
# -------------------------
def generate_insights_from_interpreter(interpreted_query: dict,
                                      use_gemini: Optional[bool] = None,
                                      max_tokens: int = 512):
    """
    Accepts: interpreted_query dict produced by the interpreter agent.
    Returns: {"insights": {title,narrative,follow_ups}, "plain_text": "...", "_meta": {...}}
    Behavior:
      - If use_gemini is True (or None and wrapper indicates ready), calls gemini_generate to
        request a succinct JSON (title,narrative,follow_ups) based only on the interpreted_query.
      - If Gemini call fails or is unavailable, falls back to _build_local_insight_from_interpreted.
      - IMPORTANT: when only the interpreted_query is provided (no numeric summary), the model is asked
        to produce *hypotheses* and recommended next steps, not to invent numbers.
    """
    print("\n==============================")
    print("ğŸ”” generate_insights_from_interpreter START")
    print("==============================")
    print("ğŸ“¥ interpreted_query preview:", _compact_context_from_interpreted(interpreted_query))

    wrapper_defined = "gemini_generate" in globals()
    gemini_ready = wrapper_defined and globals().get("GEMINI_AVAILABLE", False) and globals().get("GEMINI_CLIENT") is not None
    if use_gemini is None:
        use_gemini = gemini_ready
    print(f"ğŸš¦ use_gemini={use_gemini} (wrapper_defined={wrapper_defined}, gemini_ready={gemini_ready})")

    # If not using Gemini, produce deterministic local output
    if not use_gemini or not wrapper_defined:
        print("â�¡ï¸� Using local deterministic insights (no Gemini).")
        parsed = _build_local_insight_from_interpreted(interpreted_query)
    else:
        # Build a compact prompt asking for hypotheses + follow-ups, explicitly forbidding invented numeric claims
        prompt = (
            "You are a succinct business analyst. INPUT (interpreted query):\n"
            f"{json.dumps(interpreted_query)}\n\n"
            "Task: Based ONLY on the interpreted query (metric, timeframe, time_grain, filters, intent), "
            "write a JSON object with keys: title, narrative, follow_ups.\n"
            "Rules:\n"
            "- Do NOT invent or assert numeric values. If you refer to effects, label them as hypotheses (e.g., 'possible driver: ...').\n"
            "- Narrative: 2-4 concise sentences phrased as hypotheses + constraints.\n"
            "- title: <=8 words.\n"
            "- follow_ups: list 2-3 clear next analysis or checks (data quality, drilldowns, operational checks).\n"
            "- Return only the JSON object and nothing else.\n"
            "Return compact JSON, no extra commentary.\n"
        )

        parsed = None
        parsed_candidate_text = None
        # Try multiple budgets to reduce truncation
        budgets = [max_tokens, 1024, 2048]
        seen = set()
        budgets = [b for b in budgets if b and b not in seen and not seen.add(b)]
        last_raw_preview = None

        for b in budgets:
            try:
                print(f"ğŸ¤– Calling gemini_generate(..., max_output_tokens={b}, return_raw=True) ...")
                try:
                    maybe = gemini_generate(prompt, use_mock=False, max_output_tokens=b, return_raw=True)
                except TypeError:
                    maybe = gemini_generate(prompt, use_mock=False, max_output_tokens=b)
                # maybe can be (text, raw_resp) or text
                if isinstance(maybe, tuple) and len(maybe) == 2:
                    text_candidate, raw_resp = maybe
                else:
                    text_candidate, raw_resp = maybe, None

                # normalize candidate text
                if text_candidate and str(text_candidate).strip():
                    candidate_text = _clean_text_for_extraction(text_candidate)
                elif raw_resp is not None:
                    candidate_text = _clean_text_for_extraction(_local_extract_text_from_raw(raw_resp))
                else:
                    candidate_text = _clean_text_for_extraction(str(maybe or ""))

                last_raw_preview = candidate_text
                print("ğŸ“¥ Raw preview (first 400 chars):")
                print(candidate_text[:400])

                # detect wrapper mock sentinel
                st = candidate_text.strip()
                if st.startswith(WRAPPER_MOCK_PREFIX) or st.startswith(LEGACY_MOCK_PREFIX):
                    print("ğŸ§¾ Wrapper returned mock sentinel; will fall back to local insights.")
                    parsed = _build_local_insight_from_interpreted(interpreted_query)
                    break

                # try to extract balanced JSON from the response
                candidates_json = _find_all_balanced_json_substrings(candidate_text)
                if candidates_json:
                    candidates_json = sorted(candidates_json, key=len, reverse=True)
                    for cj in candidates_json:
                        parsed_try = _try_parse(cj)
                        if parsed_try is not None:
                            parsed = parsed_try
                            parsed_candidate_text = cj
                            print("âœ… Parsed JSON from Gemini output (balanced substring).")
                            break

                # try full-string parse as last resort
                if parsed is None:
                    parsed_try = _try_parse(candidate_text)
                    if parsed_try is not None:
                        parsed = parsed_try
                        parsed_candidate_text = candidate_text
                        print("âœ… Parsed JSON from Gemini output (full text).")

                # If parsed, stop.
                if parsed is not None:
                    break

                # If not parsed, inspect raw_resp for truncation and continue to next budget
                truncated = False
                try:
                    if raw_resp is not None:
                        fin = None
                        if hasattr(raw_resp, "candidates"):
                            candlist = getattr(raw_resp, "candidates")
                            if isinstance(candlist, (list, tuple)) and len(candlist) > 0:
                                first = candlist[0]
                                fin = getattr(first, "finish_reason", None) or (first.get("finish_reason") if isinstance(first, dict) else None)
                        else:
                            rep = str(raw_resp)
                            if "MAX_TOKENS" in rep or "finish_reason" in rep:
                                fin = "MAX_TOKENS"
                        if fin and ("MAX_TOKENS" in str(fin) or "max_tokens" in str(fin).lower()):
                            truncated = True
                            print("âš ï¸� Detected truncation (MAX_TOKENS) in raw_resp; trying larger budget.")
                except Exception:
                    pass

                # short pause before next attempt
                time.sleep(0.1)
                continue

            except Exception as e:
                print("â�— gemini_generate raised during insights call:", repr(e))
                traceback.print_exc()
                time.sleep(0.1)
                continue

        # end budgets loop

        if parsed is None:
            print("â�¡ï¸� Gemini did not produce parseable JSON; falling back to local deterministic insights.")
            parsed = _build_local_insight_from_interpreted(interpreted_query)

    # Normalize resulting structure
    parsed.setdefault("title", parsed.get("title", "")[:80])
    parsed.setdefault("narrative", parsed.get("narrative", ""))
    parsed.setdefault("follow_ups", parsed.get("follow_ups", []))

    # Build a plain text version
    plain_lines = []
    plain_lines.append(f"Insights: {parsed['title']}")
    if parsed['narrative']:
        plain_lines.append(parsed['narrative'])
    if parsed['follow_ups']:
        plain_lines.append("\nRecommended follow-ups:")
        for f in parsed['follow_ups']:
            plain_lines.append("- " + f)
    plain_text = "\n".join(plain_lines)

    meta = {
        "interpreted_query": interpreted_query,
        "used_gemini": use_gemini,
        "timestamp": time.time()
    }

    print("\nğŸ”š generate_insights_from_interpreter COMPLETE.")
    return {"insights": parsed, "plain_text": plain_text, "_meta": meta}

# -------------------------
# Quick demo if run as script (local fallback if no wrapper)
# -------------------------
if __name__ == "__main__":
    sample_interpreted = {
        "metric": "revenue",
        "timeframe": ["2025-11-24", "2025-12-01"],
        "time_grain": "week",
        "filters": {"region": "APAC"},
        "intent": "anomaly"
    }
    out = generate_insights_from_interpreter(sample_interpreted, use_gemini=None, max_tokens=512)
    print("\n--- Insights (JSON) ---")
    print(json.dumps(out["insights"], indent=2))
    print("\n--- Plain text ---")
    print(out["plain_text"])



# orchestrator
"""

This orchestrator ties together your existing agents:
 - interpret_query(user_query, session, use_mock, max_tokens)
 - load_and_summarize_kpis(data_path, metric, timeframe, filters, time_grain)
 - triage_loop(metric, timeframe, base_filters, data_path, max_depth, max_candidates)
 - generate_insights(summary, candidates, use_mock, max_tokens)
 - generate_insights_from_interpreter(interpreted_query, use_gemini, max_tokens)

"""

import time
from typing import Callable, Optional, Dict, Any, Tuple

# -------------------------
# Configurable FORCE flags
# -------------------------
# To reproduce the exact standalone insights run you showed earlier,
# set FORCE_USE_INTERPRETER_INSIGHTS = True, FORCE_USE_GEMINI=True, FORCE_MAX_TOKENS=1024
FORCE_USE_INTERPRETER_INSIGHTS = True
FORCE_USE_GEMINI = True
FORCE_MAX_TOKENS = 1024

# -------------------------
# Orchestrator
# -------------------------
def orchestrate_query(
    user_query: str,
    data_path: Optional[str] = None,
    session: Optional[dict] = None,
    data_fetch_fn: Optional[Callable[[dict], Tuple[dict, list]]] = None,
    use_mock: Optional[bool] = None,
    interpreter_max_tokens: int = 512,
    insights_max_tokens: int = 512,
    triage_max_depth: int = 3,
    triage_max_candidates: int = 5,
) -> Dict[str, Any]:
    """
    Minimal orchestrator that drives the pipeline:
      Interpreter -> (data fetch -> triage) -> Insights

    This variant forces the interpreter-driven insights path when
    FORCE_USE_INTERPRETER_INSIGHTS is True (so output matches standalone runs
    that used generate_insights_from_interpreter(...)).

    Returns a dict with keys:
      status, errors, interpreted, summary, candidates, insights, plain_text, _meta
    """
    start_time = time.time()
    session = session or {}

    print("â†’ Orchestration: started")

    result = {
        "status": "error",
        "errors": [],
        "interpreted": None,
        "summary": None,
        "candidates": None,
        "insights": None,
        "plain_text": None,
        "_meta": {}
    }

    try:
        # 1) Interpret user query
        print("â€¢ Interpreting query...")
        interpreted = interpret_query(user_query, session=session, use_mock=use_mock, max_tokens=interpreter_max_tokens)
        result["interpreted"] = interpreted

        # 2) Fetch data (either data_fetch_fn or load_and_summarize_kpis + triage)
        summary = None
        candidates = []

        print("â€¢ Fetching data / summarizing KPI...")
        if data_fetch_fn is not None:
            try:
                out = data_fetch_fn(interpreted)
                if isinstance(out, (tuple, list)):
                    summary = out[0]
                    candidates = out[1] if len(out) > 1 else []
                elif isinstance(out, dict):
                    summary = out
                    candidates = []
                else:
                    raise ValueError("data_fetch_fn returned unexpected type")
            except Exception as e:
                result["errors"].append({"step": "data_fetch_fn", "error": str(e)})
                summary = None
                candidates = []
        else:
            # call your existing tool to compute numeric summary
            try:
                metric = interpreted.get("metric")
                timeframe = interpreted.get("timeframe")
                filters = interpreted.get("filters", {})
                time_grain = interpreted.get("time_grain", None)
                summary = load_and_summarize_kpis(data_path, metric=metric, timeframe=timeframe, filters=filters, time_grain=time_grain)
            except Exception as e:
                result["errors"].append({"step": "load_and_summarize_kpis", "error": str(e)})
                summary = None

            # run triage if summary exists
            if summary is not None:
                try:
                    candidates = triage_loop(
                        metric=interpreted.get("metric"),
                        timeframe=interpreted.get("timeframe"),
                        base_filters=interpreted.get("filters", {}),
                        data_path=data_path,
                        max_depth=triage_max_depth,
                        max_candidates=triage_max_candidates
                    )
                except Exception as e:
                    result["errors"].append({"step": "triage_loop", "error": str(e)})
                    candidates = []

        result["summary"] = summary
        result["candidates"] = candidates

        print("â€¢ Triage complete (candidates found: {})".format(len(candidates) if candidates is not None else 0))

        # 3) Insights stage â€” FORCE same path as standalone run when configured
        print("â€¢ Generating insights...")

        # By default we force the interpreter-driven insights path for reproducibility.
        insights_out = None

        if FORCE_USE_INTERPRETER_INSIGHTS:
            # Call interpreter-driven insights exactly as in standalone runs
            try:
                insights_out = generate_insights_from_interpreter(
                    interpreted,
                    use_gemini=FORCE_USE_GEMINI,
                    max_tokens=FORCE_MAX_TOKENS
                )
                result["insights"] = insights_out.get("insights")
                result["plain_text"] = insights_out.get("plain_text")
                result["_meta"]["insights_path"] = "forced_interpreter_path"
                result["_meta"]["used_gemini"] = FORCE_USE_GEMINI
                result["_meta"]["max_tokens"] = FORCE_MAX_TOKENS
            except Exception as e:
                result["errors"].append({"step": "generate_insights_from_interpreter", "error": str(e)})
        else:
            # Normal summary-based behavior
            try:
                insights_out = generate_insights(summary or {}, candidates or [], use_mock=use_mock, max_tokens=insights_max_tokens)
                result["insights"] = insights_out.get("insights")
                result["plain_text"] = insights_out.get("plain_text")
                result["_meta"]["insights_path"] = "summary_based"
            except Exception as e:
                result["errors"].append({"step": "generate_insights", "error": str(e)})

        # finalize success
        result["status"] = "success"
        result["_meta"].update({
            "timestamp": time.time(),
            "elapsed_s": round(time.time() - start_time, 3),
            "candidates_count": len(candidates) if candidates is not None else 0
        })

        print("â†’ Orchestration: completed (elapsed {:.2f}s)".format(time.time() - start_time))
        return result

    except Exception as e:
        result["errors"].append({"step": "orchestrator_uncaught", "error": str(e)})
        print("â†’ Orchestration: failed:", str(e))
        return result


# -------------------------
# Minimal demo run (optional)
# -------------------------
if __name__ == "__main__":
    # Quick demo - adjust DATA_PATH or other globals as needed in your environment
    example_query = "Why did revenue drop last week in APAC?"
    try:
        out = orchestrate_query(example_query, data_path=globals().get("DATA_PATH", None), session={}, use_mock=False)
        import json
        print("\n--- Final insights (plain text) ---\n")
        print(out.get("plain_text") or json.dumps(out.get("insights", {}), indent=2))
    except Exception as exc:
        print("Orchestrator demo failed:", exc)


