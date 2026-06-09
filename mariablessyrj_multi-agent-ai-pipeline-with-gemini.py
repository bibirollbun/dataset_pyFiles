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


import requests
from bs4 import BeautifulSoup
import concurrent.futures
import time
import json
import logging
import os
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger('agents-capstone')

print("imports ready")



def web_search_duckduckgo(query: str, max_results: int = 5) -> List[Dict]:
    """
    Lightweight scraping of DuckDuckGo HTML results.
    NOTE: Kaggle internet may be restricted; if blocked, replace with cached data or a pre-downloaded file.
    """
    try:
        url = 'https://duckduckgo.com/html/'
        resp = requests.get(url, params={'q': query}, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        items = []
        # pages use anchor with class 'result__a' — may change; this is a best-effort
        for a in soup.select('a.result__a')[:max_results]:
            title = a.get_text().strip()
            href = a.get('href')
            items.append({'title': title, 'url': href})
        return items
    except Exception as e:
        logger.warning('web_search_duckduckgo failed: %s', e)
        return []

def fetch_weather_wttr(city: str) -> str:
    """Simple external tool using wttr.in (quick demo OpenAPI-type tool)"""
    try:
        resp = requests.get(f'https://wttr.in/{city}?format=3', timeout=5)
        return resp.text
    except Exception as e:
        logger.warning('fetch_weather_wttr failed: %s', e)
        return 'weather-unavailable'

print("tools ready")



class MemoryBank:
    def __init__(self, path: str = '/kaggle/working/agents_memory.json'):
        self.path = path
        self._data = []
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self._data = json.load(f)
            except Exception:
                self._data = []

    def add(self, item: str):
        logger.info('MemoryBank: adding item (len=%d chars)', len(item))
        self._data.append({'ts': time.time(), 'text': item})
        with open(self.path, 'w') as f:
            json.dump(self._data, f)

    def recall(self, n: int = None) -> List[Dict]:
        return self._data if n is None else self._data[-n:]

    def compact(self):
        # simple compaction: keep last 50 items
        self._data = self._data[-50:]
        with open(self.path, 'w') as f:
            json.dump(self._data, f)

memory = MemoryBank('/kaggle/working/agents_memory.json')
print("memory ready at", memory.path)



def call_llm(prompt: str, system: str = None, max_retries: int = 2, retry_delay: float = 1.0) -> str:
    """
    Robust Gemini (Google) LLM caller for Kaggle.
    - Reads GOOGLE_API_KEY from Kaggle Secrets (or env var fallback).
    - Uses model: models/gemini-2.5-flash (change MODEL_NAME if desired).
    - Gracefully falls back to a mock response if anything fails.
    """
    from kaggle_secrets import UserSecretsClient
    import os, time, logging

    # local logger
    lg = logging.getLogger("call_llm_gemini")
    lg.setLevel(logging.INFO)

    # Fetch API key from Kaggle Secrets (preferred) or environment
    try:
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    except Exception:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        lg.error("GOOGLE_API_KEY not found in Kaggle Secrets or environment.")
        return "<<FALLBACK_MOCK>> No API key found. " + (prompt[:300].replace("\n", " ") + ("..." if len(prompt)>300 else ""))

    # Compose final prompt (include optional system prefix)
    final_prompt = (system.strip() + "\n\n" + prompt) if system else prompt

    # Choose model from the models you listed
    MODEL_NAME = "models/gemini-2.5-flash"   # <-- good default for your key; change if you prefer another supported model

    # Lazy import of genai SDK (so cell doesn't crash if sdk missing)
    try:
        from google import genai
    except Exception as e:
        lg.error("google.genai SDK not available: %s", e)
        # fallback mock
        snippet = final_prompt[:400].replace("\n", " ") + ("..." if len(final_prompt) > 400 else "")
        return f"<<FALLBACK_MOCK - no-sdk>> Summary: {snippet}"

    client = genai.Client(api_key=api_key)

    # Attempt requests with a small retry loop
    attempt = 0
    while attempt <= max_retries:
        try:
            lg.info("Gemini LLM call (model=%s) attempt %d prompt_len=%d", MODEL_NAME, attempt+1, len(final_prompt))
            response = client.models.generate_content(model=MODEL_NAME, contents=final_prompt)

            # Try multiple safe extraction methods for text
            text = None
            # 1) SDK may expose .text
            if hasattr(response, "text") and response.text:
                text = response.text
            # 2) some SDKs return dict-like with 'candidates'
            if not text:
                try:
                    # dict-like
                    if isinstance(response, dict) and "candidates" in response and response["candidates"]:
                        cand = response["candidates"][0]
                        # candidate content might be nested
                        if isinstance(cand, dict):
                            # try content->text or content list
                            if "content" in cand:
                                cont = cand["content"]
                                # cont could be a string or list
                                if isinstance(cont, str):
                                    text = cont
                                elif isinstance(cont, list) and cont:
                                    # try to find text fields inside
                                    maybe = cont[0]
                                    if isinstance(maybe, dict) and "text" in maybe:
                                        text = maybe["text"]
                                    elif isinstance(maybe, str):
                                        text = maybe
                except Exception:
                    pass
            # 3) other SDK shapes e.g., response.candidates[0].content[0].text
            if not text:
                try:
                    cand = getattr(response, "candidates", None)
                    if cand and len(cand) > 0:
                        maybe = cand[0]
                        # try attribute chain
                        if hasattr(maybe, "content"):
                            cont = maybe.content
                            if isinstance(cont, list) and len(cont) > 0:
                                piece = cont[0]
                                if hasattr(piece, "text"):
                                    text = piece.text
                except Exception:
                    pass

            # 4) fallback: string representation
            if not text:
                try:
                    text = str(response)
                except Exception:
                    text = None

            if not text:
                raise RuntimeError("Could not extract text from Gemini response; raw response logged.")

            lg.info("Gemini call succeeded; returned %d chars", len(text))
            return text

        except Exception as e:
            # Log error, then retry if attempts remain
            lg.warning("Gemini call attempt %d failed: %s", attempt+1, str(e))
            attempt += 1
            if attempt <= max_retries:
                time.sleep(retry_delay)
            else:
                break

    # If we reached here, all attempts failed — return fallback mock (keeps notebook running)
    lg.error("All Gemini LLM attempts failed; returning fallback mock.")
    snippet = final_prompt[:400].replace("\n", " ") + ("..." if len(final_prompt) > 400 else "")
    return f"<<FALLBACK_MOCK>> Summary: {snippet}"

print("call_llm (Gemini) cell installed. Using model 'models/gemini-2.5-flash' by default.")



class ResearcherAgent:
    def __init__(self, name='researcher'):
        self.name = name

    def gather(self, queries: List[str]) -> List[Dict]:
        """Parallel web searches (returns list of result dicts)."""
        logger.info('ResearcherAgent: starting parallel search for %d queries', len(queries))
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1,len(queries)))) as ex:
            futures = {ex.submit(web_search_duckduckgo, q): q for q in queries}
            for fut in concurrent.futures.as_completed(futures):
                q = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    logger.warning('search failed for %s: %s', q, e)
                    r = []
                results.append({'query': q, 'results': r})
                # store a compact note in memory
                memory.add(f"Research ({q}): {json.dumps(r)[:1000]}")
        return results

class AnalyzerAgent:
    def __init__(self, name='analyzer'):
        self.name = name

    def synthesize(self, recall_items:int=10) -> str:
        """Reads memory and synthesizes a short analysis using the LLM placeholder."""
        items = memory.recall()
        if not items:
            return "No research data available."
        bigtext = '\n\n'.join(i['text'] for i in items[-recall_items:])
        prompt = f"Synthesize key points from the following research notes:\n\n{bigtext}"
        resp = call_llm(prompt)
        memory.add('Analyzer synthesis: ' + resp)
        return resp

class WriterAgent:
    def __init__(self, name='writer'):
        self.name = name
        self.checkpoint_path = '/kaggle/working/writer_checkpoint.json'

    def start_write(self, analysis: str) -> Dict:
        """
        Simulate a long-running formatting job by writing a checkpoint.
        The checkpoint acts as the 'paused' state and can be resumed.
        """
        logger.info('WriterAgent: starting write (simulate long-running)')
        ck = {'status': 'paused_for_formatting', 'analysis_len': len(analysis), 'ts': time.time(), 'analysis_preview': analysis[:400]}
        with open(self.checkpoint_path, 'w') as f:
            json.dump(ck, f)
        memory.add('Writer started formatting (checkpoint saved).')
        return ck

    def resume(self) -> str:
        """Resume from checkpoint and produce final report (via LLM)."""
        if not os.path.exists(self.checkpoint_path):
            raise RuntimeError('No checkpoint found at ' + self.checkpoint_path)
        with open(self.checkpoint_path, 'r') as f:
            ck = json.load(f)
        logger.info('WriterAgent: resuming from checkpoint: %s', ck)
        report = call_llm('Use analysis to make final report. Checkpoint meta: ' + json.dumps(ck))
        # remove checkpoint to mark completion
        try:
            os.remove(self.checkpoint_path)
        except Exception:
            pass
        memory.add('Writer produced report')
        return report

class EvaluatorAgent:
    def __init__(self, name='evaluator'):
        self.name = name

    def score(self, report: str) -> float:
        """Simple heuristic scoring: length and keyword presence."""
        score = 5.0
        if len(report) > 200:
            score += 3.0
        if 'Conclusion' in report or 'conclusion' in report.lower():
            score += 1.0
        # penalty if placeholder LLM prefix detected (encourages real LLM)
        if '<<LLM_MOCK>>' in report:
            score -= 1.0
        score = max(0.0, min(score, 10.0))
        logger.info('EvaluatorAgent: scored report %.2f', score)
        memory.add(f'Evaluator scored: {score}')
        return score

print("agent classes ready")



def run_pipeline(topic: str):
    researcher = ResearcherAgent()
    analyzer = AnalyzerAgent()
    writer = WriterAgent()
    evaluator = EvaluatorAgent()

    # 1) Parallel research queries
    queries = [f"{topic} overview", f"{topic} applications", f"{topic} challenges", f"{topic} examples"]
    research_out = researcher.gather(queries)
    logger.info('Research finished. Collected %d query results', len(research_out))

    # 2) Analysis (LLM placeholder)
    analysis = analyzer.synthesize()
    print("\n--- Analysis (truncated) ---\n")
    print(analysis[:800])

    # 3) Writer starts long-running job (checkpoint saved)
    ck = writer.start_write(analysis)
    print("\nWriter paused (checkpoint saved). To continue, run the Resume cell.\nCheckpoint metadata:\n", ck)
    return {'analysis': analysis, 'checkpoint': ck}

# Run pipeline for your topic
state = run_pipeline("multi-agent systems for student education")



# ===== Code cell: Resume writer, evaluate, save report; plus helper to re-run research =====

# Resume writer -> produce final report, evaluate, and save to file
try:
    writer = WriterAgent()
    report = writer.resume()
    print("\n--- FINAL REPORT (first 1200 chars) ---\n")
    print(report[:1200])
except Exception as e:
    print("Resume failed:", e)
    report = None

# Evaluate and save if we have a report
if report:
    evaluator = EvaluatorAgent()
    score = evaluator.score(report)
    print("\nEvaluation score:", score)

    out_path = '/kaggle/working/final_report.txt'
    try:
        with open(out_path, 'w') as f:
            f.write(report)
        print(f"Final report saved to: {out_path}")
    except Exception as ex:
        print("Could not save final report:", ex)

# Helper: re-run research with custom queries (useful if original scraping returned limited content)
def re_run_research(queries):
    """
    Re-run parallel research with a new list of queries (strings).
    This will:
     - run ResearcherAgent.gather(queries)
     - store search results into memory
     - run AnalyzerAgent.synthesize() to update analysis
     - start writer checkpoint again (pause)
    """
    print("Re-running research for queries:", queries)
    researcher = ResearcherAgent()
    analyzer = AnalyzerAgent()
    writer = WriterAgent()

    # Run searches
    research_out = researcher.gather(queries)
    print("Research returned", len(research_out), "items. Sample (first item):")
    if research_out:
        print(research_out[0])

    # New analysis
    analysis = analyzer.synthesize()
    print("\n--- New Analysis (truncated) ---\n")
    print(analysis[:900])

    # Start writer again (checkpoint)
    ck = writer.start_write(analysis)
    print("\nWriter paused (new checkpoint saved). Checkpoint metadata:\n", ck)
    return {'analysis': analysis, 'checkpoint': ck}

# Show last few memory entries to inspect what was saved
print("\n--- Last 6 Memory Items (most recent last) ---")
for item in memory.recall()[-6:]:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item['ts']))
    txt = item['text'][:200].replace('\n', ' ')
    print(f"[{ts}] {txt}...")

print("\nIf search results are empty, try using `re_run_research([...])` with richer queries like:\n"
      "['multi-agent tutoring systems case study', 'intelligent tutoring systems architecture', "
      "'personalized learning multi-agent system examples']")



# Show the full final report and provide a download path
out_path = '/kaggle/working/final_report.txt'
print("Saved report path:", out_path)
print("\n--- FILE CONTENT START ---\n")
with open(out_path, 'r', encoding='utf-8') as f:
    print(f.read()[:5000])   # print first 5000 chars (adjust if you want full)
print("\n--- FILE CONTENT END ---")



# Re-run research with richer queries (uses existing web_search_duckduckgo)
new_queries = [
    "multi-agent tutoring systems case study",
    "intelligent tutoring systems multi-agent architecture examples",
    "multi-agent systems personalized learning review paper",
    "multi-agent system applications in education case studies"
]
state = re_run_research(new_queries)   # re_run_research() was added earlier in notebook
# After this: run the resume/evaluate cell (Cell 7) again to complete pipeline



# Improved search: visit each result URL and extract first paragraph(s)
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup

def fetch_page_snippet(url, max_chars=800):
    """Visit a url and return a short text snippet (first paragraphs)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; agents-capstone/1.0; +https://example.com)"
        }
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        # prefer article or main content
        selectors = ['article', 'main', 'div[class*="content"]', 'div[class*="article"]', 'body']
        text = ""
        for sel in selectors:
            node = soup.select_one(sel)
            if node:
                # get first few <p> tags with non-empty text
                ps = node.find_all("p")
                for p in ps:
                    t = p.get_text().strip()
                    if len(t) > 50:
                        text += t + "\n\n"
                    if len(text) >= max_chars:
                        break
            if len(text) >= max_chars:
                break
        # fallback: collect all <p> from body
        if not text:
            ps = soup.find_all("p")
            for p in ps[:6]:
                t = p.get_text().strip()
                if t:
                    text += t + "\n\n"
                if len(text) >= max_chars:
                    break
        return text.strip()[:max_chars]
    except Exception as e:
        logger.warning("fetch_page_snippet failed for %s : %s", url, e)
        return ""

def web_search_with_snippets(query: str, max_results: int = 5):
    """Run DuckDuckGo search, then visit each result to fetch snippets."""
    items = web_search_duckduckgo(query, max_results=max_results)
    enriched = []
    for it in items:
        url = it.get("url")
        title = it.get("title")
        snippet = ""
        if url:
            # duckduckgo returns redirect URLs sometimes; try to extract real url if possible
            try:
                # many DDG hrefs start with '/l/?kh=-1&uddg=' then encoded url. Try to find 'uddg='
                if "uddg=" in url:
                    import urllib.parse
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get('uddg', [])
                    if q:
                        url = urllib.parse.unquote(q[0])
            except Exception:
                pass
            snippet = fetch_page_snippet(url, max_chars=1000)
            time.sleep(0.5)  # be polite
        enriched.append({'query_title': title, 'url': url, 'snippet': snippet})
        # store richer memory entry
        memory.add(f"Research snippet ({query}): {title} | {url} | {snippet[:800]}")
    return enriched

def re_run_research_with_snippets(queries):
    print("Running enriched research for queries:", queries)
    all_results = {}
    for q in queries:
        res = web_search_with_snippets(q, max_results=4)
        all_results[q] = res
    # run analyzer on enriched memory
    analysis = AnalyzerAgent().synthesize()
    print("\n--- Enriched Analysis (truncated) ---\n")
    print(analysis[:1200])
    # save checkpoint
    ck = WriterAgent().start_write(analysis)
    print("\nWriter paused (checkpoint saved). Checkpoint metadata:\n", ck)
    return all_results

# Example run (uncomment to execute)
# enriched = re_run_research_with_snippets([
#     "multi-agent tutoring systems case study",
#     "intelligent tutoring systems multi-agent architecture examples"
# ])



# Manually add content into memory so analyzer has real text
manual_text = """
Multi-agent tutoring systems distribute responsibilities among specialized agents: a pedagogical agent
that models student knowledge, a pedagogical planner, and a content retrieval agent. Case studies show
improvements in personalization and engagement when these agents coordinate.
"""
memory.add(manual_text)
print("Manual text added. Now run AnalyzerAgent().synthesize() and resume writer.")


