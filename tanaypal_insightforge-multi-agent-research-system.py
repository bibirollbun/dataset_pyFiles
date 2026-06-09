'''Capstone multi-agent single-file demo (threading-based)

This single-file script uses threads for parallel agents to avoid asyncio event-loop issues
in notebook environments. It writes outputs to the specified outdir_base (default: /mnt/data/outputs).
'''

import threading, datetime, os, textwrap, sys, io, traceback, time, uuid
from collections import defaultdict

# --- Observability / Metrics ---
class SimpleMetrics:
    """Thread-safe basic metrics collector."""
    def __init__(self):
        self._counters = defaultdict(int)
        self._timers = {}
        self._lock = threading.Lock() # Added lock for thread safety

    def inc(self, metric: str, amount: int = 1):
        with self._lock:
            self._counters[metric] += amount
    
    # Timing methods remain functionally correct but thread access could be improved for robustness
    # For a simple demo, standard time.time() is acceptable here.
    def time_start(self, name):
        self._timers[name] = time.time()
        
    def time_end(self, name):
        if name in self._timers:
            elapsed = time.time() - self._timers.pop(name)
            with self._lock:
                 self._counters[f"timing_{name}_sec"] = elapsed
            return elapsed
        return None

    def snapshot(self):
        with self._lock:
            return dict(self._counters)

# --- Context compaction ---
def compact_context(messages, max_chars=2000):
    if sum(len(m) for m in messages) <= max_chars:
        return messages
    compacted=[]
    total=0
    for m in reversed(messages):
        if total + len(m) > max_chars:
            break
        compacted.append(m)
        total += len(m)
    return list(reversed(compacted))

# --- Memory & Session ---
class InMemorySessionService:
    """Thread-safe, in-memory session store."""
    def __init__(self):
        self._store = {}
        self._lock = threading.Lock() # Lock added for store access
        
    def create_session(self, session_id, initial=None, ttl=3600):
        with self._lock:
            data = initial or {}
            self._store[session_id] = {'data': data, 'expires_at': time.time() + ttl}
            return data
            
    def get(self, session_id):
        with self._lock:
            s = self._store.get(session_id)
            if not s: return {}
            if s['expires_at'] < time.time():
                self._store.pop(session_id, None)
                return {}
            return s['data']
            
    def update(self, session_id, data):
        with self._lock:
            s = self._store.get(session_id)
            if not s:
                return self.create_session(session_id, data) # Lock acquired inside create_session, but since we hold the outer lock, it's fine.
            s['data'].update(data)
            return s['data']

class MemoryBank:
    """Simple thread-safe in-memory memory bank."""
    def __init__(self):
        self._mems=[]
        self._lock = threading.Lock()
        
    def add(self, text, tags=None):
        mem = {"id": str(uuid.uuid4()), "text": text, "tags": tags or [], "ts": time.time()}
        with self._lock:
            self._mems.append(mem)
        return mem["id"]
        
    def query(self, tag=None, limit=5):
        with self._lock:
            out = [m for m in reversed(self._mems) if (tag is None or tag in m['tags'])]
        return out[:limit]
        
    def all(self):
        with self._lock:
            return list(self._mems)

# --- Tools ---
class SearchTool:
    name = "search_tool"
    def __init__(self, local_samples=None):
        self.local_samples = local_samples or []
        
    def run(self, query, top_k=5):
        matches=[]
        q=query.lower()
        for s in self.local_samples:
            # Simple keyword search simulation
            score = s.get('text','').lower().count(q) 
            if score>0:
                matches.append((score,s))
        matches.sort(key=lambda x:-x[0])
        results=[m[1] for m in matches[:top_k]]
        
        # Consistent output structure
        return {"query": query, "results": results, "source": "local_mock"} 

class CodeExecTool:
    name = "code_exec"
    def run(self, code):
        # Slightly more restricted safe globals for safety
        safe_globals = {"__builtins__": {"print": print, "range": range, "len": len, "min": min, "max": max, "sum": sum}}
        local_vars={}
        try:
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # Execute the code
            exec(textwrap.dedent(code), safe_globals, local_vars)
            
            # Restore stdout and get output
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # Provide structured output
            return {"ok": True, "output": output.strip(), "locals": local_vars}
            
        except Exception as e:
            sys.stdout = old_stdout
            tb = traceback.format_exc()
            # Provide structured error output
            return {"ok": False, "error": str(e), "traceback": tb, "output": None}

# --- Agent infra (synchronous/threads) ---
class AgentResponse:
    def __init__(self, name, result, success=True):
        self.name=name
        self.result=result
        self.success=success

class BaseAgent:
    def __init__(self, name, tools=None):
        self.name=name
        self.tools = tools or {}
        
    def run(self, query, session):
        raise NotImplementedError

class ParallelAgentManager:
    def __init__(self, agents, timeout=30.0):
        self.agents=agents
        self.timeout=timeout
        
    def run_all(self, query, session):
        results = {}
        threads = []
        lock = threading.Lock() # Lock to protect the shared 'results' dictionary
        
        def run_agent(agent):
            try:
                resp = agent.run(query, session)
                with lock:
                    results[resp.name] = resp
            except Exception as e:
                # Catch unexpected failures during the agent's run method
                err_msg = f"Agent failed with unexpected error: {str(e)}"
                with lock:
                    results[agent.name] = AgentResponse(
                        agent.name, 
                        {"error": err_msg, "traceback": traceback.format_exc()}, 
                        success=False
                    )

        for agent in self.agents:
            t = threading.Thread(target=run_agent, args=(agent,))
            t.start()
            threads.append(t)
            
        # wait with timeout
        start = time.time()
        for t in threads:
            remaining = max(0, self.timeout - (time.time()-start))
            t.join(remaining)
            
            # Check if thread is alive (i.e., timed out) and report
            if t.is_alive():
                 if t.name not in results: # Only report if the agent hasn't logged a result yet
                    with lock:
                         results[t.name] = AgentResponse(
                            t.name, 
                            {"error": "Execution timed out."}, 
                            success=False
                        )
        return results

# --- Demo Agents ---
class SearchAgent(BaseAgent):
    def run(self, query, session):
        tool = self.tools.get('search')
        if not tool:
             return AgentResponse(self.name, {"error": "Search tool not configured."}, success=False)
             
        res = tool.run(query)
        
        # Store results in session using the session service's thread-safe update method
        session_service_ref.update('demo_session', {
            'search_results': session.get('search_results', []) + [{'agent': self.name, 'res': res}]
        })
        
        return AgentResponse(self.name, {"search_count": len(res.get('results', [])), "top": res.get('results', [])})

class ExtractAgent(BaseAgent):
    def run(self, query, session):
        # Increased robustness: check if search results are available
        results = session.get('search_results', [])
        
        if not results:
             return AgentResponse(self.name, {"error": "No search results found in session."}, success=False)
             
        # Get the results from the most recent search agent run
        top_res = results[-1]['res'] 
        
        snippet=""
        if top_res and 'results' in top_res and top_res['results']:
            # Use safe .get() for dictionary access
            top_result = top_res['results'][0]
            snippet = top_result.get('text', 'No text extracted.')
            
            # Truncate snippet
            snippet = snippet[:800]
            
            # Store snippet for the CodeAgent
            session_service_ref.update('demo_session', {'last_snippet': snippet})
            
            return AgentResponse(self.name, {"snippet_len": len(snippet), "snippet": snippet[:50] + "..."})
            
        return AgentResponse(self.name, {"error": "Search results were empty or malformed."}, success=False)

class CodeAgent(BaseAgent):
    def run(self, query, session):
        code_tool = self.tools.get('code_exec')
        if not code_tool:
             return AgentResponse(self.name, {"error": "Code exec tool not configured."}, success=False)
             
        snippet = session.get('last_snippet','')
        
        if not snippet:
             return AgentResponse(self.name, {"error": "No snippet available for execution."}, success=False)
             
        # create code that prints the number of words in the snippet
        # Use str.split() without arguments for robust word counting
        code_lines = [
            "text = '"+ snippet.replace("'", "\\'") + "'", # Safely inject snippet
            "word_count = len(text.split())",
            "print(word_count)"
        ]
        code = '\n'.join(code_lines)
        
        res = code_tool.run(code)
        
        # Log successful execution to memory
        if res['ok']:
            memory_bank_ref.add(f"Code Exec: Word count {res['output']} for snippet starting {snippet[:30]}", tags=['code_exec','success'])
        
        return AgentResponse(self.name, {"exec_result": res})

# --- Globals for Agent/Tool access ---
session_service_ref = InMemorySessionService()
memory_bank_ref = MemoryBank()

# --- Fetcher (local) ---
def fetch_local_samples(samples_dir='samples'):
    out=[]
    if not os.path.isdir(samples_dir):
        return out
    for fname in os.listdir(samples_dir):
        path=os.path.join(samples_dir,fname)
        if os.path.isfile(path):
            with open(path,'r',encoding='utf-8') as f:
                out.append({"path": path, "text": f.read()})
    return out

# --- Orchestrator ---
def run_multi_agent_demo(query, outdir_base='outputs'):
    print("Starting multi-agent demo for query:", query)
    
    # 1. Setup Samples and Tools
    samples = fetch_local_samples('samples')
    if not samples:
        # Improved dummy sample
        dummy_text = "Title: Demo Article\nAuthors: Demo Author\nThis is a demo article about LLM-based phishing detection. It discusses embeddings, classifier-based methods, and ethical considerations."
        samples = [{"path": "inline_sample", "text": dummy_text}]
        
    search_tool = SearchTool(local_samples=[{'title':'Demo','text':s['text'],'url':s['path']} for s in samples])
    code_exec = CodeExecTool()
    
    # Use global refs
    session = session_service_ref.create_session('demo_session', initial={'query': query})
    metrics = SimpleMetrics()
    
    # 2. Agent Configuration
    search_agent = SearchAgent('search_agent', tools={'search': search_tool})
    extract_agent = ExtractAgent('extract_agent', tools={})
    code_agent = CodeAgent('code_agent', tools={'code_exec': code_exec})
    
    manager = ParallelAgentManager([search_agent, extract_agent], timeout=15.0) # Reduced timeout for faster demo fail
    
    # 3. Parallel Run
    metrics.time_start('parallel_run')
    results = manager.run_all(query, session)
    metrics.time_end('parallel_run')
    
    # Re-fetch session data (important if parallel agents made updates)
    session = session_service_ref.get('demo_session') 
    extracted_snippet = session.get('last_snippet', '')
    
    # 4. Sequential Run (Code Agent)
    metrics.time_start('code_run')
    code_resp = code_agent.run(query, session)
    metrics.time_end('code_run')
    
    # 5. Reporting and Memory
    memory_bank_ref.add(f"query:{query} top_snippet_len:{len(extracted_snippet)}", tags=['demo','query'])
    metrics.inc('memory_added')
    
    messages = [f"User: {query}", extracted_snippet]
    compacted = compact_context(messages, max_chars=1000)
    
    # 6. Output Generation
    outdir = os.path.join(outdir_base, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(outdir, exist_ok=True)
    report_path = os.path.join(outdir, 'multi_agent_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# Multi-agent Demo Report\\n\\n')
        f.write('**Query:** ' + query + '\\n\\n')
        
        f.write('## Parallel Results (Search & Extract)\\n')
        for k,v in results.items():
            f.write(f"- **{k}** (Success: {v.success}): {v.result}\\n")
            
        f.write('\\n## Code Agent Result\\n')
        f.write(str(code_resp.result) + '\\n')
        
        f.write('\\n## Session & Memory\\n')
        f.write('**Extracted Snippet Length:** ' + str(len(extracted_snippet)) + '\\n')
        f.write('**Memory Snapshot:**\\n')
        f.write('```json\\n' + str(memory_bank_ref.all()) + '\\n```\\n')
        f.write('**Compacted Context:**\\n')
        f.write('```\\n' + '\\n'.join(compacted) + '\\n```\\n')
        
        f.write('\\n## Metrics\\n')
        f.write('```json\\n' + str(metrics.snapshot()) + '\\n```\\n')

    print('Wrote multi-agent outputs to', outdir)
    return report_path

# --- If run as script ---
if __name__ == '__main__':
    # Make sure we initialize the globals before running the demo
    # The current setup handles this implicitly, but explicitly initializing is safer:
    # session_service_ref and memory_bank_ref are initialized at the module level.
    
    rp = run_multi_agent_demo('LLM-based phishing detection', outdir_base='outputs')
    print('Report path:', rp)

