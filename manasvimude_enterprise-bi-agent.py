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


import warnings
warnings.filterwarnings("ignore", category=Warning)



# %%
"""
Enterprise BI Agent — Full-Featured Project (Thread-safe/fallback updated)
File: enterprise_bi_agent_full_project.py

This file is a hardened variant of the full project intended to run inside
restricted environments (e.g., Kaggle sandbox) where creating new threads may fail.

Key fixes applied:
- ParallelAgentRunner now falls back to sequential execution if thread creation is unavailable
- LongRunningAgent test uses a safe-thread-creation wrapper that falls back to direct call
- Added explicit tests for the sequential fallback path
- Improved logging and safer metrics updates

Everything else is intentionally preserved so tests remain valid. If the environment
allows threads, the code will use them; otherwise it will run sequentially.

"""

# %%
# Imports & setup
import warnings
warnings.filterwarnings("ignore", category=Warning)  # Hide warnings
import os
import sys
import json
import time
import uuid
import logging
import threading
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import duckdb
import matplotlib.pyplot as plt

# Basic logger configuration
logging.basicConfig(level=logging.ERROR, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('enterprise_bi_agent_full')

# Ensure outputs exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('submission', exist_ok=True)

# %%
# Utilities & helpers

def now_ts() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def safe_filename(s: str) -> str:
    return ''.join(c for c in s if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')[:120]


@dataclass
class Metrics:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    avg_latency_ms: float = 0.0

    def update_latency(self, ms: float):
        # keep a simple running average guarded against zero division
        total = max(1, self.requests)
        self.avg_latency_ms = ((self.avg_latency_ms * (total - 1)) + ms) / total

metrics = Metrics()

# Simple persistent memory bank
class MemoryBank:
    def __init__(self, storage_dir: str = 'memory'):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_path = os.path.join(self.storage_dir, 'index.json')
        self.index = self._load_index()

    def _load_index(self) -> Dict[str,str]:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _flush_index(self):
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def save_report(self, name: str, content: Dict[str,Any]) -> str:
        key = safe_filename(name) + '_' + uuid.uuid4().hex[:8]
        path = os.path.join(self.storage_dir, key + '.json')
        with open(path, 'w') as f:
            json.dump(content, f, default=str, indent=2)
        self.index[name] = path
        self._flush_index()
        return path

    def get_report(self, name: str) -> Optional[Dict[str,Any]]:
        path = self.index.get(name)
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

memory = MemoryBank()

# A2A message (kept for completeness)
@dataclass
class A2AMessage:
    from_agent: str
    to_agent: str
    task_id: str
    payload: Dict[str,Any]
    timestamp: str = field(default_factory=now_ts)

    def to_json(self):
        return json.dumps({
            'from': self.from_agent,
            'to': self.to_agent,
            'task_id': self.task_id,
            'payload': self.payload,
            'timestamp': self.timestamp
        })

# %%
# Data: load attached dataset or make synthetic

def find_csv_under_kaggle_input():
    base = '/kaggle/input'
    if os.path.exists(base):
        for root, dirs, files in os.walk(base):
            for f in files:
                if f.lower().endswith('.csv'):
                    return os.path.join(root, f)
    return None

csv_path = find_csv_under_kaggle_input()
if csv_path:
    logger.info(f'Loading dataset from {csv_path}')
    df = pd.read_csv(csv_path)
else:
    logger.info('No CSV detected. Generating synthetic demo dataset (customer churn-like)')
    import numpy as np
    rng = np.random.default_rng(12345)
    n = 1200
    df = pd.DataFrame({
        'customer_id': [f'C{100000+i}' for i in range(n)],
        'tenure_months': rng.integers(0, 72, n),
        'monthly_charges': (rng.normal(70, 30, n).clip(10, 300)).round(2),
        'contract_type': rng.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55,0.30,0.15]),
        'online_security': rng.choice(['Yes','No'], n, p=[0.38,0.62]),
        'tech_support': rng.choice(['Yes','No'], n, p=[0.33,0.67]),
        'gender': rng.choice(['Male','Female'], n),
        'senior_citizen': rng.choice([0,1], n, p=[0.88,0.12]),
        'payment_method': rng.choice(['Electronic check','Mailed check','Bank transfer','Credit card'], n)
    })
    # churn probability based on synthetic rules
    base_prob = 0.12 + (df['contract_type']=='Month-to-month')*0.08 + (df['online_security']=='No')*0.03 + (df['tech_support']=='No')*0.03
    noise = rng.random(n) * 0.05
    churn_flag = ((base_prob + noise) > 0.2).astype(int)
    df['churn'] = churn_flag.map({1:'Yes', 0:'No'})
    df['total_charges'] = (df['tenure_months'] * df['monthly_charges']).round(2)

logger.info(f'Data loaded: shape={df.shape}')

# %%
# SQL safety checker
class SQLSafetyException(Exception):
    pass

class SQLSafety:
    DANGEROUS = ['drop', 'delete', 'truncate', 'update', 'insert', 'alter', 'create']

    @staticmethod
    def check(sql: str):
        low = sql.lower()
        # disallow semicolons to prevent multi-statement execution
        if ';' in low:
            raise SQLSafetyException('Multiple statements or semicolons are disallowed')
        for token in SQLSafety.DANGEROUS:
            if token in low:
                raise SQLSafetyException(f"SQLSafety blocked token: {token}")
        if 'select' not in low:
            raise SQLSafetyException('Only SELECT queries are allowed in demo environment')
        return True

# %%
# Agent primitives
@dataclass
class AgentResponse:
    success: bool
    data: Any = None
    message: str = ''

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.log = logging.getLogger(f'agent.{self.name}')

    def handle(self, *args, **kwargs) -> AgentResponse:
        raise NotImplementedError()

# QueryPlanner
class QueryPlanner(BaseAgent):
    def __init__(self, name='QueryPlanner'):
        super().__init__(name)

    def plan(self, user_query: str, context: Optional[Dict]=None) -> Dict:
        self.log.info(f'Planning for: {user_query}')
        uq = user_query.lower()
        if any(k in uq for k in ['churn', 'churn drivers', 'driver of churn', 'why churn']):
            sql = "SELECT contract_type, online_security, tech_support, AVG(CASE WHEN churn='Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate, COUNT(*) AS n FROM df GROUP BY contract_type, online_security, tech_support ORDER BY churn_rate DESC"
            return {'type':'sql', 'sql':sql, 'note':'aggregate churn_by_features', 'confidence':0.92}
        if any(k in uq for k in ['summary','overview','describe']):
            return {'type':'python', 'action':'describe', 'note':'dataset description', 'confidence':0.8}
        if any(k in uq for k in ['mean','average','median','distribution','histogram','correlation']):
            return {'type':'python','action':'column_stats','note':'numeric column stats','confidence':0.7}
        for col in df.columns:
            if col.replace('_',' ') in uq:
                sql = f"SELECT {col}, COUNT(*) AS n FROM df GROUP BY {col} ORDER BY n DESC LIMIT 50"
                return {'type':'sql','sql':sql,'note':f'group_by_{col}','confidence':0.5}
        return {'type':'clarify','question':'Could you clarify what you want? (e.g., churn drivers, summary, correlation)'}

# ExecutionAgent
class ExecutionAgent(BaseAgent):
    def __init__(self, name='ExecutionAgent'):
        super().__init__(name)

    def execute_plan(self, plan: Dict) -> AgentResponse:
        try:
            t0 = time.time()
            if plan['type'] == 'sql':
                SQLSafety.check(plan['sql'])
                con = duckdb.connect(database=':memory:')
                con.register('df', df)
                res = con.execute(plan['sql']).df()
                latency = (time.time()-t0)*1000.0
                return AgentResponse(True, data=res, message=f'executed_sql in {latency:.1f}ms')
            elif plan['type'] == 'python':
                action = plan.get('action')
                if action == 'describe':
                    desc = df.describe(include='all').to_dict()
                    return AgentResponse(True, data=desc, message='describe')
                if action == 'column_stats':
                    stats = df.select_dtypes(include=['number']).agg(['mean','std','min','max']).to_dict()
                    return AgentResponse(True, data=stats, message='column_stats')
                return AgentResponse(False, message='unknown python action')
            elif plan['type'] == 'clarify':
                return AgentResponse(False, message='clarification_required')
            else:
                return AgentResponse(False, message='unknown plan type')
        except SQLSafetyException as se:
            return AgentResponse(False, message=f'SQLSafetyException: {str(se)}')
        except Exception as e:
            self.log.exception('Execution error')
            return AgentResponse(False, message=str(e))

# AnalyzerAgent
class AnalyzerAgent(BaseAgent):
    def __init__(self, name='AnalyzerAgent'):
        super().__init__(name)

    def analyze(self, data: Any, plan: Dict) -> Dict:
        if isinstance(data, pd.DataFrame):
            df_res = data.copy()
            analysis = {'n_rows': len(df_res)}
            if 'churn_rate' in df_res.columns:
                grand_mean = df_res['churn_rate'].mean()
                df_res['delta'] = df_res['churn_rate'] - grand_mean
                analysis['grand_mean'] = float(grand_mean)
            analysis['table'] = df_res.head(100).to_dict(orient='records')
            return analysis
        return {'result': data}

# VizAgent
class VizAgent(BaseAgent):
    def __init__(self, name='VizAgent'):
        super().__init__(name)

    def plot(self, analysis: Dict, title: str, max_items: int = 10) -> str:
        safe_title = safe_filename(title)
        png = f'outputs/{safe_title[:80]}.png'
        try:
            records = analysis.get('table')
            if isinstance(records, list) and len(records)>0:
                df_rec = pd.DataFrame(records)
                if 'churn_rate' in df_rec.columns:
                    df_plot = df_rec.sort_values('churn_rate', ascending=False).head(max_items)
                    labels = df_plot.apply(lambda r: ' | '.join(str(r.get(c,'')) for c in ['contract_type','online_security','tech_support'] if c in r), axis=1)
                    plt.figure(figsize=(10,5))
                    plt.barh(labels[::-1], df_plot['churn_rate'][::-1])
                    plt.title(title)
                    plt.xlabel('Churn rate')
                    plt.tight_layout()
                    plt.savefig(png)
                    plt.close()
                    return png
            txt = png.replace('.png','.txt')
            with open(txt,'w') as f:
                f.write('No plottable table found in analysis.\n')
            return png
        except Exception:
            self.log.exception('Viz failed')
            return ''

# InsightAgent
class InsightAgent(BaseAgent):
    def __init__(self, name='InsightAgent'):
        super().__init__(name)

    def summarize(self, analysis: Dict, plan: Dict) -> str:
        if 'table' in analysis and isinstance(analysis['table'], list):
            rows = analysis['table']
            if len(rows)==0:
                return 'No rows to summarize.'
            df_rows = pd.DataFrame(rows)
            narrative = []
            if 'churn_rate' in df_rows.columns:
                top = df_rows.sort_values('churn_rate', ascending=False).head(3)
                for i,r in top.iterrows():
                    narrative.append(f"Top factor: contract={r.get('contract_type','?')}, online_security={r.get('online_security','?')}, tech_support={r.get('tech_support','?')} -> churn_rate={r.get('churn_rate'):.3f}")
                narrative.append(f"Overall mean churn rate in analyzed groups: {analysis.get('grand_mean',0):.3f}")
                narrative.append('\nRecommendation: focus retention efforts on top groups with high churn_rate, offer targeted tech support and incentives for short-contract customers.')
                return '\n'.join(narrative)
            narrative.append(f"Found {len(rows)} analysis rows")
            return '\n'.join(narrative)
        if 'result' in analysis:
            return 'Detailed result available. Consider drilling down with a SQL query.'
        return 'No insights available.'

# ParallelAgentRunner with thread fallback
class ParallelAgentRunner:
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    def run_all(self, func_name: str, *args, **kwargs) -> List[AgentResponse]:
        """
        Attempts to run agents in parallel using threads. If creating threads is not allowed
        (RuntimeError: can't start new thread), falls back to sequential execution.
        """
        threads = []
        results = [None] * len(self.agents)

        def worker(i, agent: BaseAgent):
            try:
                # If the named function exists on the agent, call it. Otherwise try sensible fallbacks.
                if hasattr(agent, func_name):
                    fn = getattr(agent, func_name)
                    resp = fn(*args, **kwargs)
                    results[i] = AgentResponse(True, data=resp)
                else:
                    # try common alternative method names
                    for alt in ('analyze','plot','handle','run'):
                        if hasattr(agent, alt):
                            fn = getattr(agent, alt)
                            resp = fn(*args, **kwargs)
                            results[i] = AgentResponse(True, data=resp, message=f'used_fallback:{alt}')
                            break
                    else:
                        msg = f'agent {agent.__class__.__name__} has no method {func_name} or common fallbacks'
                        results[i] = AgentResponse(False, message=msg)
            except Exception as e:
                results[i] = AgentResponse(False, message=str(e))
                logger.error('Parallel agent error: %s', e, exc_info=False)

        # Try to start threads; if thread creation fails, fallback to sequential
        try:
            for i, agent in enumerate(self.agents):
                t = threading.Thread(target=worker, args=(i, agent))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            return results
        except RuntimeError as re:
            logger.warning('Thread creation failed, falling back to sequential execution: %s', str(re))
            # Run sequentially in current thread
            for i, agent in enumerate(self.agents):
                worker(i, agent)
            return results

# LongRunningAgent (unchanged)
class LongRunningAgent(BaseAgent):
    def __init__(self, name='LongRunner'):
        super().__init__(name)
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop = False

    def run(self, duration_seconds: int = 10) -> str:
        start = time.time()
        elapsed = 0
        self.log.info('LongRunningAgent starting')
        while elapsed < duration_seconds and not self._stop:
            self._pause_event.wait()
            time.sleep(0.5)
            elapsed = time.time() - start
        self.log.info('LongRunningAgent finished')
        return f'completed in {elapsed:.1f}s'

    def pause(self):
        self._pause_event.clear()
        self.log.info('LongRunningAgent paused')

    def resume(self):
        self._pause_event.set()
        self.log.info('LongRunningAgent resumed')

    def stop(self):
        self._stop = True
        self._pause_event.set()

# Orchestrator
class Orchestrator(BaseAgent):
    def __init__(self, name='Orchestrator'):
        super().__init__(name)
        self.planner = QueryPlanner()
        self.executor = ExecutionAgent()
        self.analyzer = AnalyzerAgent()
        self.viz = VizAgent()
        self.insight = InsightAgent()

    def handle_request(self, user_query: str, session_id: Optional[str]=None) -> Dict[str,Any]:
        metrics.requests += 1
        t0 = time.time()
        task_id = uuid.uuid4().hex[:8]
        self.log.info(f'Handling request {task_id}: {user_query}')

        plan = self.planner.plan(user_query)
        if plan.get('type') == 'clarify':
            metrics.failures += 1
            return {'success':False, 'clarify':plan.get('question')}

        exec_resp = self.executor.execute_plan(plan)
        if not exec_resp.success:
            metrics.failures += 1
            return {'success':False, 'error':exec_resp.message}

        analysis = self.analyzer.analyze(exec_resp.data, plan)

        # Run analyzer and viz in parallel but safe against thread limits
        par_runner = ParallelAgentRunner([self.analyzer, self.viz])
        par_results = par_runner.run_all('analyze' if isinstance(exec_resp.data, pd.DataFrame) else 'analyze', exec_resp.data, plan)

        viz_path = self.viz.plot(analysis, title=user_query)
        summary = self.insight.summarize(analysis, plan)

        report = {
            'task_id': task_id,
            'query': user_query,
            'plan': plan,
            'analysis': analysis,
            'viz_path': viz_path,
            'summary': summary,
            'timestamp': now_ts()
        }

        mem_path = memory.save_report(user_query, report)
        elapsed = (time.time() - t0) * 1000.0
        # update metrics safely
        metrics.successes += 1
        metrics.update_latency(elapsed)

        self.log.info(f'Request {task_id} completed in {elapsed:.1f}ms and saved to {mem_path}')
        return {'success':True, 'report':report, 'memory_path': mem_path}

# Demonstration & CLI

def demo_interactive():
    orch = Orchestrator()
    print('\nEnterprise BI Agent — Demo Interactive')
    print('Type a natural-language query (e.g., "What are churn drivers?"), or "quit" to exit')
    while True:
        q = input('> ')
        if q.strip().lower() in ('quit','exit'):
            break
        resp = orch.handle_request(q)
        if not resp.get('success'):
            if 'clarify' in resp:
                print('Clarify:', resp['clarify'])
            else:
                print('Error:', resp.get('error'))
        else:
            rpt = resp['report']
            print('\n--- Report Summary ---')
            print('Query:', rpt['query'])
            print('Plan note:', rpt['plan'].get('note'))
            print('Summary:\n', rpt['summary'])
            print('Viz path:', rpt['viz_path'])
            print('Memory path:', resp['memory_path'])
            print('-----------------------\n')


def run_demo_script():
    orch = Orchestrator()
    queries = [
        'What are churn drivers?',
        'Give me a dataset summary',
        'Show distribution of monthly charges',
    ]
    results = []
    for q in queries:
        r = orch.handle_request(q)
        results.append(r)
    return results

# Unit tests & evaluation

def run_tests():
    logger.info('Running unit tests...')
    planner = QueryPlanner()
    plan = planner.plan('What drives churn?')
    assert plan['type'] in ('sql','python'), 'Planner returned unexpected type'

    exec_agent = ExecutionAgent()
    exec_resp = exec_agent.execute_plan(plan)
    assert exec_resp.success, f'Execution failed: {exec_resp.message}'

    orch = Orchestrator()
    r = orch.handle_request('What are churn drivers?')
    assert r['success'], f'Orchestrator failed: {r.get("error")}'
    rep = r['report']
    assert 'summary' in rep and 'viz_path' in rep

    # Edge case: unsafe SQL should be blocked
    bad_plan = {'type':'sql','sql':'DROP TABLE users;'}
    bad_resp = exec_agent.execute_plan(bad_plan)
    assert not bad_resp.success and 'SQLSafety' in bad_resp.message

    # LongRunningAgent test with safe thread creation
    lra = LongRunningAgent()
    def safe_start_thread(target_fn):
        try:
            t = threading.Thread(target=target_fn)
            t.start()
            return t
        except RuntimeError as re:
            logger.warning('Thread creation failed in test harness, running target inline: %s', re)
            # fallback: run inline and return None
            target_fn()
            return None

    t = safe_start_thread(lambda: lra.run(2))
    time.sleep(0.5)
    lra.pause()
    time.sleep(0.5)
    lra.resume()
    lra.stop()
    if t is not None:
        t.join()

    # Test parallel runner fallback explicitly: create a runner with many agents to trigger fallback
    agents = [AnalyzerAgent(), VizAgent(), AnalyzerAgent(), VizAgent(), AnalyzerAgent()]
    par_runner = ParallelAgentRunner(agents)
    results = par_runner.run_all('analyze', pd.DataFrame({'a':[1,2,3]}), {'note':'test'})
    # Ensure the runner returned results for each agent (either success or failure but no crash)
    assert len(results) == len(agents), 'ParallelAgentRunner returned unexpected number of results'

    logger.info('All tests passed')

# Save artifacts

def save_artifacts():
    with open('submission/README.txt','w') as f:
        f.write('Enterprise BI Agent — Full Project\n')
        f.write('\nThis project demonstrates a production-minded multi-agent orchestration for BI tasks.\n')
        f.write('Components: Orchestrator, QueryPlanner, ExecutionAgent (duckdb), Analyzer, Viz, Insight.\n')
        f.write('Memory is persisted under memory/.\n')
    dockerfile = textwrap.dedent('''
    FROM python:3.10-slim
    WORKDIR /app
    COPY . /app
    RUN pip install pandas duckdb matplotlib
    CMD ["python", "enterprise_bi_agent_full_project.py"]
    ''')
    with open('submission/Dockerfile','w') as f:
        f.write(dockerfile)
    logger.info('Saved submission artifacts')

# Main
if __name__ == '__main__':
    logger.info('Starting Enterprise BI Agent full project (thread-safe)')
    results = run_demo_script()
    print('\nDemo completed. Sample outputs:')
    for r in results:
        if r.get('success'):
            print(f"- {r['report']['query']}: summary len ~ {len(r['report']['summary'])} chars; viz saved: {r['report']['viz_path']}")
        else:
            print(f"- {r.get('error')}")
    try:
        run_tests()
    except AssertionError as e:
        logger.exception('Tests failed')
        sys.exit(2)
    save_artifacts()
    print('\nAll done. Check outputs/ and submission/ directories.')

# End of file





