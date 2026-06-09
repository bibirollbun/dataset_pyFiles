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


# Dependencies (uncomment to install if needed on other platforms)

import os
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('AutoDataScientist')

# Metrics collector (thread-safe counters)
class Metrics:
    def __init__(self):
        self._counters = {}
        self._lock = threading.Lock()
    def incr(self, name: str, amount: int = 1):
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount
    def get(self, name: str):
        return self._counters.get(name, 0)
    def snapshot(self):
        with self._lock:
            return dict(self._counters)

metrics = Metrics()


class InMemorySessionService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def create_session(self, session_id: str):
        with self.lock:
            self.sessions[session_id] = {"created_at": time.time(), "state": {}}
            logger.info(f"Session created: {session_id}")

    def get_state(self, session_id: str):
        return self.sessions.get(session_id, {}).get('state')

    def update_state(self, session_id: str, key: str, value: Any):
        with self.lock:
            self.sessions.setdefault(session_id, {"created_at": time.time(), "state": {}})
            self.sessions[session_id]['state'][key] = value
            logger.info(f"Session {session_id} state updated: {key}")

class MemoryBank:
    def __init__(self, path: str = '/kaggle/working/autods_memory.jsonl'):
        self.path = path
        open(self.path, 'a').close()
        self.lock = threading.Lock()

    def add(self, key: str, value: Any):
        record = {'ts': time.time(), 'key': key, 'value': value}
        with self.lock:
            with open(self.path, 'a') as f:
                f.write(json.dumps(record) + '\n')
        metrics.incr('memory_add')

    def query_recent(self, n: int = 10):
        with self.lock:
            with open(self.path, 'r') as f:
                lines = f.readlines()[-n:]
        return [json.loads(l) for l in lines]



from functools import lru_cache

@lru_cache(maxsize=1024)
def compact_context(messages: str, max_tokens: int = 512) -> str:
    if len(messages) <= max_tokens:
        return messages
    head = messages[: max_tokens // 3]
    tail = messages[-(max_tokens // 3):]
    mid = f"...[{len(messages) - len(head) - len(tail)} chars omitted]..."
    return head + mid + tail


class CodeExecutionTool:
    def __init__(self):
        pass
    def run(self, code: str) -> Dict[str, Any]:
        safe_globals = {"__builtins__": {"range": range, "len": len, "print": print, "min": min, "max": max}}
        local_vars = {}
        try:
            exec(code, safe_globals, local_vars)
            metrics.incr('code_exec_success')
            return {"ok": True, "locals": {k: repr(v) for k, v in local_vars.items()}}
        except Exception as e:
            metrics.incr('code_exec_error')
            return {"ok": False, "error": str(e)}

class LLMToolStub:
    def __init__(self, persona: str = 'assistant'):
        self.persona = persona
    def generate(self, prompt: str) -> str:
        metrics.incr('llm_calls')
        return f"[LLM-{self.persona}] Received prompt (first 120 chars): {prompt[:120]}"


class Message:
    def __init__(self, sender: str, receiver: str, payload: dict):
        self.sender = sender
        self.receiver = receiver
        self.payload = payload

class Agent:
    def __init__(self, name: str, llm: Optional[LLMToolStub] = None, tools: Optional[Dict[str, Any]] = None):
        self.name = name
        self.llm = llm or LLMToolStub(persona=name)
        self.tools = tools or {}
        self.inbox: List[Message] = []
        self.lock = threading.Lock()

    def send(self, msg: Message, other: 'Agent'):
        logger.info(f"{self.name} -> {other.name}: {msg.payload.get('type')}")
        other.receive(msg)
        metrics.incr('a2a_messages')

    def receive(self, msg: Message):
        with self.lock:
            self.inbox.append(msg)
            logger.info(f"{self.name} received message from {msg.sender}")

    def act(self, session_id: str, session_service: InMemorySessionService, memory_bank: MemoryBank) -> Dict[str, Any]:
        with self.lock:
            messages = [m.payload for m in self.inbox]
            self.inbox.clear()
        prompt = json.dumps(messages)
        prompt = compact_context(prompt, max_tokens=800)
        output = self.llm.generate(prompt)
        memory_bank.add(f"agent:{self.name}:{session_id}", {"prompt_len": len(prompt)})
        logger.info(f"{self.name} acted for session {session_id}")
        return {"agent": self.name, "output": output}


class DataAgent(Agent):
    def __init__(self, name='DataAgent'):
        super().__init__(name)
    def act(self, session_id: str, session_service: InMemorySessionService, memory_bank: MemoryBank):
        iris = datasets.load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        summary = {
            'n_samples': df.shape[0],
            'n_features': df.shape[1] - 1,
            'target_counts': df['target'].value_counts().to_dict(),
            'feature_means': df.drop('target', axis=1).mean().to_dict()
        }
        out_dir = '/kaggle/working/autods_outputs'
        os.makedirs(out_dir, exist_ok=True)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
        cols = df.columns[:-1]
        for ax, col in zip(axes.flatten(), cols):
            ax.hist(df[col], bins=15)
            ax.set_title(col)
        fig.suptitle('Feature distributions')
        hist_path = os.path.join(out_dir, 'feature_histograms.png')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(hist_path)
        plt.close(fig)
        fig2 = plt.figure(figsize=(8,8))
        scatter_matrix(df[cols], diagonal='kde')
        sm_path = os.path.join(out_dir, 'scatter_matrix.png')
        plt.suptitle('Scatter matrix (features)')
        plt.savefig(sm_path)
        plt.close()
        memory_bank.add(f"dataset_summary:{session_id}", summary)
        logger.info(f"{self.name} produced dataset summary and saved EDA charts to {out_dir}")
        return {"agent": self.name, "summary": summary, "eda_paths": [hist_path, sm_path]}

class ModelingAgent(Agent):
    def __init__(self, name='ModelingAgent', tools: Optional[Dict[str, Any]] = None):
        super().__init__(name, tools=tools)
    def act(self, session_id: str, session_service: InMemorySessionService, memory_bank: MemoryBank):
        iris = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        out_dir = '/kaggle/working/autods_outputs'
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, 'rf_model.joblib')
        joblib.dump(clf, model_path)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6,6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
        disp.plot(ax=ax_cm)
        cm_path = os.path.join(out_dir, 'confusion_matrix.png')
        fig_cm.suptitle(f'Confusion Matrix (acc={acc:.3f})')
        fig_cm.savefig(cm_path)
        plt.close(fig_cm)
        importances = clf.feature_importances_
        fig_fi, ax_fi = plt.subplots(figsize=(6,4))
        ax_fi.bar(range(len(importances)), importances)
        ax_fi.set_xticks(range(len(importances)))
        ax_fi.set_xticklabels(iris.feature_names, rotation=45, ha='right')
        ax_fi.set_title('Feature importances')
        fi_path = os.path.join(out_dir, 'feature_importances.png')
        fig_fi.tight_layout()
        fig_fi.savefig(fi_path)
        plt.close(fig_fi)
        result = {'accuracy': acc, 'model_path': model_path, 'cm_path': cm_path, 'fi_path': fi_path}
        memory_bank.add(f"model_result:{session_id}", result)
        logger.info(f"{self.name} finished model training; artifacts saved to {out_dir}")
        return {"agent": self.name, "result": result}

class EvaluatorAgent(Agent):
    def __init__(self, name='EvaluatorAgent'):
        super().__init__(name)
    def act(self, session_id: str, session_service: InMemorySessionService, memory_bank: MemoryBank):
        recent = memory_bank.query_recent(10)
        model_results = [r['value'] for r in recent if r['key'].startswith('model_result')]
        evaluations = []
        for m in model_results:
            evaluations.append({'accuracy': m.get('accuracy'), 'model_path': m.get('model_path')})
        logger.info(f"{self.name} evaluated models: found {len(evaluations)} results")
        return {"agent": self.name, "evaluations": evaluations}


class AgentOrchestrator:
    def __init__(self, agents: List[Agent], session_service: InMemorySessionService, memory_bank: MemoryBank):
        self.agents = {a.name: a for a in agents}
        self.session_service = session_service
        self.memory_bank = memory_bank

    def run_parallel(self, session_id: str) -> Dict[str, Any]:
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.agents)) as ex:
            futures = {ex.submit(a.act, session_id, self.session_service, self.memory_bank): a.name for a in self.agents.values()}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    r = fut.result()
                    results[name] = r
                except Exception as e:
                    results[name] = {'error': str(e)}
        metrics.incr('orchestrations')
        return results

    def run_sequential(self, session_id: str, order: List[str]) -> Dict[str, Any]:
        results = {}
        for name in order:
            agent = self.agents[name]
            r = agent.act(session_id, self.session_service, self.memory_bank)
            results[name] = r
            msg = Message(sender=name, receiver='broadcast', payload={'type': 'agent_output', 'content': r})
            for other_name, other in self.agents.items():
                if other_name != name:
                    agent.send(msg, other)
        metrics.incr('sequential_runs')
        return results


if __name__ == '__main__':
    session_service = InMemorySessionService()
    memory_bank = MemoryBank(path='/kaggle/working/autods_memory.jsonl')
    session_id = 'kaggle-publish-session-1'
    session_service.create_session(session_id)

    code_tool = CodeExecutionTool()
    data_agent = DataAgent()
    modeling_agent = ModelingAgent(tools={'code_exec': code_tool})
    evaluator_agent = EvaluatorAgent()

    orchestrator = AgentOrchestrator([data_agent, modeling_agent, evaluator_agent], session_service, memory_bank)

    print('Running sequential pipeline: Data -> Modeling -> Evaluation')
    seq_results = orchestrator.run_sequential(session_id, ['DataAgent', 'ModelingAgent', 'EvaluatorAgent'])
    print(json.dumps(seq_results, indent=2))

    print('Saved artifacts:')
    out_dir = '/kaggle/working/autods_outputs'
    for fname in os.listdir(out_dir):
        print('-', fname)

    print('Metrics snapshot:', metrics.snapshot())

    # Display images when running interactively
    try:
        from IPython.display import Image, display
        display(Image(os.path.join(out_dir, 'feature_histograms.png')))
        display(Image(os.path.join(out_dir, 'scatter_matrix.png')))
        display(Image(os.path.join(out_dir, 'confusion_matrix.png')))
        display(Image(os.path.join(out_dir, 'feature_importances.png')))
    except Exception:
        pass

