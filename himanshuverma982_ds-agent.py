# ============================================================
#  CLEAN MULTI-AGENT CAPSTONE PROJECT (NO SQLITE + NO THREADS)
# ============================================================

import time
import json
import logging
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# ------------------------------------------------------------
# Logging / Metrics (Observability)
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("capstone")
METRICS = defaultdict(int)

def timed(metric):
    def wrap(fn):
        def inner(*args, **kwargs):
            start = time.time()
            out = fn(*args, **kwargs)
            METRICS[f"{metric}_count"] += 1
            METRICS[f"{metric}_time"] += time.time() - start
            return out
        return inner
    return wrap


# ------------------------------------------------------------
# Session & Memory (In-Memory)
# ------------------------------------------------------------
class SessionMemory:
    def __init__(self):
        self.sessions = {}

    def create(self, sid: str, data: Dict[str, Any]):
        self.sessions[sid] = data
        return sid

    def update(self, sid: str, data: Dict[str, Any]):
        self.sessions[sid].update(data)

    def get(self, sid: str):
        return self.sessions.get(sid, {})


class MemoryBank:
    """Simple long-term memory using Python lists (no SQLite)."""
    def __init__(self):
        self.store_list = []

    def store(self, tag: str, value: Any):
        self.store_list.append({"tag": tag, "value": value, "ts": time.time()})

    def query(self, tag: str):
        return [x for x in self.store_list if x["tag"].startswith(tag)]


# ------------------------------------------------------------
# Tools (mock for Kaggle)
# ------------------------------------------------------------
class Tool:
    def run(self, *args, **kwargs):
        raise NotImplementedError

class YouTubeSearchTool(Tool):
    @timed("youtube")
    def run(self, query: str, max_results=5):
        return [{"title": f"{query} video {i}",
                 "description": "mock desc",
                 "views": 1000 + i * 250} for i in range(max_results)]

class WebSearchTool(Tool):
    @timed("web")
    def run(self, query: str):
        return [f"mock web result about {query} #{i}" for i in range(3)]

class CodeExecutionTool(Tool):
    def run(self, code: str):
        try:
            return eval(code, {"__builtins__": {}}, {})
        except:
            return "Error"


# ------------------------------------------------------------
# LLM Mock (Kaggle offline safe)
# ------------------------------------------------------------
class LLM:
    @timed("llm")
    def generate(self, prompt: str):
        return f"[MOCK LLM OUTPUT FOR]: {prompt[:50]}..."


# ------------------------------------------------------------
# Base Agent + A2A Message
# ------------------------------------------------------------
class AgentMessage:
    def __init__(self, sender: str, receiver: str, content: str):
        self.sender = sender
        self.receiver = receiver
        self.content = content

class Agent:
    def __init__(self, name, llm, tools, memory):
        self.name = name
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def receive(self, msg: AgentMessage):
        reply = self.llm.generate(f"{self.name} received: {msg.content}")
        return AgentMessage(self.name, msg.sender, reply)

    def tool(self, name, *args, **kwargs):
        return self.tools[name].run(*args, **kwargs)


# ------------------------------------------------------------
# Multi-Agent System
# ------------------------------------------------------------
class DataCollectorAgent(Agent):
    def collect(self, seeds: List[str]):
        output = {}
        for seed in seeds:
            data = self.tool("youtube", seed)
            output[seed] = data
        self.memory.store("collector", output)
        return output


class AnalyzerAgent(Agent):
    def analyze(self, data):
        out = {}
        for q, videos in data.items():
            phrases = []
            for i in range(5):
                phrases.append((f"{q} keyword {i}", 1.0 - i * 0.1))
            out[q] = phrases
            self.memory.store("analyze", phrases)
        return out


class RecommenderAgent(Agent):
    def recommend(self, analyzed):
        final = {}
        for q, kws in analyzed.items():
            ranked = kws[:3]
            suggestions = []
            for word, score in ranked:
                suggestions.append({
                    "keyword": word,
                    "score": score,
                    "title": f"{word} - Complete Guide",
                    "desc": f"This video explains everything about {word}"
                })
            final[q] = suggestions
            self.memory.store("recommend", final[q])
        return final


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
def evaluate(recommendations):
    report = {}
    for q, recs in recommendations.items():
        avg = sum([r["score"] for r in recs]) / len(recs)
        report[q] = {"avg_score": avg, "count": len(recs)}
    return report


# ------------------------------------------------------------
# A2A Broker
# ------------------------------------------------------------
class Broker:
    def __init__(self):
        self.agents = {}

    def add(self, agent):
        self.agents[agent.name] = agent

    def send(self, msg: AgentMessage):
        return self.agents[msg.receiver].receive(msg)


# ------------------------------------------------------------
# MAIN PIPELINE (No SQLite, No threads)
# ------------------------------------------------------------
def main():
    session = SessionMemory()
    memory = MemoryBank()
    llm = LLM()
    tools = {
        "youtube": YouTubeSearchTool(),
        "web": WebSearchTool(),
        "code": CodeExecutionTool(),
    }

    collector = DataCollectorAgent("collector", llm, tools, memory)
    analyzer = AnalyzerAgent("analyzer", llm, tools, memory)
    recommend = RecommenderAgent("recommender", llm, tools, memory)

    broker = Broker()
    broker.add(collector)
    broker.add(analyzer)
    broker.add(recommend)

    seeds = ["python tutorial", "kaggle tips", "video editing"]

    collected = collector.collect(seeds)
    analyzed = analyzer.analyze(collected)
    final = recommend.recommend(analyzed)

    msg = AgentMessage("recommender", "analyzer", "Please critique recommendations")
    critique = broker.send(msg)

    evaluation = evaluate(final)

    session_id = session.create("s1", {"seeds": seeds})
    session.update("s1", {"final": final})

    print("\n=== FINAL RECOMMENDATIONS ===")
    print(json.dumps(final, indent=2))

    print("\n=== EVALUATION ===")
    print(json.dumps(evaluation, indent=2))

    print("\n=== METRICS ===")
    print(dict(METRICS))

main()

   

