import time
import logging
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any
from collections import defaultdict

# =========================
# Logging setup (quiet mode)
# =========================
logging.basicConfig(
    level=logging.ERROR,  # Only show errors, not INFO logs
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# =========================
# Sessions & Memory
# =========================
@dataclass
class InMemorySessionService:
    sessions: Dict[str, Dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))

    def get(self, session_id: str, key: str, default=None):
        return self.sessions[session_id].get(key, default)

    def set(self, session_id: str, key: str, value: Any):
        self.sessions[session_id][key] = value

    def get_all(self, session_id: str) -> Dict[str, Any]:
        return self.sessions[session_id]

@dataclass
class MemoryBank:
    preferences: Dict[str, Any] = field(default_factory=dict)
    topics_history: List[str] = field(default_factory=list)
    saved_insights: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def add_topic(self, topic: str):
        self.topics_history.append(topic)

    def set_preference(self, key: str, value: Any):
        self.preferences[key] = value

    def add_insights(self, topic: str, insights: List[str]):
        self.saved_insights[topic].extend(insights)

# =========================
# Tools
# =========================
def mock_google_search(topic: str) -> List[Dict[str, str]]:
    corpus = [
        {"source": "EduBlog", "content": f"{topic} overview: recent trends, challenges, use cases in education."},
        {"source": "JournalX", "content": f"Study on {topic}: methodology, results, limitations; neutral tone."},
        {"source": "NewsSite", "content": f"{topic} breakthrough reported; mixed reactions from experts and students."},
        {"source": "Forum", "content": f"Students discuss {topic} pros/cons; some positive, some skeptical."}
    ]
    return corpus

def custom_sentiment_score(text: str) -> float:
    positives = {"good", "great", "positive", "helpful", "efficient", "breakthrough", "success"}
    negatives = {"bad", "negative", "problem", "challenge", "skeptical", "limitation", "risk"}
    score = 0
    words = text.lower().split()
    for w in words:
        if w in positives:
            score += 1
        elif w in negatives:
            score -= 1
    return round(score / max(5, len(words)), 3)

def extract_key_points(text: str, max_points: int = 3) -> List[str]:
    parts = [p.strip() for p in text.replace(";", ".").split(".") if p.strip()]
    return parts[:max_points]

# =========================
# Agents
# =========================
@dataclass
class AgentOutput:
    data: Any
    metrics: Dict[str, Any]

class SearchAgent:
    def run(self, topic: str) -> AgentOutput:
        start = time.time()
        results = []
        lock = threading.Lock()

        def fetch(source_idx: int):
            time.sleep(0.05 * (source_idx + 1))
            item = mock_google_search(topic)[source_idx]
            with lock:
                results.append(item)

        threads = []
        for i in range(4):
            t = threading.Thread(target=fetch, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        duration = round(time.time() - start, 3)
        metrics = {"duration_s": duration, "items": len(results)}
        return AgentOutput(data=results, metrics=metrics)

class AnalysisAgent:
    def run(self, items: List[Dict[str, str]]) -> AgentOutput:
        start = time.time()
        analyzed = []
        for item in items:
            content = item["content"]
            points = extract_key_points(content)
            sentiment = custom_sentiment_score(content)
            analyzed.append({
                "source": item["source"],
                "points": points,
                "sentiment": sentiment
            })
        avg_sentiment = round(sum(d["sentiment"] for d in analyzed) / max(1, len(analyzed)), 3)
        duration = round(time.time() - start, 3)
        metrics = {"duration_s": duration, "avg_sentiment": avg_sentiment, "items": len(analyzed)}
        return AgentOutput(data=analyzed, metrics=metrics)

class SummarizerAgent:
    def run(self, topic: str, analyzed: List[Dict[str, Any]], session: InMemorySessionService, session_id: str) -> AgentOutput:
        start = time.time()
        prior_pref = session.get(session_id, "tone", "student-friendly")

        lines = [f"Topic: {topic}", f"Tone: {prior_pref}", ""]
        lines.append("Key findings:")
        for d in analyzed:
            src = d["source"]
            pts = d["points"] or ["(no points extracted)"]
            lines.append(f"- {src}:")
            for p in pts:
                lines.append(f"  • {p}")
            lines.append(f"  • Sentiment: {d['sentiment']}")

        avg_sentiment = round(sum(d["sentiment"] for d in analyzed) / max(1, len(analyzed)), 3)
        lines.append("")
        lines.append(f"Overall sentiment (approx): {avg_sentiment}")
        lines.append("Summary:")
        if avg_sentiment > 0.1:
            conclusion = "Overall reactions lean positive, with useful applications and encouraging results."
        elif avg_sentiment < -0.1:
            conclusion = "Overall reactions lean cautious/negative, noting risks and limitations."
        else:
            conclusion = "Overall reactions appear mixed/neutral, with both promising aspects and challenges."
        lines.append(conclusion)

        summary_text = "\n".join(lines)
        duration = round(time.time() - start, 3)
        metrics = {"duration_s": duration, "avg_sentiment": avg_sentiment, "tone": prior_pref}
        return AgentOutput(data=summary_text, metrics=metrics)

# =========================
# Orchestrator
# =========================
class SmartResearchAssistant:
    def __init__(self):
        self.search_agent = SearchAgent()
        self.analysis_agent = AnalysisAgent()
        self.summarizer_agent = SummarizerAgent()
        self.session = InMemorySessionService()
        self.memory_bank = MemoryBank()
        self.metrics = {"runs": 0, "timings": []}

    def set_user_preference(self, key: str, value: Any):
        self.session.set("default", key, value)
        self.memory_bank.set_preference(key, value)

    def run(self, topic: str, session_id: str = "default") -> Dict[str, Any]:
        overall_start = time.time()
        self.memory_bank.add_topic(topic)
        self.session.set(session_id, "last_topic", topic)

        search_out = self.search_agent.run(topic)
        analysis_out = self.analysis_agent.run(search_out.data)
        summary_out = self.summarizer_agent.run(topic, analysis_out.data, self.session, session_id)

        insights = []
        for d in analysis_out.data:
            for p in d["points"]:
                insights.append(p)
        self.memory_bank.add_insights(topic, insights)

        total_time = round(time.time() - overall_start, 3)
        self.metrics["runs"] += 1
        self.metrics["timings"].append(total_time)

        return {
            "topic": topic,
            "summary": summary_out.data,
            "metrics": {
                "total_time_s": total_time,
                "search": search_out.metrics,
                "analysis": analysis_out.metrics,
                "summary": summary_out.metrics,
                "runs": self.metrics["runs"]
            },
            "session_state": self.session.get_all(session_id),
            "memory_bank": {
                "preferences": self.memory_bank.preferences,
                "topics_history": self.memory_bank.topics_history,
                "saved_insights_count": len(self.memory_bank.saved_insights.get(topic, []))
            }
        }

# =========================
# Evaluation
# =========================
def evaluate_outputs(output: Dict[str, Any]) -> Dict[str, Any]:
    summary = output["summary"]
    lines = summary.splitlines()
    checks = {
        "has_topic_line": any(l.startswith("Topic:") for l in lines),
        "has_key_findings": any("Key findings:" in l for l in lines),
        "has_overall_sentiment": any("Overall sentiment" in l for l in lines),
        "length_ok": 10 <= len(lines) <= 80,
        "time_ok": output["metrics"]["total_time_s"] <= 2.0
    }
    score = sum(1 for v in checks.values() if v) / len(checks)
    return {"checks": checks, "score": round(score, 3)}

# =========================

