"""
RP-MAS: Research & Planning Multi-Agent System
Single-file reference implementation (Python)

This is a modular, extensible template you can run locally and extend.
It contains four agent classes:
 - SupervisorAgent: accepts user task, creates plan, routes work
 - ResearchAgent: performs research (pluggable search/tool adapters)
 - PlanningAgent: turns research into action plans
 - EvaluatorAgent: validates outputs

This code is intentionally framework-agnostic and does NOT call any paid APIs by default.
It includes placeholder adapters for WebSearch and Document ingestion that you can implement.

How to use:
1. Save this file as rp_mas.py (or run directly inside the canvas editor preview)
2. Install dependencies (if you will use optional parts):
   pip install requests beautifulsoup4 python-dotenv tabulate
3. Optionally, provide API keys via environment variables for a real search provider.
4. Run: python rp_mas.py

Extend by swapping the WebSearchTool implementation with LangChain / SerpAPI / Bing.

"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
import json
import os
import textwrap

# Optional imports for the example web fetch
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None


# -------------------------------
# Utilities
# -------------------------------

def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def short(text: str, n: int = 200) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


# -------------------------------
# Tool Adapters (pluggable)
# -------------------------------

class WebSearchTool:
    """
    Minimal web search adapter.

    By default it uses duckduckgo-lite (unofficial) via html scraping if requests & bs4 are available.
    Replace this class with one that calls SerpAPI / Bing / Google with an API key for production.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Return list of {'title','snippet','url'}"""
        if self.api_key:
            # Placeholder: implement API-based search here
            raise NotImplementedError("Plug your search provider here (SerpAPI/Bing/etc.)")

        if not requests or not BeautifulSoup:
            # Fallback: return empty results and a hint
            return [
                {
                    "title": "(search unavailable)",
                    "snippet": "requests or bs4 not installed; install them to enable simple scraping fallback",
                    "url": "",
                }
            ]

        # Very simple DuckDuckGo HTML scraping (fragile) just for prototyping
        try:
            resp = requests.get("https://html.duckduckgo.com/html/", params={"q": query}, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for a in soup.select("a.result__a")[:max_results]:
                title = a.get_text().strip()
                url = a.get('href')
                snippet_tag = a.find_next("a").find_next_sibling(text=True)
                snippet = short(snippet_tag or "", 300)
                results.append({"title": title, "snippet": snippet, "url": url})
            if not results:
                # alternative parse
                for r in soup.select("div.result")[:max_results]:
                    t = r.select_one("a.result__a")
                    if t:
                        results.append({"title": t.get_text().strip(), "snippet": "", "url": t.get('href')})
            return results
        except Exception:
            return [
                {
                    "title": "(search failed)",
                    "snippet": "An error occurred while trying to perform web search. Check network or install a proper search provider.",
                    "url": "",
                }
            ]


class DocumentTool:
    """
    Minimal document ingestion tool. It can read local files (txt, json) and return content.
    Extend to parse PDFs using PyPDF2, or DOCX using python-docx.
    """

    SUPPORTED = (".txt", ".json")

    def read(self, path: str) -> str:
        _, ext = os.path.splitext(path.lower())
        if ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.dumps(json.load(f), indent=2)
        else:
            raise ValueError(f"Unsupported file type: {ext}")


# -------------------------------
# Agent dataclasses
# -------------------------------

@dataclass
class ResearchItem:
    title: str
    snippet: str
    url: str
    source: str = "web"


@dataclass
class ResearchResult:
    query: str
    items: List[ResearchItem] = field(default_factory=list)

    def summarize(self, max_chars: int = 1000) -> str:
        lines = [f"Research summary for: {self.query}"]
        for i, it in enumerate(self.items, start=1):
            lines.append(f"{i}. {it.title} — {short(it.snippet, 300)}")
            lines.append(f"   Source: {it.url}")
        s = "\n".join(lines)
        return short(s, max_chars)


# -------------------------------
# Agents (Supervisor, Research, Planning, Evaluator)
# -------------------------------

class ResearchAgent:
    def __init__(self, search_tool: WebSearchTool):
        self.search_tool = search_tool

    def research(self, query: str, topics: List[str], max_results: int = 5) -> Dict[str, ResearchResult]:
        out = {}
        for t in topics:
            q = f"{query} {t}"
            raw = self.search_tool.search(q, max_results=max_results)
            items = [ResearchItem(title=r.get("title", ""), snippet=r.get("snippet", ""), url=r.get("url", "")) for r in raw]
            out[t] = ResearchResult(query=q, items=items)
        return out


class PlanningAgent:
    def __init__(self):
        pass

    def create_plan(self, research_pack: Dict[str, ResearchResult], plan_name: str = "Strategic Plan") -> Dict[str, Any]:
        # Very simple plan generation: synthesize phases from research topics
        phases = []
        for i, (topic, result) in enumerate(research_pack.items(), start=1):
            phase = {
                "phase_id": i,
                "title": f"Investigate {topic}",
                "summary": result.summarize(500),
                "deliverables": [f"Summary report on {topic}", f"Risks & opportunities for {topic}"],
                "duration_days": max(3, len(result.items)),
            }
            phases.append(phase)

        plan = {
            "name": plan_name,
            "created_at": now_ts(),
            "phases": phases,
            "milestones": [
                {"name": "Research complete", "due_in_days": sum(p['duration_days'] for p in phases)},
                {"name": "Plan draft", "due_in_days": sum(p['duration_days'] for p in phases) + 3},
            ],
        }
        return plan


class EvaluatorAgent:
    def __init__(self):
        pass

    def validate_research(self, research_pack: Dict[str, ResearchResult]) -> Tuple[bool, List[str]]:
        """Simple validation: ensure each topic has at least one non-empty URL/title/snippet."""
        issues = []
        for topic, res in research_pack.items():
            if not res.items:
                issues.append(f"No results for topic: {topic}")
            else:
                all_blank = all((not it.title.strip() and not it.snippet.strip() and not it.url.strip()) for it in res.items)
                if all_blank:
                    issues.append(f"All results blank for topic: {topic}")
        return (len(issues) == 0, issues)

    def validate_plan(self, plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues = []
        if not plan.get("phases"):
            issues.append("Plan contains no phases")
        else:
            for p in plan['phases']:
                if 'summary' not in p or not p['summary'].strip():
                    issues.append(f"Phase {p.get('phase_id')} has empty summary")
        return (len(issues) == 0, issues)


class SupervisorAgent:
    def __init__(self, research_agent: ResearchAgent, planning_agent: PlanningAgent, evaluator: EvaluatorAgent):
        self.research_agent = research_agent
        self.planning_agent = planning_agent
        self.evaluator = evaluator

    def handle_request(self, user_query: str, topics: List[str]) -> Dict[str, Any]:
        # 1. Kick off research
        print(f"[Supervisor] Received task: {user_query}")
        research = self.research_agent.research(user_query, topics)
        print(f"[Supervisor] Research completed for topics: {', '.join(topics)}")

        # 2. Validate research
        ok, issues = self.evaluator.validate_research(research)
        if not ok:
            print(f"[Supervisor] Evaluator found research issues: {issues}")
            # For this simple template, we'll continue but mark issues in the output

        # 3. Create plan
        plan = self.planning_agent.create_plan(research)
        print(f"[Supervisor] PlanningAgent produced a plan with {len(plan['phases'])} phases")

        # 4. Validate plan
        ok2, plan_issues = self.evaluator.validate_plan(plan)
        if not ok2:
            print(f"[Supervisor] Plan validation issues: {plan_issues}")

        # 5. Consolidate output
        output = {
            "query": user_query,
            "topics": topics,
            "research": {t: {"items": [it.__dict__ for it in res.items]} for t, res in research.items()},
            "research_issues": issues,
            "plan": plan,
            "plan_issues": plan_issues,
            "generated_at": now_ts(),
        }
        return output


# -------------------------------
# CLI / Example usage
# -------------------------------

EXAMPLE_TOPICS = ["market size", "competitors", "regulatory landscape", "technology trends"]


def example_run():
    print("RP-MAS: example run\n")
    search_tool = WebSearchTool(api_key=os.getenv("SEARCH_API_KEY"))
    research_agent = ResearchAgent(search_tool)
    planning_agent = PlanningAgent()
    evaluator = EvaluatorAgent()
    supervisor = SupervisorAgent(research_agent, planning_agent, evaluator)

    user_query = "Electric vehicle market India 2025-2030"
    result = supervisor.handle_request(user_query, EXAMPLE_TOPICS)

    # Pretty print summary
    print('\n===== SUMMARY =====')
    print(f"Task: {result['query']}")
    print(f"Generated at: {result['generated_at']}")
    print('\nResearch issues:', result['research_issues'])
    print('\nPlan issues:', result['plan_issues'])
    print('\nPlan phases:')
    for ph in result['plan']['phases']:
        print(f" - {ph['title']}: {short(ph['summary'],200)} (duration {ph['duration_days']} days)")

    # Save to file
    out_path = 'rp_mas_output.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\nFull output written to: {out_path}")


# -------------------------------
# If run as script
# -------------------------------

if __name__ == '__main__':
    example_run()


