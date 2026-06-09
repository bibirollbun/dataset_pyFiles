import uuid
import time
import json
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# ---------------------------
# Utilities & Mock LLM
# ---------------------------

def mock_llm(prompt: str, role: str = "assistant", temperature: float = 0.2) -> str:
    """
    Simple deterministic mock LLM to simulate different agent outputs.
    In production, call a real LLM here.
    """
    # Simulate latency
    time.sleep(0.2)

    # routing by keywords in prompt or role
    p = prompt.lower()
    if role == "research":
        return ("Found 3 resources: (1) overview article, (2) seminal paper, (3) tutorial repo. "
                "Key takeaways: concept A, trade-offs, typical pitfalls.")
    if role == "curriculum":
        return ("Week plan (4 weeks): Week1 - foundations, Week2 - core algorithms, "
                "Week3 - project build, Week4 - evaluation & deployment. Daily tasks included.")
    if role == "generator":
        return ("Generated starter code scaffold and architecture docs. See function train(), eval().")
    if role == "experiment":
        return ("Experiment recipe: dataset split, baseline model, metrics (acc, f1).")
    if role == "critic":
        # a simple feedback
        return ("Critique: expand evaluation, add ablation study, clarify dataset license.")
    # default
    return "Short instructive answer."

def save_text_file(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ---------------------------
# Data classes
# ---------------------------

@dataclass
class Resource:
    title: str
    url: str
    summary: str
    relevance: float = 0.8

@dataclass
class Flashcard:
    question: str
    answer: str
    ease: float = 2.5

@dataclass
class Plan:
    weeks: int
    outline: str
    daily_tasks: List[str]

# ---------------------------
# Session service
# ---------------------------

class InMemorySessionService:
    """Simple session store keyed by session id."""
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        sid = str(uuid.uuid4())
        with self.lock:
            self.sessions[sid] = {
                "user_id": user_id,
                "created_at": time.time(),
                "metadata": metadata or {},
                "state": {}
            }
        return sid

    def get(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {})

    def set_state(self, session_id: str, key: str, value: Any):
        if session_id not in self.sessions:
            raise KeyError("session not found")
        with self.lock:
            self.sessions[session_id]["state"][key] = value

    def get_state(self, session_id: str, key: str, default=None):
        return self.sessions.get(session_id, {}).get("state", {}).get(key, default)

# ---------------------------
# Agents
# ---------------------------

class ResearchAgent:
    """Search & synthesize resources (mocked)."""
    def run(self, topic: str) -> List[Resource]:
        prompt = f"Research resources for topic: {topic}"
        out = mock_llm(prompt, role="research")
        # mock parsing into resources
        resources = [
            Resource(title=f"{topic} — Overview Article", url="https://example.com/overview",
                     summary="A brief overview covering problem, motivation, and key results.", relevance=0.95),
            Resource(title=f"{topic} — Seminal Paper", url="https://example.com/paper",
                     summary="A seminal paper describing main algorithmic idea and math.", relevance=0.9),
            Resource(title=f"{topic} — Tutorial Repo", url="https://github.com/example/repo",
                     summary="A practical tutorial repo with code and examples.", relevance=0.8),
        ]
        return resources

class CurriculumAgent:
    """Create an adaptive curriculum plan."""
    def run(self, topic: str, weekly_hours: int = 5) -> Plan:
        prompt = f"Design a curriculum for {topic} given {weekly_hours} weekly hours"
        out = mock_llm(prompt, role="curriculum")
        outline = out
        daily_tasks = [
            "Read overview and take notes",
            "Implement small example",
            "Work through coding exercise",
            "Review and write reflection",
            "Optional: advanced reading"
        ]
        return Plan(weeks=4, outline=outline, daily_tasks=daily_tasks)

class GeneratorAgent:
    """Generates starter code, slides (markdown), and diagrams (text)."""
    def run(self, topic: str, plan: Plan) -> Dict[str, str]:
        prompt = f"Generate starter project for {topic} based on plan: {plan.outline}"
        out = mock_llm(prompt, role="generator")
        # create a starter script (mock)
        starter_code = f"""# starter_{topic.replace(' ', '_')}.py
\"\"\"Starter prototype for {topic}\"\"\"
def train():
    print('train model — placeholder')

def evaluate():
    print('evaluate model — placeholder')

if __name__ == '__main__':
    train()
    evaluate()
"""
        slide_md = f"# {topic} — Project Slides\n\n## Overview\n\n{plan.outline}\n\n## Plan\n\n" + "\n".join(f"- {d}" for d in plan.daily_tasks)
        architecture_doc = "Architecture: Data ingestion -> Model -> Eval -> Deploy. (Textual placeholder)"
        return {"starter_code.py": starter_code, "slides.md": slide_md, "architecture.txt": architecture_doc}

class ExperimentAgent:
    """Design experiments and evaluation recipes."""
    def run(self, topic: str) -> str:
        prompt = f"Design experiments for {topic}"
        out = mock_llm(prompt, role="experiment")
        return out

class FlashcardAgent:
    """Generate flashcards based on core concepts."""
    def run(self, topic: str, n_cards: int = 5) -> List[Flashcard]:
        cards = []
        for i in range(n_cards):
            q = f"What is key concept {i+1} in {topic}?"
            a = mock_llm(q, role="assistant")
            cards.append(Flashcard(question=q, answer=a, ease=2.0 + i*0.1))
        return cards

class CriticAgent:
    """Critique outputs and propose improvements."""
    def run(self, assembled_outputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Critique the outputs: keys={list(assembled_outputs.keys())}"
        critique = mock_llm(prompt, role="critic")
        # propose simple improvement suggestions
        suggestions = {
            "critique_text": critique,
            "improvements": [
                "Add ablation experiments",
                "Expand dataset sources",
                "Add evaluation scripts with metrics"
            ]
        }
        return suggestions

# ---------------------------
# Coordinator
# ---------------------------

class StudyCoordinator:
    """Coordinate multiple agents to produce a goal->product pipeline."""
    def __init__(self, session_service: InMemorySessionService):
        self.sessions = session_service
        self.research_agent = ResearchAgent()
        self.curr_agent = CurriculumAgent()
        self.gen_agent = GeneratorAgent()
        self.exp_agent = ExperimentAgent()
        self.flash_agent = FlashcardAgent()
        self.critic_agent = CriticAgent()

    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self.sessions.create_session(user_id, metadata)

    def plan_and_build(self, session_id: str, topic: str, weekly_hours: int = 5) -> Dict[str, Any]:
        # step1: plan (single agent)
        plan = self.curr_agent.run(topic, weekly_hours)
        self.sessions.set_state(session_id, "plan", asdict(plan))

        # step2: run research + flashcard + experiments in parallel
        results: Dict[str, Any] = {
            "resources": [],
            "flashcards": [],
            "experiments": None,
            "generated": {}
        }

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                ex.submit(self.research_agent.run, topic): "resources",
                ex.submit(self.flash_agent.run, topic, 6): "flashcards",
                ex.submit(self.exp_agent.run, topic): "experiments",
                ex.submit(self.gen_agent.run, topic, plan): "generated"
            }
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"error": str(e)}
                results[key] = res

        # Ensure types are safe (resources should be list etc.)
        resources_list = results.get("resources") or []
        if not isinstance(resources_list, list):
            resources_list = [resources_list]

        flashcards_list = results.get("flashcards") or []
        if not isinstance(flashcards_list, list):
            flashcards_list = [flashcards_list]

        generated_dict = results.get("generated") or {}
        if not isinstance(generated_dict, dict):
            generated_dict = {}

        # store intermediate results
        self.sessions.set_state(session_id, "research_results", [asdict(r) for r in resources_list])
        self.sessions.set_state(session_id, "flashcards", [asdict(f) for f in flashcards_list])
        self.sessions.set_state(session_id, "generated", generated_dict)
        self.sessions.set_state(session_id, "experiments", results.get("experiments"))

        # step3: assemble product
        assembled = {
            "topic": topic,
            "plan": asdict(plan),
            "resources": [asdict(r) for r in resources_list],
            "flashcards": [asdict(f) for f in flashcards_list],
            "experiments": results.get("experiments"),
            "generated": generated_dict
        }

        # step4: critique & self-improve loop
        critique = self.critic_agent.run(assembled)
        self.sessions.set_state(session_id, "critique", critique)

        # Very simple loop: if critic suggests 'expand dataset' or 'ablation', append a fake resource and regenerate starter code safely
        improvements = critique.get("improvements", [])
        if any("dataset" in s.lower() or "ablation" in s.lower() for s in improvements):
            # simulate making an improvement: add dataset resource
            new_resource = Resource(title=f"{topic} — Additional Dataset", url="https://example.com/dataset",
                                    summary="Additional dataset to test generalization.", relevance=0.7)
            resources = resources_list + [new_resource]

            # --- SAFER STARTER CODE EXTRACTION & UPDATE ---
            generated = dict(generated_dict)  # shallow copy

            # Possible keys the generator might use
            possible_keys = ["starter_code", "starter_code.py", "starter", "code"]
            starter_key = None

            for key in possible_keys:
                if key in generated:
                    starter_key = key
                    break

            # If not found, try to dynamically search for keys containing 'starter' or '.py'
            if starter_key is None:
                for k in generated.keys():
                    if "starter" in k.lower() or k.lower().endswith(".py"):
                        starter_key = k
                        break

            # If still missing → create a safe placeholder
            if starter_key is None:
                starter_key = "starter_code"
                generated[starter_key] = (
                    "# Starter code placeholder\n"
                    "def train():\n"
                    "    pass\n"
                )

            # Create/overwrite a normalized starter_code.py entry for export & editing
            base_code = generated.get(starter_key, "# placeholder starter code\n")
            # Ensure it's a string
            if not isinstance(base_code, str):
                base_code = str(base_code)

            generated["starter_code.py"] = base_code + "\n# NOTE: Added dataset loading placeholder\n"

            # update assembled and session state
            assembled["resources"] = [asdict(r) for r in resources]
            assembled["generated"] = generated
            self.sessions.set_state(session_id, "research_results", [asdict(r) for r in resources])
            self.sessions.set_state(session_id, "generated", generated)

        # final save
        self.sessions.set_state(session_id, "final_assembled", assembled)
        return assembled

    def export_outputs(self, session_id: str, out_dir: str = "outputs") -> str:
        assembled = self.sessions.get_state(session_id, "final_assembled")
        if not assembled:
            assembled = {
                "topic": "unknown",
                "generated": self.sessions.get_state(session_id, "generated") or {},
                "resources": self.sessions.get_state(session_id, "research_results") or [],
                "flashcards": self.sessions.get_state(session_id, "flashcards") or []
            }

        os.makedirs(out_dir, exist_ok=True)

        # write slides
        slides_md = assembled.get("generated", {}).get("slides.md") or assembled.get("generated", {}).get("slides", "# slides\nNo slides generated")
        save_text_file(os.path.join(out_dir, "slides.md"), slides_md)

        # write starter code
        starter_code = assembled.get("generated", {}).get("starter_code.py", assembled.get("generated", {}).get("starter_code", "# no starter code"))
        save_text_file(os.path.join(out_dir, "starter_code.py"), starter_code)

        # write architecture doc
        arch = assembled.get("generated", {}).get("architecture.txt", "")
        save_text_file(os.path.join(out_dir, "architecture.txt"), arch)

        # write resources & flashcards
        save_text_file(os.path.join(out_dir, "resources.json"), json.dumps(assembled.get("resources", []), indent=2))
        save_text_file(os.path.join(out_dir, "flashcards.json"), json.dumps(assembled.get("flashcards", []), indent=2))

        # write summary report
        report = {
            "topic": assembled.get("topic"),
            "plan": assembled.get("plan"),
            "experiments": assembled.get("experiments"),
            "critique": self.sessions.get_state(session_id, "critique")
        }
        save_text_file(os.path.join(out_dir, "summary_report.json"), json.dumps(report, indent=2))

        return out_dir

# ---------------------------
# Demo / Main
# ---------------------------

def main_demo():
    print("Starting SynaptiQ MVP demo...")
    session_store = InMemorySessionService()
    coord = StudyCoordinator(session_store)

    sid = coord.create_session(user_id="user_demo", metadata={"preferred_format": "slides+code"})
    print(f"Created session id: {sid}")

    topic = "Plant Disease Detection with Computer Vision"
    assembled = coord.plan_and_build(sid, topic, weekly_hours=8)

    print("\n--- Plan outline ---")
    print(assembled.get("plan", {}).get("outline", "<no outline>"))

    print("\n--- Top resources ---")
    for r in assembled.get("resources", [])[:3]:
        print(f"- {r.get('title')} ({r.get('url')}) [{r.get('relevance')}]")

    print("\n--- Flashcards (sample) ---")
    for f in assembled.get("flashcards", [])[:3]:
        print(f"- Q: {f.get('question')} => A: {f.get('answer')[:80] if f.get('answer') else ''}...")

    outdir = coord.export_outputs(sid, out_dir="outputs_demo")
    print(f"\nExported outputs to: {outdir}")

    print("\n--- Critique ---")
    print(json.dumps(session_store.get_state(sid, "critique"), indent=2))

    print("\nDemo complete. Inspect the outputs_demo folder for generated artifacts.")

if __name__ == "__main__":
    main_demo()

