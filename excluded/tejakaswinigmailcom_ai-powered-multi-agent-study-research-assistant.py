
import os
import re
import uuid
import json
import time
import logging
import threading
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from collections import OrderedDict

# ---------------- Observability ----------------
logger = logging.getLogger("study_assistant")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(trace_id)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.propagate = False

def trace(func):
    def wrapper(*args, **kwargs):
        trace_id = kwargs.pop("_trace_id", str(uuid.uuid4()))
        extra = {"trace_id": trace_id}
        logger_adapter = logging.LoggerAdapter(logger, extra)
        logger_adapter.debug(f"ENTER {func.__name__}")
        start = time.time()
        try:
            out = func(*args, **kwargs)
            duration = time.time() - start
            logger_adapter.debug(f"EXIT {func.__name__} (t={duration:.3f}s)")
            return out
        except Exception as e:
            logger_adapter.exception(f"ERR {func.__name__}: {e}")
            raise
    return wrapper

# ---------------- LLM Client (OpenAI optional) ----------------
class LLMClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.use_openai = False
        self.openai = None
        if self.api_key:
            try:
                import openai
                openai.api_key = self.api_key
                self.openai = openai
                self.use_openai = True
            except Exception:
                self.use_openai = False

    @trace
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, model: str = "gpt-4o-mini", **kwargs) -> str:
        if self.use_openai and self.openai:
            try:
                messages = [
                    {"role": "system", "content": "You are a concise, professional assistant formatting study material with emojis and markdown."},
                    {"role": "user", "content": prompt}
                ]
                resp = self.openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
                text = resp["choices"][0]["message"]["content"]
                return text.strip()
            except Exception:
                logger.exception("OpenAI call failed; falling back to offline.")
        # Offline fallback: deterministic, safe summarizer
        cleaned = re.sub(r"\s+", " ", prompt.replace("\n", " ")).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        summary = ". ".join(sentences[:6])
        if summary:
            return summary + ("" if summary.endswith(".") else ".")
        return prompt[:min(400, len(prompt))]

# ---------------- Tools ----------------
class PDFTool:
    @trace
    def extract_text(self, pdf_path: str) -> str:
        # Try PyPDF2 for real PDF extraction; fallback to reading text file or mock
        try:
            import PyPDF2
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = [p.extract_text() or "" for p in reader.pages]
                return "\n".join(pages)
        except Exception:
            try:
                with open(pdf_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return f"[Mock PDF content from {pdf_path}] " + "\n".join(["Point 1: ...", "Point 2: ...", "Conclusion: ..."])

class VideoTranscriptTool:
    @trace
    def get_transcript(self, video_id: str) -> str:
        return f"Transcript for {video_id}. This demo transcript explains topic A and gives examples."

class SearchTool:
    @trace
    def search(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
        return [{"title": f"Result {i+1} for {query}", "snippet": f"Snippet {i+1}.", "url": f"https://example.com/{i+1}"} for i in range(limit)]

# ---------------- Session & Memory ----------------
@dataclass
class InMemorySession:
    session_id: str
    created_at: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

class InMemorySessionService:
    def __init__(self):
        self.sessions: Dict[str, InMemorySession] = {}

    def create_session(self) -> InMemorySession:
        sid = str(uuid.uuid4())
        s = InMemorySession(session_id=sid)
        self.sessions[sid] = s
        logger.info(f"Created session {sid}", extra={"trace_id": sid})
        return s

    def get_session(self, session_id: str) -> Optional[InMemorySession]:
        return self.sessions.get(session_id)

    def update_session(self, session_id: str, key: str, value: Any):
        s = self.get_session(session_id)
        if not s:
            raise KeyError("session not found")
        s.data[key] = value

class MemoryBank:
    def __init__(self, filename: str = "memory_bank.json"):
        self.filename = filename
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                self.store = json.load(f)
        except Exception:
            self.store = {}

    def get(self, user_id: str) -> Dict[str, Any]:
        return self.store.get(user_id, {})

    def put(self, user_id: str, key: str, value: Any):
        self.store.setdefault(user_id, {})[key] = value
        self._persist()

    def append_note(self, user_id: str, note: str):
        self.store.setdefault(user_id, {}).setdefault("notes", []).append({"t": time.time(), "note": note})
        self._persist()

    def _persist(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.store, f, indent=2)

# ---------------- Context Compaction ----------------
class ContextCompactor:
    def __init__(self, llm: LLMClient, token_budget: int = 1500):
        self.llm = llm
        self.token_budget = token_budget

    @trace
    def compact(self, messages: List[str]) -> str:
        combined = "\n\n".join(messages)
        if len(combined) <= self.token_budget:
            return combined
        chunks = [combined[i:i+self.token_budget] for i in range(0, len(combined), self.token_budget)]
        summaries = []
        for c in chunks:
            s = self.llm.generate(f"Summarize concisely:\n\n{c}")
            summaries.append(s)
        final = "\n\n".join(summaries)
        if len(final) > self.token_budget:
            final = final[:self.token_budget]
        return final

# ---------------- Helper functions ----------------
def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s).strip().lower()

def _offline_paraphrase(text: str) -> str:
    swaps = {
        "machine learning": "ML",
        "supervised learning": "supervised methods",
        "loss functions": "loss metrics",
        "model evaluation": "evaluating models",
        "regularization techniques": "regularization",
        "datasets": "data sets",
        "training loops": "training procedures",
        "best practices": "recommended practices",
        "references": "citations"
    }
    out = text.strip()
    for k, v in swaps.items():
        out = re.sub(re.escape(k), v, out, flags=re.IGNORECASE)
    if len(out.split()) > 28:
        out = " ".join(out.split()[:22]) + "..."
    return out

def _paraphrase_via_llm_or_offline(llm: LLMClient, text: str, role_hint: str = "") -> str:
    try:
        if getattr(llm, "use_openai", False):
            prompt = f"Paraphrase concisely and professionally for a study summary ({role_hint}):\n\n{text}\n\nKeep it short (1-2 sentences)."
            out = llm.generate(prompt, max_tokens=160, temperature=0.3)
            return out.strip()
    except Exception:
        pass
    return _offline_paraphrase(text)

# ---------------- Agents ----------------
class BaseAgent:
    def __init__(self, llm: LLMClient, tools: Dict[str, Any] = None):
        self.llm = llm
        self.tools = tools or {}

    def act(self, *args, **kwargs):
        raise NotImplementedError

class SummarizerAgent(BaseAgent):
    @trace
    def act(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"summary_md": "No content provided.", "sections": {}, "algorithms_used": []}

        sentences = _split_sentences(text)
        seen = set()
        uniq = []
        for s in sentences:
            k = _norm(s)
            if k and k not in seen:
                seen.add(k)
                uniq.append(s)

        sections = {"overview": "", "key_concepts": [], "steps": [], "example": "", "takeaways": []}
        used_norms = set()
        if uniq:
            sections["overview"] = uniq[0]
            used_norms.add(_norm(uniq[0]))

        for s in uniq[1:6]:
            if len(sections["key_concepts"]) >= 3:
                break
            kn = _norm(s)
            if kn not in used_norms:
                sections["key_concepts"].append(s)
                used_norms.add(kn)

        for s in uniq:
            kn = _norm(s)
            if kn in used_norms:
                continue
            if len(sections["steps"]) < 4:
                sections["steps"].append(s)
                used_norms.add(kn)

        for s in reversed(uniq):
            kn = _norm(s)
            if kn in used_norms:
                continue
            sections["takeaways"].insert(0, s)
            used_norms.add(kn)
            if len(sections["takeaways"]) >= 2:
                break

        if sections["key_concepts"]:
            sections["example"] = f"Apply '{sections['key_concepts'][0]}' on a tiny dataset: split into train/test, train, evaluate."

        # paraphrase each section
        if sections["overview"]:
            sections["overview"] = _paraphrase_via_llm_or_offline(self.llm, sections["overview"], "overview")
        sections["key_concepts"] = [ _paraphrase_via_llm_or_offline(self.llm, kc, "key concept") for kc in sections["key_concepts"] ]
        sections["steps"] = [ _paraphrase_via_llm_or_offline(self.llm, s, "step") for s in sections["steps"] ]
        if sections["example"]:
            sections["example"] = _paraphrase_via_llm_or_offline(self.llm, sections["example"], "example")
        sections["takeaways"] = [ _paraphrase_via_llm_or_offline(self.llm, t, "takeaway") for t in sections["takeaways"] ]

        algorithms_used = [
            "Chunked parallel summarization (ThreadPoolExecutor)",
            "Context compaction (iterative summarization)",
            "Session + Memory persistence (MemoryBank JSON)",
            "Agent orchestration pipeline (Summarizer â†’ Note â†’ Quiz)"
        ]

        md_parts = []
        md_parts.append("ğŸ“˜ **1) High-level Overview**\n" + (sections["overview"] or "-"))
        md_parts.append("\n---\nğŸ§  **2) Key Concepts**\n" + ("\n".join(f"- ğŸ”¹ {kc}" for kc in sections["key_concepts"]) if sections["key_concepts"] else "-"))
        md_parts.append("\n---\nğŸ› ï¸� **3) Step-by-step Explanation**\n" + ("\n".join(f"- {i+1}. {s}" for i, s in enumerate(sections["steps"])) if sections["steps"] else "-"))
        md_parts.append("\n---\nğŸ”� **4) Examples**\n" + (sections["example"] or "-"))
        md_parts.append("\n---\nâœ… **5) Final Takeaways**\n" + ("\n".join(f"- {t}" for t in sections["takeaways"]) if sections["takeaways"] else "-"))
        summary_md = "\n\n".join(md_parts)

        return {"summary_md": summary_md, "sections": sections, "algorithms_used": algorithms_used}

class NoteAgent(BaseAgent):
    @trace
    def act(self, summary_obj: Dict[str, Any]) -> Dict[str, Any]:
        secs = summary_obj.get("sections", {})
        overview = secs.get("overview", "")
        key_concepts = secs.get("key_concepts", [])
        steps = secs.get("steps", [])
        example = secs.get("example", "")
        takeaways = secs.get("takeaways", [])

        lines = []
        lines.append("ğŸ“š **Study Notes**")
        lines.append("\n---\n### ğŸ“Œ Overview\n")
        lines.append(f"- {overview}" if overview else "-")
        lines.append("\n---\n### ğŸ§  Key Concepts\n")
        if key_concepts:
            lines.extend([f"- âœ… {kc}" for kc in key_concepts])
        else:
            lines.append("-")
        lines.append("\n---\n### ğŸ› ï¸� Step-by-step\n")
        if steps:
            lines.extend([f"- {i+1}. {s}" for i, s in enumerate(steps)])
        else:
            lines.append("-")
        lines.append("\n---\n### ğŸ”� Example\n")
        lines.append(example if example else "-")
        lines.append("\n---\n### âœ… Final Takeaways\n")
        if takeaways:
            lines.extend([f"- {t}" for t in takeaways])
        else:
            lines.append("-")
        lines.append("\n---\n### âš¡ Quick Actions\n- âœ�ï¸� Make flashcards\n- ğŸ§ª Try the example in a notebook\n")
        formatted_text = "\n".join(lines)
        return {"formatted_text": formatted_text, "overview": overview, "key_concepts": key_concepts, "steps": steps, "example": example, "takeaways": takeaways}

class QuizAgent(BaseAgent):
    @trace
    def act(self, notes_obj: Dict[str, Any], n_questions: int = 5) -> List[Dict[str, Any]]:
        pool = list(OrderedDict.fromkeys((notes_obj.get("key_concepts") or []) + (notes_obj.get("takeaways") or [])))
        questions = []
        if not pool:
            return questions
        for i in range(min(n_questions, len(pool))):
            fact = pool[i]
            q = f"â�“ Q{i+1}: In your own words, explain â€” {fact}"
            a = f"ğŸŸ¢ A{i+1}: {fact}"
            try:
                if getattr(self.llm, "use_openai", False):
                    q = self.llm.generate(f"Rewrite this quiz prompt concisely for students: In your own words, explain â€” {fact}", max_tokens=80, temperature=0.2)
            except Exception:
                pass
            questions.append({"id": i+1, "q": q, "a": a})
        k = 0
        while len(questions) < n_questions:
            fact = pool[k % len(pool)]
            q = f"â�“ Q{len(questions)+1}: Give a simple example or application of â€” {fact}"
            a = f"ğŸŸ¢ A{len(questions)+1}: An example/application of {fact}"
            questions.append({"id": len(questions)+1, "q": q, "a": a})
            k += 1
        return questions

# ---------------- Orchestrator ----------------
class Orchestrator:
    def __init__(self, llm: LLMClient, tools: Dict[str, Any], session_service: InMemorySessionService, memory_bank: MemoryBank):
        self.llm = llm
        self.tools = tools
        self.session_service = session_service
        self.memory_bank = memory_bank
        self.summarizer = SummarizerAgent(llm, tools)
        self.note_agent = NoteAgent(llm, tools)
        self.quiz_agent = QuizAgent(llm, tools)
        self.context_compactor = ContextCompactor(llm)
        self.metrics = {"jobs_started": 0, "jobs_finished": 0}
        self.job_store: Dict[str, Dict] = {}
        self.lock = threading.Lock()

    @trace
    def start_job(self, session_id: str, inputs: Dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        with self.lock:
            self.job_store[job_id] = {"status": "running", "session_id": session_id, "result": None}
            self.metrics["jobs_started"] += 1
        t = threading.Thread(target=self._run_pipeline, args=(job_id, inputs), daemon=True)
        t.start()
        return job_id

    @trace
    def pause_job(self, job_id: str):
        with self.lock:
            if job_id in self.job_store:
                self.job_store[job_id]["status"] = "paused"

    @trace
    def resume_job(self, job_id: str):
        with self.lock:
            if job_id in self.job_store and self.job_store[job_id]["status"] == "paused":
                self.job_store[job_id]["status"] = "running"
                t = threading.Thread(target=self._run_pipeline, args=(job_id, self.job_store[job_id].get("inputs", {})), daemon=True)
                t.start()

    def _should_continue(self, job_id: str) -> bool:
        s = self.job_store.get(job_id, {})
        return s.get("status") == "running"

    @trace
    def _run_pipeline(self, job_id: str, inputs: Dict[str, Any]):
        try:
            with self.lock:
                self.job_store[job_id]["inputs"] = inputs
            session_id = inputs.get("session_id")
            source_type = inputs.get("source_type")
            raw = ""
            if source_type == "pdf":
                raw = self.tools["pdf"].extract_text(inputs["path"])
            elif source_type == "video":
                raw = self.tools["video"].get_transcript(inputs["video_id"])
            elif source_type == "url":
                hits = self.tools["search"].search(inputs["url"])
                raw = "\n".join([h["snippet"] for h in hits])
            else:
                raw = inputs.get("text", "")

            chunks = [raw[i:i+1000] for i in range(0, len(raw), 1000)] or [raw]
            results = []
            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = [ex.submit(self.summarizer.act, c) for c in chunks]
                for f in as_completed(futures):
                    r = f.result()
                    results.append(r["summary_md"] if isinstance(r, dict) else r)

            combined_summary = "\n\n".join(results)
            for _ in range(2):
                if not self._should_continue(job_id):
                    logger.info("Job paused or cancelled", extra={"trace_id": job_id})
                    return
                if self.llm.use_openai:
                    combined_summary = self.llm.generate(f"Refine this summary to be clearer and shorter:\n\n{combined_summary}", max_tokens=400)
                else:
                    combined_summary = self.context_compactor.compact([combined_summary])

            summary_obj = self.summarizer.act(combined_summary)
            notes = self.note_agent.act(summary_obj)
            quiz = self.quiz_agent.act(notes, n_questions=5)
            user_id = inputs.get("user_id", "anonymous")
            self.memory_bank.append_note(user_id, notes.get("formatted_text") if isinstance(notes, dict) else str(notes))

            result = {"summary": summary_obj.get("summary_md"), "notes": notes, "quiz": quiz, "algorithms_used": summary_obj.get("algorithms_used", [])}
            with self.lock:
                self.job_store[job_id]["status"] = "finished"
                self.job_store[job_id]["result"] = result
                self.metrics["jobs_finished"] += 1
        except Exception as e:
            with self.lock:
                self.job_store[job_id]["status"] = "errored"
                self.job_store[job_id]["error"] = str(e)
            logger.exception("Job failed", extra={"trace_id": job_id})

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self.job_store.get(job_id, {})

# ---------------- Utilities: export ----------------
def export_markdown(result: Dict[str, Any], path: str):
    """Save summary, notes and quiz to a Markdown file."""
    md = []
    md.append("# Automated Study Export\n")
    md.append("## Summary\n")
    md.append(result.get("summary", ""))
    md.append("\n## Notes\n")
    notes = result.get("notes")
    if isinstance(notes, dict):
        md.append(notes.get("formatted_text", ""))
    else:
        md.append(str(notes))
    md.append("\n## Quiz\n")
    for q in result.get("quiz", []):
        md.append(f"- **Q:** {q.get('q')}\n  - **A:** {q.get('a')}\n")
    md.append("\n## Algorithms Used\n")
    for a in result.get("algorithms_used", []):
        md.append(f"- {a}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md))
    logger.info(f"Exported markdown to {path}", extra={"trace_id": str(uuid.uuid4())})

def export_json(result: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Exported json to {path}", extra={"trace_id": str(uuid.uuid4())})

def export_text(result: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("SUMMARY\n\n")
        f.write(result.get("summary", "") + "\n\n")
        f.write("NOTES\n\n")
        notes = result.get("notes")
        if isinstance(notes, dict):
            f.write(notes.get("formatted_text", "") + "\n\n")
        else:
            f.write(str(notes) + "\n\n")
        f.write("QUIZ\n\n")
        for q in result.get("quiz", []):
            f.write(q.get("q") + "\n")
            f.write(q.get("a") + "\n\n")
    logger.info(f"Exported text to {path}", extra={"trace_id": str(uuid.uuid4())})

# ---------------- Instantiate services ----------------
llm = LLMClient(api_key=None)  # set OPENAI_API_KEY env var to enable real LLM
tools = {"pdf": PDFTool(), "video": VideoTranscriptTool(), "search": SearchTool()}
session_svc = InMemorySessionService()
memory = MemoryBank("memory_bank.json")
orch = Orchestrator(llm, tools, session_svc, memory)

# ---------------- Demo helpers ----------------
@trace
def demo_run_text_sync(text: str):
    """Run summarizer+notes+quiz synchronously for instant output (no threads)."""
    summ = SummarizerAgent(llm, tools)
    notes_ag = NoteAgent(llm, tools)
    quiz_ag = QuizAgent(llm, tools)
    summary_obj = summ.act(text)
    notes = notes_ag.act(summary_obj)
    quiz = quiz_ag.act(notes, n_questions=5)
    return {"summary": summary_obj.get("summary_md"), "notes": notes, "quiz": quiz, "algorithms_used": summary_obj.get("algorithms_used", [])}

@trace
def demo_run_text_example(text: str, user_id: str = "student123", wait: bool = True, timeout: int = 60) -> Dict[str, Any]:
    session = session_svc.create_session()
    job_id = orch.start_job(session.session_id, {"session_id": session.session_id, "source_type": "text", "text": text, "user_id": user_id})
    print(f"Started job {job_id} for session {session.session_id}")
    if not wait:
        return {"job_id": job_id}
    for _ in range(timeout):
        job = orch.get_job(job_id)
        status = job.get("status")
        print(f"Job status: {status}")
        if status == "finished":
            return job["result"]
        if status == "errored":
            raise RuntimeError(f"Job errored: {job.get('error')}")
        time.sleep(1)
    raise TimeoutError("Job did not finish in time")

def summarize_pdf(path: str, user_id: str = "student123", wait: bool = True):
    # reads pdf or falls back to mock
    text = tools["pdf"].extract_text(path)
    return demo_run_text_example(text, user_id=user_id, wait=wait)

# ---------------- If run interactively as script ----------------
if __name__ == "__main__":
    sample = (
        "This is a sample study text about machine learning. It covers supervised learning, loss functions, "
        "model evaluation, and regularization techniques. The text also mentions datasets and training loops. "
        "Finally, it summarizes best practices and references."
    )
    res = demo_run_text_sync(sample)
    print("\n--- SUMMARY ---\n")
    print(res["summary"])
    print("\n--- NOTES ---\n")
    print(res["notes"]["formatted_text"])
    print("\n--- QUIZ ---\n")
    for q in res["quiz"]:
        print(q["q"])
        print(q["a"])
        print()


