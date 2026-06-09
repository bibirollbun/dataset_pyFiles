# ============================================================
# Agentic-RAG Codebase Documenter - Kaggle Notebook
# ============================================================

# Optional: install deps (google-generativeai + scikit-learn)
# !pip install -q google-generativeai scikit-learn
# !pip install -U google-generativeai

import os
import time
import ast
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai

print("=== Files under /kaggle/input ===")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ------------------------------------------------------------
# 1) Get Gemini API key (Kaggle secret or env var)
# ------------------------------------------------------------
def get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if key is None:
        try:
            from kaggle_secrets import UserSecretsClient
            us = UserSecretsClient()
            key = us.get_secret("GOOGLE_API_KEY")
        except Exception as e:
            print("Could not load GEMINI_API_KEY from kaggle_secrets:", e)

    if key is None:
        raise ValueError(
            "Gemini API key not found. Set GEMINI_API_KEY/GOOGLE_API_KEY "
            "or create a Kaggle secret named 'GEMINI_API_KEY'."
        )
    return key

GEMINI_API_KEY = get_gemini_api_key()
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.0-flash-001"

# ------------------------------------------------------------
# 2) Find sample_repo path under /kaggle/input
# ------------------------------------------------------------
def find_sample_repo(base="/kaggle/input") -> str:
    for root, dirs, files in os.walk(base):
        # Look for a directory literally named 'sample_repo'
        for d in dirs:
            if d == "sample_repo":
                return os.path.join(root, d)
    raise FileNotFoundError(
        "Could not find 'sample_repo' folder under /kaggle/input. "
        "Make sure you uploaded sample_repo.zip as an input dataset."
    )

REPO_PATH = find_sample_repo()
print("\nUsing codebase repo path:", REPO_PATH)



# ============================================================
# LLM wrapper + logging / observability
# ============================================================

class GeminiLLM:
    def __init__(self, model_name: str = GEMINI_MODEL_NAME):
        self.model = genai.GenerativeModel(model_name)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        prompt = system_prompt + "\n\nUser:\n" + user_prompt
        response = self.model.generate_content(prompt)
        try:
            return response.text
        except AttributeError:
            return "".join(part.text for part in response.candidates[0].content.parts)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("agentic_rag")

def log_agent_call(agent_name: str):
    """Decorator to log calls and measure duration."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"[{agent_name}] Starting {func.__name__}")
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"[{agent_name}] Finished {func.__name__} in {duration:.2f}s")
            return result
        return wrapper
    return decorator


# ============================================================
# Code model (functions), loader, parser, retrieval, session, memory
# ============================================================

@dataclass
class CodeFunction:
    file_path: str
    name: str
    start_line: int
    end_line: int
    source: str
    docstring: Optional[str] = None

    @property
    def loc(self) -> int:
        return self.end_line - self.start_line + 1


class CodebaseLoader:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def list_python_files(self) -> List[str]:
        paths = []
        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if f.endswith(".py"):
                    paths.append(os.path.join(root, f))
        return paths

    def read_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


class CodeParsingTool:
    """Parses Python files and extracts functions with basic metadata."""
    def parse_functions_in_file(self, file_path: str, source: str) -> List[CodeFunction]:
        tree = ast.parse(source)
        functions: List[CodeFunction] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno
                end_line = getattr(node, "end_lineno", node.lineno)
                src_lines = source.splitlines()
                func_source = "\n".join(src_lines[start_line - 1: end_line])
                docstring = ast.get_docstring(node)
                functions.append(
                    CodeFunction(
                        file_path=file_path,
                        name=node.name,
                        start_line=start_line,
                        end_line=end_line,
                        source=func_source,
                        docstring=docstring,
                    )
                )
        return functions


class RetrievalIndex:
    """TF-IDF index over function source code."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.matrix = None
        self.functions: List[CodeFunction] = []

    def build(self, functions: List[CodeFunction]):
        self.functions = functions
        corpus = [f.source for f in functions]
        if not corpus:
            self.matrix = None
            return
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[CodeFunction, float]]:
        if self.matrix is None or not self.functions:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idx_scores = sorted(
            enumerate(sims), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [(self.functions[i], float(score)) for i, score in idx_scores]


@dataclass
class Session:
    id: str
    repo_path: str
    created_at: float = field(default_factory=time.time)


class SessionService:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}

    def create_session(self, session_id: str, repo_path: str) -> Session:
        session = Session(id=session_id, repo_path=repo_path)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)


class MemoryBank:
    """Stores generated documentation keyed by file and function name."""
    def __init__(self):
        self.docs: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def save_doc(self, cf: CodeFunction, docstring: str, summary: str, refactor: str):
        self.docs[(cf.file_path, cf.name)] = {
            "docstring": docstring,
            "summary": summary,
            "refactor": refactor,
        }

    def get_doc(self, cf: CodeFunction) -> Optional[Dict[str, Any]]:
        return self.docs.get((cf.file_path, cf.name))



# ============================================================
# Agent context & base agent
# ============================================================

@dataclass
class AgentContext:
    session: Session
    memory: MemoryBank
    index: RetrievalIndex
    codebase_loader: CodebaseLoader
    parser: CodeParsingTool
    llm: GeminiLLM


class AgentBase:
    def __init__(self, name: str, ctx: AgentContext):
        self.name = name
        self.ctx = ctx

    def log(self, msg: str):
        logger.info(f"[{self.name}] {msg}")




# Cell 5 - Agents: Retriever, Reasoner, Documenter, Refactor, Orchestrator
# (Paste this directly into your Kaggle notebook)

class RetrieverAgent(AgentBase):

    @log_agent_call("RetrieverAgent")
    def build_index(self) -> List[CodeFunction]:
        files = self.ctx.codebase_loader.list_python_files()
        self.log(f"Found {len(files)} Python files")
        all_funcs: List[CodeFunction] = []
        for path in files:
            src = self.ctx.codebase_loader.read_file(path)
            funcs = self.ctx.parser.parse_functions_in_file(path, src)
            all_funcs.extend(funcs)
        self.log(f"Parsed {len(all_funcs)} functions across repo")
        self.ctx.index.build(all_funcs)
        return all_funcs

    @log_agent_call("RetrieverAgent")
    def retrieve_related(self, query: str, top_k: int = 5) -> List[CodeFunction]:
        results = self.ctx.index.search(query, top_k=top_k)
        self.log(f"Retrieved {len(results)} related functions for query: {query}")
        return [cf for cf, score in results]


class ReasonerAgent(AgentBase):

    SYSTEM_PROMPT = (
        "You are a senior software engineer analyzing Python ETL code. "
        "Given a function, explain what it does, its responsibilities, and any potential issues "
        "such as mixed concerns, excessive length, or unclear naming."
    )

    @log_agent_call("ReasonerAgent")
    def analyze_function(self, cf: CodeFunction) -> str:
        user_prompt = f"""
File: {cf.file_path}
Function: {cf.name}

Code:
```python
{cf.source}
```

Explain what this function does, what its main responsibilities are, and note any design smells.
"""
        return self.ctx.llm.generate(self.SYSTEM_PROMPT, user_prompt)


class DocumenterAgent(AgentBase):

    DOCSTRING_SYSTEM_PROMPT = (
        "You are an expert Python developer. Write concise, clear docstrings "
        "for functions, following Google-style docstrings."
    )

    SUMMARY_SYSTEM_PROMPT = (
        "You write concise documentation for developers who need to understand an ETL pipeline quickly."
    )

    @log_agent_call("DocumenterAgent")
    def generate_docstring(self, cf: CodeFunction, analysis: str) -> str:
        user_prompt = f"""
File: {cf.file_path}
Function: {cf.name}

Code:
```python
{cf.source}
```

Analysis:
{analysis}

Write a Google-style Python docstring that accurately describes this function,
its arguments, returns, and side effects.
"""
        return self.ctx.llm.generate(self.DOCSTRING_SYSTEM_PROMPT, user_prompt)

    @log_agent_call("DocumenterAgent")
    def generate_summary(self, cf: CodeFunction, analysis: str) -> str:
        user_prompt = f"""
You are documenting an ETL pipeline.

Based on the function and analysis below, write a short (2-4 sentence) summary for developer documentation.

Function: {cf.name}
File: {cf.file_path}

Analysis:
{analysis}
"""
        return self.ctx.llm.generate(self.SUMMARY_SYSTEM_PROMPT, user_prompt)


class RefactorAgent(AgentBase):

    SYSTEM_PROMPT = (
        "You are a software architect. Suggest refactorings for a given Python function, "
        "focusing on splitting responsibilities, improving naming, and simplifying control flow. "
        "Do NOT rewrite the full code, only describe concrete refactor suggestions."
    )

    @log_agent_call("RefactorAgent")
    def suggest_refactors(self, cf: CodeFunction, analysis: str) -> str:
        user_prompt = f"""
File: {cf.file_path}
Function: {cf.name}
Lines of code: {cf.loc}

Code:
```python
{cf.source}
```

Analysis:
{analysis}

Suggest specific refactorings that would make this function easier to understand and maintain.
"""
        return self.ctx.llm.generate(self.SYSTEM_PROMPT, user_prompt)


class OrchestratorAgent(AgentBase):

    @log_agent_call("OrchestratorAgent")
    def analyze_repo(self, max_functions: int = 10, loc_threshold: int = 5):
        retriever = RetrieverAgent("RetrieverAgent", self.ctx)
        reasoner = ReasonerAgent("ReasonerAgent", self.ctx)
        documenter = DocumenterAgent("DocumenterAgent", self.ctx)
        refactor = RefactorAgent("RefactorAgent", self.ctx)

        all_funcs = retriever.build_index()
        target_funcs = sorted(all_funcs, key=lambda cf: cf.loc, reverse=True)[:max_functions]

        results = []
        for cf in target_funcs:
            if cf.loc < loc_threshold:
                continue

            self.log(f"Processing {cf.file_path}::{cf.name} (LOC={cf.loc})")

            existing = self.ctx.memory.get_doc(cf)
            if existing:
                self.log("Using cached documentation from memory")
                results.append((cf, existing))
                continue

            analysis = reasoner.analyze_function(cf)
            docstring = documenter.generate_docstring(cf, analysis)
            summary = documenter.generate_summary(cf, analysis)
            refactor_suggestions = refactor.suggest_refactors(cf, analysis)

            self.ctx.memory.save_doc(cf, docstring, summary, refactor_suggestions)
            results.append((cf, {
                "analysis": analysis,
                "docstring": docstring,
                "summary": summary,
                "refactor": refactor_suggestions
            }))

        return results



## ▶️ Cell 6 — Initialize context, run orchestrator, show results

# ============================================================
# Initialize context & run the multi-agent pipeline
# ============================================================

session_service = SessionService()
session = session_service.create_session("session-1", repo_path=REPO_PATH)

codebase_loader = CodebaseLoader(root_dir=REPO_PATH)
parser = CodeParsingTool()
index = RetrievalIndex()
memory = MemoryBank()
llm = GeminiLLM()

ctx = AgentContext(
    session=session,
    memory=memory,
    index=index,
    codebase_loader=codebase_loader,
    parser=parser,
    llm=llm,
)

orchestrator = OrchestratorAgent("OrchestratorAgent", ctx)

results = orchestrator.analyze_repo(max_functions=20, loc_threshold=1)

print("\n=== Documentation Results ===")
for cf, docs in results:
    print("=" * 80)
    print(f"File: {cf.file_path}")
    print(f"Function: {cf.name} (LOC={cf.loc})")
    print("\n--- Generated Docstring ---\n")
    print(docs["docstring"])
    print("\n--- Summary ---\n")
    print(docs["summary"])
    print("\n--- Refactor Suggestions ---\n")
    print(docs["refactor"])



# ============================================================
# Export output files to /kaggle/working/output/
# ============================================================

import json
import os

OUTPUT_DIR = "/kaggle/working/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

docstrings = {}
summaries = {}
refactors = {}

for cf, docs in results:
    key = f"{cf.file_path}::{cf.name}"
    docstrings[key] = docs["docstring"]
    summaries[key] = docs["summary"]
    refactors[key] = docs["refactor"]

# Save JSON files
with open(f"{OUTPUT_DIR}/docstrings.json", "w") as f:
    json.dump(docstrings, f, indent=4)

with open(f"{OUTPUT_DIR}/summaries.json", "w") as f:
    json.dump(summaries, f, indent=4)

with open(f"{OUTPUT_DIR}/refactor_suggestions.json", "w") as f:
    json.dump(refactors, f, indent=4)

# Save full Markdown report
with open(f"{OUTPUT_DIR}/full_report.md", "w") as f:
    for key in docstrings:
        f.write(f"## {key}\n\n")
        f.write("### Docstring\n")
        f.write(docstrings[key] + "\n\n")
        f.write("### Summary\n")
        f.write(summaries[key] + "\n\n")
        f.write("### Refactor Suggestions\n")
        f.write(refactors[key] + "\n\n")
        f.write("---\n\n")

print("Exported files to:", OUTPUT_DIR)



def simple_token_overlap(a: str, b: str) -> float:
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

ground_truth = "Loads a CSV file into a pandas DataFrame from the specified path."
target_func_name = "load_csv"

found = False
for cf, docs in results:
    if cf.name == target_func_name:
        generated_doc = docs["docstring"]
        score = simple_token_overlap(ground_truth, generated_doc)
        print("\n=== Evaluation ===")
        print(f"Overlap score for {target_func_name}: {score:.2f}")
        print("\nGenerated docstring:\n")
        print(generated_doc)
        found = True
        break

if not found:
    print(f"No function named {target_func_name} found in results. "
          "Try lowering loc_threshold in analyze_repo().")


