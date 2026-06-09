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


# Cell 1: Metadata & Imports
# Run this first. Standard imports for the prototype.

import os
import time
import glob
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple


# Cell 2: LLM Client abstraction using Gemini API via google-generativeai
# API Key should be stored in Kaggle Secrets as GOOGLE_API_KEY.

import google.generativeai as genai

class LLMClient:
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Base interface for any LLM client."""
        raise NotImplementedError


class GeminiLLM(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        """
        Load API key from environment or Kaggle secret.
        Example in Kaggle:
            GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise RuntimeError("GOOGLE_API_KEY not found. Please set it in Kaggle Secrets.")

        genai.configure(api_key=api_key)
        self.model = model
        self.client = genai.GenerativeModel(self.model)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Generate output using Google's Gemini API."""
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        return response.text.strip()


# Cell 3: MockLLM for offline deterministic testing

class MockLLM(LLMClient):
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        # Deterministic stub outputs to let you test the orchestration without an API key.
        lower = prompt.lower()
        if "create a structured outline" in lower or "structured outline" in lower:
            return (
                "1. Introduction: why the topic matters\n"
                "2. Background & motivation\n"
                "3. Architecture and design\n"
                "4. Implementation details\n"
                "   - Code snippet: example()\n"
                "5. Example usage\n"
                "6. Conclusion & next steps\n"
            )
        if "write a full blog post" in lower or "write a full blog post in markdown" in lower:
            return (
                "# Sample Blog Post Title\n\n"
                "This is a generated blog post based on the outline.\n\n"
                "## Implementation\n\n"
                "```python\n"
                "def example():\n"
                "    return 'hello'\n"
                "```\n\n"
                "## Conclusion\n\nThis is the end."
            )
        if "generate 2 linkedin post" in lower or "generate 2 linkedin" in lower or "generate 2 linkedin post variations" in lower:
            return "LinkedIn Post 1: ...\nLinkedIn Post 2: ...\nTwitter: ...\nTwitter: ..."
        # Fallback
        return "LLM placeholder output for testing."


# Cell 4: Tools - analyze_codebase and save_blog_post_to_file

def analyze_codebase(path: str, max_files: int = 50) -> str:
    """Traverse `path` and return concatenated snippets for context (first 1200 chars per file)."""
    if not path:
        return ""
    if not os.path.exists(path):
        return f"Path not found: {path}"
    snippets = []
    files = glob.glob(os.path.join(path, "**", "*.*"), recursive=True)[:max_files]
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                text = fh.read()
            snippet = text[:1200]
            rel = os.path.relpath(f, path)
            snippets.append(f"--- {rel} ---\n{snippet}\n")
        except UnicodeDecodeError:
            try:
                with open(f, "r", encoding="latin-1") as fh:
                    text = fh.read()
                snippet = text[:1200]
                rel = os.path.relpath(f, path)
                snippets.append(f"--- {rel} (latin-1) ---\n{snippet}\n")
            except Exception:
                snippets.append(f"--- {os.path.relpath(f, path)} --- (skipped due to read error)\n")
        except Exception:
            snippets.append(f"--- {os.path.relpath(f, path)} --- (skipped)\n")
    return "\n".join(snippets) if snippets else "No codebase files found."

def save_blog_post_to_file(md_text: str, out_path: str):
    """Save markdown text to out_path. Create directories if needed."""
    if not out_path:
        raise ValueError("out_path must be provided")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(md_text)
    return out_path


# Cell 5: Validation checkers and ValidationResult dataclass

@dataclass
class ValidationResult:
    ok: bool
    reasons: List[str]

def outline_validation_checker(outline_text: str) -> ValidationResult:
    """Simple outline validator: requires at least 3 headings and presence of intro & conclusion."""
    lines = [l.strip() for l in outline_text.splitlines() if l.strip()]
    headings = [l for l in lines if l[0].isdigit() or l.endswith(":") or l.startswith("-")]
    reasons = []
    if len(headings) < 3:
        reasons.append("Outline has fewer than 3 headings.")
    if not any("intro" in l.lower() or "introduction" in l.lower() for l in lines):
        reasons.append("No introduction found in outline.")
    if not any("conclusion" in l.lower() for l in lines):
        reasons.append("No conclusion found.")
    return ValidationResult(ok=(len(reasons) == 0), reasons=reasons)

def blog_post_validation_checker(post_text: str) -> ValidationResult:
    """Simple blog post validator: length, presence of code blocks, presence of top-level title."""
    reasons = []
    if len(post_text.split()) < 200:
        reasons.append("Post is too short (<200 words).")
    if "```" not in post_text:
        reasons.append("No code block detected; include code examples for technical posts.")
    first_lines = post_text.strip().splitlines()
    if not first_lines or not (first_lines[0].startswith("#") or "<h1" in first_lines[0].lower()):
        reasons.append("Post may lack a top-level title.")
    return ValidationResult(ok=(len(reasons) == 0), reasons=reasons)


# Cell 6: LoopAgent base + BlogPlanner, BlogWriter, SocialMediaWriter

class LoopAgent:
    def __init__(self, name: str, llm: LLMClient, max_attempts: int = 3, delay: float = 0.5):
        self.name = name
        self.llm = llm
        self.max_attempts = max_attempts
        self.delay = delay

    def run_with_validation(self, prompt_template: str, validator_fn, **kwargs) -> Tuple[str, List[str]]:
        """Format prompt_template with kwargs, run LLM until validator returns ok or attempts exhausted."""
        attempts = 0
        last_reasons = []
        last_response = ""
        while attempts < self.max_attempts:
            attempts += 1
            prompt = prompt_template.format(**kwargs)
            print(f"[{self.name}] Attempt {attempts}: calling LLM")
            try:
                response = self.llm.generate(prompt)
            except Exception as e:
                response = f"LLM error: {e}"
            last_response = response
            valid = validator_fn(response)
            if valid.ok:
                print(f"[{self.name}] Validation passed.")
                return response, []
            last_reasons = valid.reasons
            print(f"[{self.name}] Validation failed: {last_reasons}")
            time.sleep(self.delay)
        print(f"[{self.name}] Exhausted attempts. Returning last response with reasons.")
        return last_response, last_reasons

class BlogPlanner:
    def __init__(self, llm: LLMClient):
        self.agent = LoopAgent("robust_blog_planner", llm)

    def plan(self, topic: str, audience: str, codebase_context: str = "") -> Tuple[str, List[str]]:
        prompt = (
            "Create a structured outline for a technical blog post about: '{topic}'.\n"
            "Target audience: {audience}.\n"
            "If codebase context is provided include sections for code snippets and architecture notes.\n"
            "Requirements: include Introduction, Background, Implementation (with subsections), Example Usage, Conclusion.\n\nOutline:\n"
        )
        return self.agent.run_with_validation(prompt, outline_validation_checker,
                                              topic=topic, audience=audience, codebase_context=codebase_context)

class BlogWriter:
    def __init__(self, llm: LLMClient):
        self.agent = LoopAgent("robust_blog_writer", llm)

    def write(self, outline: str, codebase_context: str = "", tone: str = "professional", length: str = "long") -> Tuple[str, List[str]]:
        prompt = (
            "Write a full blog post in Markdown based on the following outline:\n\n{outline}\n\n"
            "Use tone: {tone}. Expected length: {length} (aim for 700-1200+ words).\n"
            "If code context is provided include code examples from context and explain them.\n\nWrite the full blog post now:\n"
        )
        return self.agent.run_with_validation(prompt, blog_post_validation_checker,
                                              outline=outline, codebase_context=codebase_context, tone=tone, length=length)

class SocialMediaWriter:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(self, title: str, summary: str) -> str:
        prompt = (
            "Generate 2 LinkedIn post variations and 2 Twitter/X variations to promote a blog titled: '{title}'.\n"
            "Use short, attention grabbing hooks. Include relevant hashtags. Provide suggested image caption.\n\nSummary: {summary}\n"
        )
        return self.llm.generate(prompt.format(title=title, summary=summary), max_tokens=250)


# Cell 7: Orchestrator - InteractiveBloggerAgent

class InteractiveBloggerAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.planner = BlogPlanner(llm)
        self.writer = BlogWriter(llm)
        self.social = SocialMediaWriter(llm)

    def create_post(self, topic: str, audience: str = "technical developers", repo_path: Optional[str] = None,
                    tone: str = "professional", out_md_path: str = "output/post.md") -> Dict[str,Any]:
        code_context = analyze_codebase(repo_path) if repo_path else ""
        outline, outline_reasons = self.planner.plan(topic, audience, code_context)
        post, post_reasons = self.writer.write(outline, code_context, tone)
        saved_path = None
        try:
            saved_path = save_blog_post_to_file(post, out_md_path)
        except Exception as e:
            print("Error saving file:", e)
        title_line = post.splitlines()[0] if post else topic
        summary = (post[:800] + "...") if len(post) > 800 else post
        social_posts = self.social.generate(title_line, summary)
        return {
            "topic": topic,
            "audience": audience,
            "outline": outline,
            "outline_validation_issues": outline_reasons,
            "post": post,
            "post_validation_issues": post_reasons,
            "saved_path": saved_path,
            "social_posts": social_posts
        }


# Cell 8: Small automated editor (style tweaks). Simple, optional.

def blog_editor_auto_fixes(post_md: str) -> str:
    """Apply simple automated edits: ensure title exists, add trailing newline, minimal fixes."""
    if not post_md:
        return post_md
    lines = post_md.splitlines()
    if not lines[0].startswith("#"):
        # Prepend a simple title if missing
        lines.insert(0, "# Untitled Post")
    # Ensure there is a newline at EOF
    if not post_md.endswith("\n"):
        lines.append("")
    return "\n".join(lines)


# Cell 9: Demo runner - chooses LLM (OpenAI if available), runs an example

def demo_run(topic: str = None, repo_path: Optional[str] = None, use_openai: bool = False):
    # Choose LLM implementation: prefer OpenAI if requested and available
    llm = None
    if use_openai and OpenAILLM is not None:
        try:
            llm = OpenAILLM(os.getenv("OPENAI_API_KEY"))
            print("Using OpenAI LLM.")
        except Exception as e:
            print("OpenAI init failed:", e)
    if llm is None:
        llm = MockLLM()
        print("Using MockLLM (offline mode).")

    agent = InteractiveBloggerAgent(llm)
    topic = topic or "Building a real-time license plate detection service with YOLO"
    out_path = "output/generated_post.md"
    result = agent.create_post(topic, audience="software engineers", repo_path=repo_path, tone="professional", out_md_path=out_path)
    # Apply editor fixes
    result["post"] = blog_editor_auto_fixes(result["post"])
    if result["saved_path"]:
        # rewrite with editor applied
        save_blog_post_to_file(result["post"], result["saved_path"])
    return result

# If running interactively, call demo_run() to see a full end-to-end example.


# Cell 10: Run multiple demo tests

# ------------------------------
# Helper: run & display results
# ------------------------------
def test_topic(topic, repo_path=None, use_openai=False):
    print("\n" + "="*80)
    print(f"ðŸ”¥ Testing Topic: {topic}")
    print("="*80)

    res = demo_run(topic=topic, repo_path=repo_path, use_openai=use_openai)

    print("\n=== Outline ===\n")
    print(res["outline"][:1000])

    print("\n=== Post Validation Issues ===")
    print(res["post_validation_issues"])

    print("\n=== Saved Path ===")
    print(res["saved_path"])

    print("\n=== Social Posts ===")
    print(res["social_posts"])
    print("\n\n")


# ------------------------------------
# Run multiple test cases (samples)
# ------------------------------------

topics = [
    "How to build a YOLO-based real-time license plate detector",
    "Building a ChatGPT-like chatbot using RNNs and Attention",
    "End-to-end guide to building a Flask API for machine learning models",
    "How to implement a real-time face recognition attendance system",
    "Beginner guide to vector databases and RAG architecture",
    "Building a scalable backend using FastAPI and PostgreSQL",
    "Design and architecture of multi-agent systems using LLMs",
]

for t in topics:
    test_topic(t, repo_path=None, use_openai=False)




