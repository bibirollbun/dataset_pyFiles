"""
Smart Task Automation Agent (single-file)

How to use
1. Install (recommended):
   pip install openai PyPDF2

2. If you want better planning/summarization, set OPENAI_API_KEY in env:
   export OPENAI_API_KEY="sk-..."

3. Run examples at the bottom or call functions from a notebook.

Description
- Lightweight, modular agent that can: read text/PDF, summarize, organize files, and create simple reminders.
- Uses OpenAI for planning and summarization if API key is available; otherwise falls back to simple heuristics.

This file is intentionally self-contained and easy to extend with new tools.
"""

import os
import json
import shutil
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional

# Optional imports
try:
    import openai
except Exception:
    openai = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


# ---------------------- Configuration ----------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY

# Simple local reminder store
REMINDERS_FILE = Path("agent_reminders.json")


# ---------------------- Tools ----------------------

def read_text_file(path: str) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_pdf(path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 not installed. Install with `pip install PyPDF2` to read PDFs.")
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def summarize_text(text: str, max_tokens: int = 200) -> str:
    """Summarize using OpenAI when available, otherwise fallback to naive summarization."""
    text = text.strip()
    if not text:
        return "(no text)"

    if openai and OPENAI_API_KEY:
        prompt = (
            "You are a helpful assistant. Provide a concise summary (3-6 short bullets) of the following text:\n\n" + text
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini" if False else "gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            summary = resp["choices"][0]["message"]["content"].strip()
            return summary
        except Exception as e:
            print("OpenAI call failed, falling back to simple summary:", e)

    # fallback: naive summarization using textwrap
    # take first N sentences
    sentences = text.replace("\n", " ").split('.')
    short = '.'.join(sentences[:3]).strip()
    if not short.endswith('.'):
        short += '.'
    return textwrap.shorten(short + ' ' + (' '.join(sentences[3:5]) if len(sentences) > 3 else ''), width=400)


def organize_files_by_extension(src_dir: str, dst_base: str = "organized_files") -> Dict[str, List[str]]:
    """Move files into folders by extension. Returns a mapping of folder->moved files."""
    src = Path(src_dir)
    dst_base = Path(dst_base)
    dst_base.mkdir(exist_ok=True)
    moved = {}

    for p in src.iterdir():
        if p.is_file():
            ext = p.suffix.lower().lstrip('.') or 'noext'
            folder = dst_base / ext
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / p.name
            shutil.copy2(p, dest)
            moved.setdefault(ext, []).append(str(dest))
    return moved


def add_reminder(title: str, description: str, when: Optional[str] = None) -> None:
    reminders = []
    if REMINDERS_FILE.exists():
        try:
            reminders = json.loads(REMINDERS_FILE.read_text())
        except Exception:
            reminders = []
    reminders.append({"title": title, "description": description, "when": when})
    REMINDERS_FILE.write_text(json.dumps(reminders, indent=2))


# ---------------------- Planner / Agent ----------------------

def llm_plan(task: str) -> List[str]:
    """Ask the LLM to break a user's instruction into steps. Returns list of steps."""
    if openai and OPENAI_API_KEY:
        prompt = (
            "Break the following user instruction into a short ordered list of concrete steps that a script could perform. "
            "Return only the numbered list.\n\nInstruction:\n" + task
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
            )
            content = resp["choices"][0]["message"]["content"].strip()
            # parse numbered list
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            steps = []
            for l in lines:
                # remove leading numbering like '1.' or '-'
                step = l
                if step and (step[0].isdigit() or step.startswith('-')):
                    # drop up to first space after dot
                    parts = step.split('.', 1)
                    if len(parts) == 2:
                        step = parts[1].strip()
                    else:
                        step = step.lstrip('-').strip()
                steps.append(step)
            return steps
        except Exception as e:
            print("LLM planning failed:", e)

    # fallback planner: simple heuristics
    # split by commas and ' then '
    heur = [s.strip() for s in task.replace(' then ', ',').split(',') if s.strip()]
    if len(heur) > 1:
        return heur
    # if single sentence, make plausible steps
    return [task]


class SmartAgent:
    def __init__(self):
        self.tools = {
            'read_text_file': read_text_file,
            'extract_text_from_pdf': extract_text_from_pdf,
            'summarize_text': summarize_text,
            'organize_files_by_extension': organize_files_by_extension,
            'add_reminder': add_reminder,
        }

    def run(self, instruction: str) -> Dict[str, Any]:
        print("Planning...")
        steps = llm_plan(instruction)
        print(f"Planned {len(steps)} step(s):")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")

        results = {
            'instruction': instruction,
            'steps': [],
        }

        for step in steps:
            res = self.execute_step(step)
            results['steps'].append({'step': step, 'result': res})
        return results

    def execute_step(self, step: str) -> Any:
        # naive routing: check for keywords, call appropriate tool
        s = step.lower()
        try:
            if 'pdf' in s or step.endswith('.pdf') or 'document' in s:
                # find a filename in step
                words = step.split()
                fn = None
                for w in words:
                    if w.lower().endswith('.pdf'):
                        fn = w
                        break
                if not fn:
                    return 'No PDF filename provided in step.'
                txt = self.tools['extract_text_from_pdf'](fn)
                summary = self.tools['summarize_text'](txt)
                return {'text_length': len(txt), 'summary': summary}

            if 'summar' in s or 'summary' in s or 'summarize' in s:
                # look for "file <name>" or else summarize provided inline
                # If step contains 'file:' pattern
                if ':' in step and step.count(':') == 1 and step.strip().split()[-1].endswith('.txt'):
                    fn = step.strip().split()[-1]
                    txt = self.tools['read_text_file'](fn)
                    return self.tools['summarize_text'](txt)
                # else just summarize remainder
                return self.tools['summarize_text'](step)

            if 'organize' in s or 'sort' in s or 'move files' in s:
                # expect a directory path in the step
                words = step.split()
                dir_candidate = None
                for w in words:
                    if os.path.isdir(w):
                        dir_candidate = w
                        break
                if not dir_candidate:
                    # fallback - use current directory
                    dir_candidate = '.'
                moved = self.tools['organize_files_by_extension'](dir_candidate)
                return {'organized': moved}

            if 'remind' in s or 'reminder' in s or 'remind me' in s:
                # naive parsing: "Remind me to X at Y"
                # This is a simple demonstration
                parts = step.split(' to ', 1)
                if len(parts) == 2:
                    action = parts[1]
                else:
                    action = step
                add_reminder(title=action[:50], description=action, when=None)
                return 'Reminder added.'

            # default: echo + summarize
            return self.tools['summarize_text'](step)
        except Exception as e:
            return {'error': str(e)}


# ---------------------- Example usage ----------------------
if __name__ == '__main__':
    agent = SmartAgent()

    # Example 1: Summarize a text instruction
    instruction1 = "Summarize the following text: This project builds a smart agent that automates tasks like summarizing emails and organizing files. Make a short 3-bullet summary."
    print('\n--- Example 1 ---')
    out1 = agent.run(instruction1)
    print(json.dumps(out1, indent=2))

    # Example 2: Organize files in a sample directory (will copy files into 'organized_files')
    print('\n--- Example 2 ---')
    instruction2 = "Organize files in ./sample_files by extension"
    out2 = agent.run(instruction2)
    print(json.dumps(out2, indent=2))

    # Example 3: Add a reminder
    print('\n--- Example 3 ---')
    instruction3 = "Remind me to submit the Kaggle capstone writeup tomorrow morning"
    out3 = agent.run(instruction3)
    print(json.dumps(out3, indent=2))

    print('\nDone.')


