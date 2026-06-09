"""
GenMind Pro - Kaggle-ready capstone prototype (Python)
Features included:
- Personalized Roadmap Generator
- Smart Prompt Engine
- Quiz Generator
- Notes export (Markdown)
- Progress tracker (simple JSON persistence)
- Optional integration with Google Gemini (GenAI) via the google-genai SDK

Usage:
- By default the notebook runs in offline fallback mode (USE_API=False) so Kaggle will run even without network/keys.
- To enable real Gemini calls:
    1. pip install -U google-genai
    2. Set environment variable GEMINI_API_KEY (or GOOGLE_API_KEY) in Kaggle Secrets.
    3. Set USE_API = True (or pass use_api=True to functions)

References: Gemini SDK quickstart and docs.
"""

import os
import json
import hashlib
import random
from typing import List, Dict, Any
from datetime import datetime

# --- Configuration ---
USE_API = False  # Set to True when you configure the Gemini API
API_PROVIDER = os.environ.get("GENMIND_API_PROVIDER", "gemini")
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GENMIND_API_KEY", "")
DEFAULT_SEED = 42
OUTPUT_FOLDER = "genmind_outputs"

# --- Prompt templates ---
ROADMAP_PROMPT = (
    "Create a personalized learning roadmap for a user with the following specification:\n"
    "Level: {level}\nGoal: {goal}\nDuration_weeks: {weeks}\nTopics_of_interest: {topics}\n\n"
    "Output format: JSON with fields: 'week', 'topics', 'milestones', 'resources' (list of URLs or short descriptions)."
)

IDEA_PROMPT_TEMPLATE = (
    "Generate {n} creative, original, and actionable ideas for the following topic:\n\n"
    "Topic: {topic}\n\n"
    "Constraints / user notes: {notes}\n\n"
    "Format: a JSON list of objects with fields: title, short_description, potential_users, difficulty_estimate"
)

QUIZ_PROMPT_TEMPLATE = (
    "Generate {n} multiple-choice questions (with 4 options and a single correct answer) for the topic: {topic}.\n"
    "Output format: JSON list of objects with fields: question, options (list), correct_index (0-3), explanation"
)

CONTENT_PROMPT_TEMPLATE = (
    "Write a {kind} for the topic '{topic}' with tone '{tone}' and length '{length}'.\n"
    "Requirements:\n- Include a strong opening sentence\n- Provide 3 actionable bullet points (if applicable)\n- Keep it clear and concise\n\nOutput: Plain text."
)

# --- Utilities ---

def deterministic_hash_seed(text: str, base_seed: int = DEFAULT_SEED) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    v = int(h[:16], 16)
    return (v ^ base_seed) % (2**31 - 1)


def ensure_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def save_json(name: str, data: Any) -> str:
    ensure_output_folder()
    safe = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).rstrip()
    path = os.path.join(OUTPUT_FOLDER, f"{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

# --- Local fallback generators (deterministic) ---

def local_generate_roadmap(level: str, goal: str, weeks: int, topics: List[str]) -> List[Dict]:
    seed = deterministic_hash_seed(level + goal + ''.join(topics))
    rng = random.Random(seed)
    roadmap = []
    topic_pool = topics if topics else ["fundamentals", "projects", "practice"]
    for w in range(1, max(1, weeks) + 1):
        tcount = min(len(topic_pool), max(1, rng.randint(1, 3)))
        chosen = [rng.choice(topic_pool) for _ in range(tcount)]
        roadmap.append({
            "week": w,
            "topics": list(dict.fromkeys(chosen)),
            "milestones": [f"Complete {len(chosen)} learning items"],
            "resources": [f"Study {topic} - tutorial/guide" for topic in chosen]
        })
    return roadmap


def local_generate_ideas(topic: str, n: int =5, notes: str ="") -> List[Dict]:
    if not topic or not topic.strip():
        raise ValueError("Topic must be non-empty.")
    seed = deterministic_hash_seed(topic + notes)
    rng = random.Random(seed)
    base_actions = ["automate","summarize","personalize","visualize","analyze","recommend","annotate","translate","organize","prototype"]
    domains = ["education","content marketing","productivity","healthcare","cybersecurity","developer tools","small business","e-learning","finance","creative writing"]
    ideas = []
    for i in range(n):
        action = rng.choice(base_actions)
        domain = rng.choice(domains)
        title = f"{action.capitalize()} {domain} insights for '{topic}'" if rng.random() > 0.3 else f"{topic.capitalize()} {action} toolkit"
        short_description = f"A lightweight AI assistant that can {action} {topic} content for {domain} users. It produces practical, actionable steps."
        potential_users = rng.sample(["students","creators","developers","small businesses","educators","marketers"], k=2)
        difficulty_estimate = rng.choice(["low","medium","high"])
        ideas.append({"title":title, "short_description":short_description, "potential_users":potential_users, "difficulty_estimate":difficulty_estimate})
    return ideas


def local_generate_quiz(topic: str, n:int=5) -> List[Dict]:
    if not topic:
        raise ValueError("Topic must be non-empty for quiz generation.")
    seed = deterministic_hash_seed(topic + str(n))
    rng = random.Random(seed)
    quiz = []
    for i in range(n):
        correct = rng.randint(0,3)
        options = [f"Option {j+1} for {topic}" for j in range(4)]
        quiz.append({
            "question": f"What is {topic} concept {i+1}?",
            "options": options,
            "correct_index": correct,
            "explanation": f"Answer {correct+1} is correct because..."
        })
    return quiz


def local_generate_content(topic: str, kind: str="blog_intro", tone:str="professional", length:str="short") -> str:
    if not topic:
        raise ValueError("Topic must be non-empty")
    seed = deterministic_hash_seed(topic + kind + tone + length)
    rng = random.Random(seed)
    opening = f"{topic.capitalize()} — a concise guide to the essentials."
    bullets = [
        f"{rng.choice(['Start by','Begin with','First,'])} defining the scope and users.",
        f"{rng.choice(['Use existing datasets','Collect small examples','Prototype quickly'])} to iterate fast.",
        f"{rng.choice(['Measure outcomes','Request feedback','A/B test ideas'])} and refine."
    ]
    body = f"{opening}\n\nThis piece gives you a quick set of actionable steps and ideas to build and adapt {topic}."
    return body + "\n\nActionable points:\n- " + "\n- ".join(bullets)

# --- Gemini API layer (best-effort wrapper) ---

def _setup_gemini_client(api_key: str):
    """Attempts to set up the Google GenAI SDK client. Returns the module object if successful."""
    try:
        # Prefer the newer package name if available
        import google.generativeai as genai
    except Exception:
        try:
            # Some docs also show `from google import genai`
            from google import genai
        except Exception as e:
            raise RuntimeError("Google GenAI SDK not installed. Run `pip install -U google-genai` in the notebook.")
    # Configure client API key if possible
    try:
        if hasattr(genai, 'configure'):
            if api_key:
                genai.configure(api_key=api_key)
    except Exception:
        # Non-fatal: proceed and let call fail with clearer message
        pass
    return genai


def call_gemini(prompt: str, model: str = "gemini-1.5", max_output_tokens: int = 512) -> str:
    """Call Gemini (GenAI) and return text output. This function attempts several common SDK patterns.
    If something fails, it raises informative errors so you can fix environment / key.
    """
    genai = _setup_gemini_client(API_KEY)
    # Try common SDK patterns
    # 1) GenerativeModel interface
    try:
        if hasattr(genai, 'GenerativeModel'):
            model_obj = genai.GenerativeModel(model)
            if hasattr(model_obj, 'generate_text'):
                resp = model_obj.generate_text(prompt=prompt, max_output_tokens=max_output_tokens)
                # resp may be string or object with .text
                return resp.text if hasattr(resp, 'text') else str(resp)
            if hasattr(model_obj, 'generate_content'):
                resp = model_obj.generate_content(prompt=prompt, max_output_tokens=max_output_tokens)
                # parse a bunch of possible response shapes
                if hasattr(resp, 'candidates') and resp.candidates:
                    c = resp.candidates[0]
                    return getattr(c, 'output', getattr(c, 'content', str(c)))
                return str(resp)
    except Exception:
        # fall through to next attempt
        pass

    # 2) genai.generate_text helper
    try:
        if hasattr(genai, 'generate_text'):
            out = genai.generate_text(model=model, prompt=prompt, max_output_length=max_output_tokens)
            # out may be dict-like
            if isinstance(out, str):
                return out
            if isinstance(out, dict) and 'candidates' in out and out['candidates']:
                return out['candidates'][0].get('content', str(out['candidates'][0]))
            return str(out)
    except Exception:
        pass

    # 3) genai.chat.create pattern
    try:
        if hasattr(genai, 'chat') and hasattr(genai.chat, 'create'):
            resp = genai.chat.create(model=model, messages=[{"role":"user","content":prompt}])
            # resp.choices[0].message.content or resp.last or similar
            if hasattr(resp, 'last') and hasattr(resp.last, 'content'):
                return resp.last.content
            if hasattr(resp, 'choices') and resp.choices:
                ch = resp.choices[0]
                return getattr(ch, 'message', getattr(ch, 'content', str(ch)))
            return str(resp)
    except Exception as e:
        # Raise an informative error
        raise RuntimeError(f"Gemini API call failed: {e}\nEnsure GEMINI_API_KEY/GOOGLE_API_KEY is set and the google-genai package is installed.")

    # If none of the patterns matched
    raise RuntimeError("Could not call Gemini: SDK pattern not recognised. Please check the google-genai docs and installed package version.")

# --- High-level functions that use API if enabled else fallback ---

def generate_roadmap(level: str, goal: str, weeks: int, topics: List[str], use_api: bool = USE_API) -> List[Dict]:
    prompt = ROADMAP_PROMPT.format(level=level, goal=goal, weeks=weeks, topics=', '.join(topics))
    if use_api:
        raw = call_gemini(prompt)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            # else fallback
            return local_generate_roadmap(level, goal, weeks, topics)
        except Exception:
            return local_generate_roadmap(level, goal, weeks, topics)
    else:
        return local_generate_roadmap(level, goal, weeks, topics)


def generate_ideas(topic: str, n: int =5, notes: str ="", use_api: bool = USE_API) -> List[Dict]:
    prompt = IDEA_PROMPT_TEMPLATE.format(topic=topic, n=n, notes=notes)
    if use_api:
        raw = call_gemini(prompt)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            return local_generate_ideas(topic, n, notes)
        except Exception:
            return local_generate_ideas(topic, n, notes)
    else:
        return local_generate_ideas(topic, n, notes)


def generate_quiz(topic: str, n:int=5, use_api: bool = USE_API) -> List[Dict]:
    prompt = QUIZ_PROMPT_TEMPLATE.format(topic=topic, n=n)
    if use_api:
        raw = call_gemini(prompt)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            return local_generate_quiz(topic, n)
        except Exception:
            return local_generate_quiz(topic, n)
    else:
        return local_generate_quiz(topic, n)


def generate_content(topic: str, kind:str="blog_intro", tone:str="professional", length:str="short", use_api: bool = USE_API) -> str:
    prompt = CONTENT_PROMPT_TEMPLATE.format(topic=topic, kind=kind, tone=tone, length=length)
    if use_api:
        return call_gemini(prompt)
    else:
        return local_generate_content(topic, kind, tone, length)

# --- Progress tracker simple implementation ---

def load_progress(user_id: str) -> Dict:
    path = os.path.join(OUTPUT_FOLDER, f"progress_{user_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"user_id": user_id, "completed": [], "last_updated": None}


def update_progress(user_id: str, item: str) -> str:
    data = load_progress(user_id)
    if item not in data["completed"]:
        data["completed"].append(item)
    data["last_updated"] = datetime.utcnow().isoformat()
    return save_json(f"progress_{user_id}", data)

# --- Notes export (Markdown) ---

def export_notes_md(title: str, sections: List[Dict[str,str]]) -> str:
    ensure_output_folder()
    safe = "".join(c for c in title if c.isalnum() or c in (" ","_","-")).rstrip()
    path = os.path.join(OUTPUT_FOLDER, f"{safe}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for sec in sections:
            f.write(f"## {sec.get('heading','Notes')}\n\n")
            f.write(sec.get('content','') + "\n\n")
    return path

# --- Demo / Kaggle friendly runner ---

def run_demo(use_api: bool = USE_API):
    topic = "cybersecurity learning path"
    print(f"Running GenMind Pro demo (USE_API={use_api}) for topic: {topic}\n")

    roadmap = generate_roadmap(level="Beginner", goal="Get internship", weeks=6, topics=["networking","linux","web security"], use_api=use_api)
    print("Roadmap (first 2 weeks):")
    for wk in roadmap[:2]:
        print(wk)

    ideas = generate_ideas(topic, n=5, notes="Capstone-focused ideas", use_api=use_api)
    print("\nIdeas sample:")
    for i,it in enumerate(ideas,1):
        print(i, it['title'])

    quiz = generate_quiz("web security", n=3, use_api=use_api)
    print("\nQuiz sample:")
    for q in quiz:
        print(q['question'])

    content = generate_content("cybersecurity learning path", kind="blog_intro", tone="friendly", length="short", use_api=use_api)
    print("\nContent sample:\n", content[:300])

    # Export notes
    md_path = export_notes_md("GenMind Demo Notes", [{"heading":"Roadmap","content":json.dumps(roadmap[:2], indent=2)}, {"heading":"Ideas","content":json.dumps(ideas[:2], indent=2)}])
    print(f"\nExported notes to: {md_path}")

    # Save combined outputs
    outpath = save_json("genmind_demo_output", {"roadmap":roadmap, "ideas":ideas, "quiz":quiz, "content":content})
    print(f"Saved demo output JSON to: {outpath}")

# --- Self-tests to ensure no runtime error in offline mode ---

def _run_self_tests():
    a = generate_ideas("test topic", n=3, notes="note1", use_api=False)
    b = generate_ideas("test topic", n=3, notes="note1", use_api=False)
    assert a == b
    c = generate_ideas("another topic", n=3, notes="note1", use_api=False)
    assert a != c
    s = generate_content("sample topic", kind="blog_intro", tone="casual", length="short", use_api=False)
    assert isinstance(s, str) and len(s) > 0
    path = save_json("test_project", {"x":1})
    assert os.path.exists(path)
    md = export_notes_md("test md", [{"heading":"h","content":"c"}])
    assert os.path.exists(md)
    print("All self-tests passed.")

if __name__ == "__main__":
    run_demo(use_api=False)
    _run_self_tests()


