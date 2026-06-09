"""
PrepMate AI — Notebook Demo (MVP)
Save as prep_mate_notebook_demo.py or paste into a Kaggle notebook cell-by-cell.
Features:
- Parse a syllabus / topic list
- Build a short multi-day study plan
- Generate practice questions (LLM if OPENAI_API_KEY present; fallback templates otherwise)
- Grade answers and update simple memory (topic mastery)
- Save submission artifacts
"""

import os
import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pprint import pprint

# ---------------------------
# Configuration / LLM Setup
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # set this in Kaggle secrets or local env
USE_LLM = bool(OPENAI_API_KEY)

if USE_LLM:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
    except Exception as e:
        print("OpenAI library not available or failed to import:", e)
        USE_LLM = False

def llm_generate(prompt: str, max_tokens: int = 256) -> str:
    """
    Minimal wrapper that calls OpenAI ChatCompletion if configured, otherwise raises.
    Adjust model name for the environment (gpt-4o-mini / gpt-3.5-turbo etc).
    """
    if not USE_LLM:
        raise RuntimeError("LLM not enabled; no OPENAI_API_KEY found or openai import failed.")
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---------------------------
# Data models
# ---------------------------
@dataclass
class Topic:
    title: str
    weight: float = 1.0
    notes: str = ""
    mastery: float = 0.0  # 0.0..1.0 based on grading history

@dataclass
class UserProfile:
    name: str
    exam_date: str
    daily_minutes: int
    topics: List[Topic]
    memory_path: str = "memory.json"

# ---------------------------
# Syllabus parser
# ---------------------------
def parse_syllabus_text(text: str) -> List[Topic]:
    """
    Naive parser: split by newlines/commas; optional 'weight:N' in a line.
    Example:
        "Interrupts (weight:2)"
        "DMA"
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    topics: List[Topic] = []
    for ln in lines:
        title = ln
        weight = 1.0
        if "weight" in ln.lower():
            try:
                parts = ln.split("weight")
                tail = parts[-1]
                num = ''.join(ch for ch in tail if (ch.isdigit() or ch=='.'))
                if num:
                    weight = float(num)
                title = ln.split("(")[0].strip()
            except Exception:
                pass
        if "," in title:
            for sub in title.split(","):
                sub = sub.strip()
                if sub:
                    topics.append(Topic(title=sub, weight=weight))
        else:
            topics.append(Topic(title=title, weight=weight))
    return topics

# ---------------------------
# Scheduler
# ---------------------------
def build_plan(user: UserProfile, days: int = 7) -> Dict[str, List[Dict[str,Any]]]:
    """
    Greedy plan: allocate minutes per topic based on weight and user's daily minutes.
    Returns dict like {'Day 1': [{'topic':'Interrupts','minutes':40}, ...], ...}
    """
    total_weight = sum(t.weight for t in user.topics) or 1.0
    plan = {f"Day {i+1}": [] for i in range(days)}
    avg_minutes = user.daily_minutes
    topics_sorted = sorted(user.topics, key=lambda t: -t.weight)
    for day_idx in range(days):
        remaining = avg_minutes
        assigned = []
        for t in topics_sorted:
            if remaining <= 0:
                break
            minutes = max(10, int(avg_minutes * (t.weight/total_weight)))
            if minutes <= remaining:
                assigned.append({"topic": t.title, "minutes": minutes})
                remaining -= minutes
        if not assigned and topics_sorted:
            assigned.append({"topic": topics_sorted[0].title, "minutes": min(30, avg_minutes)})
        plan[f"Day {day_idx+1}"] = assigned
    return plan

# ---------------------------
# Question generator
# ---------------------------
def generate_questions_for_topic(topic: str, n:int=5) -> List[Dict[str,Any]]:
    """
    Returns list of questions as dicts: {'q':..., 'choices': [...], 'answer': ..., 'type':'mcq'|'short'}
    If LLM available, uses it; otherwise template fallback.
    """
    if USE_LLM:
        prompt = (
            f"Generate {n} practice questions for the topic: {topic}. "
            "Provide answers and difficulty levels. Output as a JSON list of objects "
            "with fields: q, choices (array or []), answer, difficulty, type (mcq/short)."
        )
        try:
            raw = llm_generate(prompt, max_tokens=512)
            start = raw.find('[')
            end = raw.rfind(']')
            if start != -1 and end != -1:
                json_text = raw[start:end+1]
                return json.loads(json_text)
        except Exception as e:
            print("LLM generation failed or response couldn't be parsed, falling back:", e)

    # Fallback template generator
    out = []
    for i in range(n):
        if i % 3 == 0:
            correct = f"Correct answer for {topic} Q{i+1}"
            choices = [correct] + [f"Wrong {j} for {topic}" for j in range(1,3)]
            random.shuffle(choices)
            out.append({"q": f"What is an important concept about {topic}? (MCQ)", "choices": choices, "answer": correct, "difficulty":"medium", "type":"mcq"})
        else:
            out.append({"q": f"Explain briefly: {topic} concept example {i+1}.", "choices": [], "answer": f"Expected points about {topic} - key idea", "difficulty":"easy", "type":"short"})
    return out

# ---------------------------
# Grader
# ---------------------------
def grade_answer(question: Dict[str,Any], user_answer: str) -> float:
    """
    Return score in 0..1.
    For MCQ: exact match -> 1 else 0.
    For short: naive overlap-based partial scoring.
    """
    if question.get("type") == "mcq":
        correct = question.get("answer","").strip().lower()
        return 1.0 if user_answer.strip().lower() == correct.lower() else 0.0
    else:
        expected = question.get("answer","").lower().split()
        given = user_answer.lower().split()
        common = set(expected).intersection(set(given))
        score = min(1.0, len(common)/max(1, len(set(expected))))
        return score

# ---------------------------
# Memory (lightweight)
# ---------------------------
def load_memory(path="memory.json") -> Dict[str,Any]:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_memory(mem: Dict[str,Any], path="memory.json"):
    with open(path, "w") as f:
        json.dump(mem, f, indent=2)

def update_topic_mastery(mem: Dict[str,Any], user: str, topic: str, score: float):
    key = f"{user}::{topic}"
    prev = mem.get(key, {"mastery":0.0, "count":0})
    alpha = 0.4
    new_mastery = (1-alpha)*prev["mastery"] + alpha*score
    mem[key] = {"mastery": new_mastery, "count": prev["count"]+1}
    return mem[key]

# ---------------------------
# End-to-end session run
# ---------------------------
def run_daily_session(user: UserProfile, day_plan: List[Dict[str,Any]], mem_path="memory.json", interactive=False):
    mem = load_memory(mem_path)
    day_results = []
    for item in day_plan:
        topic = item["topic"]
        minutes = item["minutes"]
        print(f"\n--- Topic: {topic} | Planned minutes: {minutes}")
        qs = generate_questions_for_topic(topic, n=3)
        topic_scores = []
        for qi, q in enumerate(qs, start=1):
            print(f"\nQ{qi}: {q['q']}")
            if q.get("choices"):
                for idx, ch in enumerate(q["choices"], start=1):
                    print(f"  {idx}. {ch}")
            # interactive vs simulated answer
            if interactive:
                ans = input("Your answer: ")
            else:
                key = f"{user.name}::{topic}"
                prior = mem.get(key, {}).get("mastery", 0.0)
                if q.get("type") == "mcq":
                    if random.random() < prior + 0.2:
                        ans = q['answer']
                    else:
                        wrongs = [c for c in q.get("choices",[]) if c != q['answer']]
                        ans = wrongs[0] if wrongs else "I don't know"
                else:
                    if random.random() < prior + 0.2:
                        ans = q['answer']
                    else:
                        ans = "partial " + q['answer'].split()[0]
            score = grade_answer(q, ans)
            print(f" -> answer: {ans[:80]} | score: {score:.2f}")
            topic_scores.append(score)
        avg_score = sum(topic_scores)/len(topic_scores) if topic_scores else 0.0
        update = update_topic_mastery(mem, user.name, topic, avg_score)
        print(f"Topic mastery updated: {update}")
        day_results.append({"topic":topic, "avg_score":avg_score, "update":update})
    save_memory(mem, mem_path)
    return day_results

# ---------------------------
# Adaptation / Rescheduler
# ---------------------------
def adapt_plan_based_on_memory(user: UserProfile, plan: Dict[str,List[Dict[str,Any]]], mem_path="memory.json"):
    mem = load_memory(mem_path)
    masteries = {}
    for t in user.topics:
        key = f"{user.name}::{t.title}"
        masteries[t.title] = mem.get(key, {}).get("mastery", 0.0)
    print("Current masteries:", masteries)
    to_prioritize = [t for t,m in masteries.items() if m < 0.5]
    if not to_prioritize:
        print("All topics reasonably mastered; no change.")
        return plan
    new_plan = dict(plan)
    day1_tasks = [{"topic":t, "minutes": min(40, user.daily_minutes//2)} for t in to_prioritize]
    remaining_minutes = user.daily_minutes - sum(d["minutes"] for d in day1_tasks)
    day1_old = []
    for task in plan.get("Day 1", []):
        if remaining_minutes <= 0: break
        if task["minutes"] <= remaining_minutes:
            day1_old.append(task)
            remaining_minutes -= task["minutes"]
    new_plan["Day 1"] = day1_tasks + day1_old
    print("Adapted Day 1 plan:", new_plan["Day 1"])
    return new_plan

# ---------------------------
# Save artifacts for submission
# ---------------------------
def save_submission_artifacts(user: UserProfile, plan: Dict[str,Any], mem_path="memory.json", out_path="submission_artifacts.json"):
    mem = load_memory(mem_path)
    artifacts = {
        "user": {"name": user.name, "exam_date": user.exam_date, "daily_minutes": user.daily_minutes},
        "plan": plan,
        "masteries": {t.title: mem.get(f"{user.name}::{t.title}", {}) for t in user.topics}
    }
    with open(out_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"Saved artifacts to {out_path}")

# ---------------------------
# Demo main (example)
# ---------------------------
def demo_run():
    example_text = """Interrupts (weight:2)
DMA
Hashing, K-maps
System Testing (weight:1.5)
"""
    topics_demo = parse_syllabus_text(example_text)
    user = UserProfile(name="Shifa", exam_date="2025-12-01", daily_minutes=60, topics=topics_demo)
    print("Parsed topics:")
    pprint([asdict(t) for t in topics_demo])

    plan = build_plan(user, days=7)
    print("\nInitial plan:")
    pprint(plan)

    # Run Day 1 simulated session
    day1 = plan["Day 1"]
    res = run_daily_session(user, day1, interactive=False)
    pprint(res)

    # Adapt plan based on memory
    plan = adapt_plan_based_on_memory(user, plan)
    pprint(plan["Day 1"])

    # Save artifacts
    save_submission_artifacts(user, plan)

if __name__ == "__main__":
    demo_run()


