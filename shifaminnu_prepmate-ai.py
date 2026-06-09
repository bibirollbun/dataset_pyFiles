# PrepMate AI — Notebook Demo (MVP)
This notebook is a minimal, runnable demo of the PrepMate AI study assistant:
- Parse a syllabus / topic list
- Build a short multi-day study plan
- Generate practice questions (uses LLM if `OPENAI_API_KEY` is set; fallback templates otherwise)
- Grade answers and update simple memory (topic mastery)
# Cell 1 — Install (Kaggle: many libs are preinstalled; run anyway)
# !pip install openai tqdm
# Uncomment the above line if running locally and you want OpenAI support.
print("Ready. If using LLM, set OPENAI_API_KEY in your environment (do NOT store in repo).")
# Cell 2 — Imports & simple utilities
import os
import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pprint import pprint

# Simple helper
def ensure_list(x):
    if isinstance(x, list): return x
    return [x]
# Cell 3 — Optional LLM helper (OpenAI) with safe fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # set this in Kaggle secret or local env if you want LLM
USE_LLM = bool(OPENAI_API_KEY)

if USE_LLM:
    import openai
    openai.api_key = OPENAI_API_KEY

def llm_generate(prompt: str, max_tokens: int = 256) -> str:
    """
    Minimal wrapper: if OPENAI_API_KEY is set, call OpenAI completion (GPT-3.5/4 style).
    Otherwise raise RuntimeError to let caller fall back to templates.
    """
    if not USE_LLM:
        raise RuntimeError("LLM not enabled; no OPENAI_API_KEY found.")
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini", # use an available model on your platform or change to gpt-3.5-turbo
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
# Cell 4 — Data models
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
# Cell 5 — Syllabus parser (simple)
def parse_syllabus_text(text: str) -> List[Topic]:
    """
    Naive parser: split by newlines/commas; allow optional 'weight:N' in a line.
    Example lines:
      "Interrupts (weight:2)"
      "DMA"
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    topics = []
    for ln in lines:
        title = ln
        weight = 1.0
        if "weight" in ln.lower():
            # find weight:X
            try:
                parts = ln.split("weight")
                # parse number after :
                tail = parts[-1]
                num = ''.join(ch for ch in tail if (ch.isdigit() or ch=='.'))
                if num:
                    weight = float(num)
                title = ln.split("(")[0].strip()
            except Exception:
                pass
        # also split by comma if user provided comma-separated topics in one line
        if "," in title:
            for sub in title.split(","):
                sub = sub.strip()
                if sub:
                    topics.append(Topic(title=sub, weight=weight))
        else:
            topics.append(Topic(title=title, weight=weight))
    return topics

# example usage:
example_text = """Interrupts (weight:2)
DMA
Hashing, K-maps
System Testing (weight:1.5)
"""
topics_demo = parse_syllabus_text(example_text)
pprint([asdict(t) for t in topics_demo])
# Cell 6 — Scheduler (distribute topics to days)
def build_plan(user: UserProfile, days: int = 7) -> Dict[str, List[Dict[str,Any]]]:
    """
    Greedy plan: allocate time per topic based on weight and user's daily minutes.
    Returns dict like {'Day 1': [{'topic':'Interrupts','minutes':40}, ...], ...}
    """
    total_weight = sum(t.weight for t in user.topics)
    plan = {f"Day {i+1}": [] for i in range(days)}
    # compute minutes available per day
    avg_minutes = user.daily_minutes
    # create a list of topics ordered by weight descending for initial pass
    topics_sorted = sorted(user.topics, key=lambda t: -t.weight)
    # naive distribution: each day assign N topics so that total minutes ~ daily_minutes
    for day_idx in range(days):
        remaining = avg_minutes
        assigned = []
        # iterate topics and assign at most once per circulation
        for t in topics_sorted:
            if remaining <= 0:
                break
            # base minutes for topic proportional to weight
            minutes = max(10, int(avg_minutes * (t.weight/total_weight)))
            if minutes <= remaining:
                assigned.append({"topic": t.title, "minutes": minutes})
                remaining -= minutes
        # if nothing assigned (rare), assign smallest chunk to first topic
        if not assigned and topics_sorted:
            assigned.append({"topic": topics_sorted[0].title, "minutes": min(30, avg_minutes)})
        plan[f"Day {day_idx+1}"] = assigned
    return plan

# demo build
user = UserProfile(name="Shifa", exam_date="2025-12-01", daily_minutes=60, topics=topics_demo)
plan = build_plan(user, days=7)
pprint(plan)
# Cell 7 — Question generator (LLM-backed if available; fallback templates)
def generate_questions_for_topic(topic: str, n:int=5) -> List[Dict[str,Any]]:
    """
    Returns list of questions as dicts: {'q':..., 'choices': [...], 'answer': ..., 'type':'mcq'|'short'}
    If LLM available, use it; otherwise produce simple templated questions.
    """
    if USE_LLM:
        prompt = f"Generate {n} practice questions for the topic: {topic}. Provide questions as JSON list with fields: q, choices (or []), answer, difficulty (easy/medium/hard), type (mcq/short)."
        try:
            raw = llm_generate(prompt, max_tokens=512)
            # Try to parse JSON from response (LLM may wrap in text)
            # naive extraction: find first '[' and last ']'
            start = raw.find('[')
            end = raw.rfind(']')
            if start != -1 and end != -1:
                json_text = raw[start:end+1]
                return json.loads(json_text)
        except Exception as e:
            print("LLM generation failed, falling back to template:", e)
    # Fallback template generator
    out = []
    for i in range(n):
        if i % 3 == 0:
            # MCQ
            correct = f"Correct answer for {topic} Q{i+1}"
            choices = [correct] + [f"Wrong {j} for {topic}" for j in range(1,3)]
            random.shuffle(choices)
            out.append({"q": f"What is an important concept about {topic}? (MCQ)", "choices": choices, "answer": correct, "difficulty":"medium", "type":"mcq"})
        else:
            out.append({"q": f"Explain briefly: {topic} concept example {i+1}.", "choices": [], "answer": f"Expected points about {topic} - key idea", "difficulty":"easy", "type":"short"})
    return out

# demo gen for one topic
pprint(generate_questions_for_topic("Interrupts", n=4))
# Cell 8 — Simple grader
def grade_answer(question: Dict[str,Any], user_answer: str) -> float:
    """
    Return score in 0..1.
    For MCQ: exact match -> 1 else 0.
    For short: fuzzy check by keyword overlap (very naive).
    """
    if question.get("type") == "mcq":
        correct = question.get("answer","").strip().lower()
        return 1.0 if user_answer.strip().lower() == correct.lower() else 0.0
    else:
        # short answer grading: count common words
        expected = question.get("answer","").lower().split()
        given = user_answer.lower().split()
        common = set(expected).intersection(set(given))
        score = min(1.0, len(common)/max(1, len(set(expected))))
        return score

# demo grading
q = {"q":"What is an important concept about Interrupts? (MCQ)", "choices":["A","B"], "answer":"A", "type":"mcq"}
print("MCQ grade (A):", grade_answer(q, "A"))
print("Short grade:", grade_answer({"q":"Explain", "answer":"priority handling interrupt", "type":"short"}, "interrupt priority handling"))
# Cell 9 — Memory (very lightweight JSON memory)
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
    """
    Simple exponential moving average for mastery per topic per user.
    """
    key = f"{user}::{topic}"
    prev = mem.get(key, {"mastery":0.0, "count":0})
    alpha = 0.4
    new_mastery = (1-alpha)*prev["mastery"] + alpha*score
    mem[key] = {"mastery": new_mastery, "count": prev["count"]+1}
    return mem[key]

# demo
mem = load_memory()
m= update_topic_mastery(mem, "Shifa", "Interrupts", 0.8)
pprint(m)
save_memory(mem)  # writes to memory.json in the notebook directory
# Cell 10 — End-to-end demo function (simulate a study session for one day)
def run_daily_session(user: UserProfile, day_plan: List[Dict[str,Any]], mem_path="memory.json", interactive=False):
    mem = load_memory(mem_path)
    day_results = []
    for item in day_plan:
        topic = item["topic"]
        minutes = item["minutes"]
        print(f"\n--- Topic: {topic} | Planned minutes: {minutes}")
        # generate 3 questions quickly
        qs = generate_questions_for_topic(topic, n=3)
        topic_scores = []
        for qi, q in enumerate(qs, start=1):
            print(f"\nQ{qi}: {q['q']}")
            if q.get("choices"):
                for idx, ch in enumerate(q["choices"], start=1):
                    print(f"  {idx}. {ch}")
            # non-interactive mode: simulate student answer using heuristics
            if interactive:
                ans = input("Your answer: ")
            else:
                # simulate: if we have previous memory mastery, simulate better answers
                key = f"{user.name}::{topic}"
                prior = mem.get(key, {}).get("mastery", 0.0)
                # probability of correct proportional to mastery
                if q.get("type") == "mcq":
                    if random.random() < prior + 0.2:  # some chance to be correct
                        ans = q['answer']
                    else:
                        # random wrong choice
                        wrongs = [c for c in q.get("choices",[]) if c != q['answer']]
                        ans = wrongs[0] if wrongs else "I don't know"
                else:
                    # short answer simulate partial
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

# demo run for Day 1 plan
day1 = plan["Day 1"]
res = run_daily_session(user, day1, interactive=False)
pprint(res)
# Cell 11 — Adaptation: simple rescheduler using mastery
def adapt_plan_based_on_memory(user: UserProfile, plan: Dict[str,List[Dict[str,Any]]], mem_path="memory.json"):
    """
    Look at memory and bump topics with low mastery into earlier days.
    """
    mem = load_memory(mem_path)
    # compute mastery per topic
    masteries = {}
    for t in user.topics:
        key = f"{user.name}::{t.title}"
        masteries[t.title] = mem.get(key, {}).get("mastery", 0.0)
    print("Current masteries:", masteries)
    # For simplicity: topics with mastery < 0.5 get re-inserted on Day1
    to_prioritize = [t for t,m in masteries.items() if m < 0.5]
    if not to_prioritize:
        print("All topics reasonably mastered; no change.")
        return plan
    # Move prioritized topics to Day 1, replacing some tasks
    new_plan = dict(plan)  # shallow copy
    # Prepend prioritized topics to Day1 as 30-minute tasks
    day1_tasks = [{"topic":t, "minutes": min(40, user.daily_minutes//2)} for t in to_prioritize]
    # keep a few original tasks if space
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

# demo adaptation
plan = adapt_plan_based_on_memory(user, plan)
pprint(plan["Day 1"])
# Cell 12 — Save demo outputs (plan + memory summary) for submission attachments
def save_submission_artifacts(user: UserProfile, plan: Dict[str,Any], mem_path="memory.json", out_path="submission_artifacts.json"):
    mem = load_memory(mem_path)
    # prepare simple artifacts
    artifacts = {
        "user": {"name": user.name, "exam_date": user.exam_date, "daily_minutes": user.daily_minutes},
        "plan": plan,
        "masteries": {t.title: mem.get(f"{user.name}::{t.title}", {}) for t in user.topics}
    }
    with open(out_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"Saved artifacts to {out_path}")

save_submission_artifacts(user, plan)


