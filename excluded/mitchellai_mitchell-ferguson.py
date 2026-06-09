!pip install google-genai


import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
GOOGLE_API_KEY = secrets.get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("ready")


from google import genai
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
print(client.models.generate_content(model="gemini-2.0-flash", contents="connected").text)


import json
from pathlib import Path

# ---- Observability containers ----
TRACES = []   # list of episodes, each with: {"task": str, "steps": [tool_name, ...]}
METRICS = {
    "primitive_calls": [],  # number of primitive tool calls per episode
    "macro_calls": []       # we'll use this later (for now, log 0)
}

# ---- Research notebook (long-term memory) ----
NOTES_PATH = Path("/kaggle/working/research_notes.json")

def load_notes() -> dict:
    if NOTES_PATH.exists():
        with open(NOTES_PATH, "r") as f:
            return json.load(f)
    return {}

def save_notes(notes: dict) -> None:
    with open(NOTES_PATH, "w") as f:
        json.dump(notes, f, indent=2)

RESEARCH_NOTES = load_notes()
print("Observability + memory initialised.")


def begin_trace(task: str):
    """Start a new episode trace for a research task."""
    TRACES.append({"task": task, "steps": []})

def add_trace_step(tool_name: str):
    """Record that a tool was called in the current episode."""
    TRACES[-1]["steps"].append(tool_name)

def record_metrics(primitive_calls: int, macro_calls: int = 0):
    """Store simple per-episode metrics."""
    METRICS["primitive_calls"].append(primitive_calls)
    METRICS["macro_calls"].append(macro_calls)


def research(topic: str) -> dict:
    """
    Primitive tool.
    Use Gemini as a pseudo web search + aggregator to get key information on a topic.
    """
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Act as a web search + aggregator.\n"
                     f"Return key up-to-date facts and sources about:\n{topic}"
        ).text
        return {"status": "success", "data": resp}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def summarise(text: str) -> dict:
    """
    Primitive tool.
    Summarise long research text into a concise, well-structured paragraph.
    """
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Summarise the following research concisely:\n{text}"
        ).text
        return {"status": "success", "data": resp}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def write_note(topic: str, content: str) -> dict:
    """
    Primitive tool.
    Store a short note for a topic into the research notebook (long-term memory).
    """
    try:
        RESEARCH_NOTES[topic] = content
        save_notes(RESEARCH_NOTES)
        return {"status": "success", "data": f"Note saved for topic: {topic}"}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def read_notes(topic: str) -> dict:
    """
    Primitive tool.
    Retrieve any existing notes for a topic from the research notebook.
    """
    try:
        notes = RESEARCH_NOTES.get(topic)
        if notes is None:
            return {"status": "success", "data": None}
        return {"status": "success", "data": notes}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


TOOLS = {
    "research": research,
    "summarise": summarise,
    "write_note": write_note,
    "read_notes": read_notes,
}

def research_agent(task: str) -> str:
    """
    Very first version of the research agent.
    1. Check memory for existing notes.
    2. If none, call research + summarise.
    3. Save summary into notes.
    4. Return final answer.
    
    Also logs traces + metrics.
    """
    begin_trace(task)
    primitive_count = 0

    # 1) check memory
    add_trace_step("read_notes")
    primitive_count += 1
    memory_result = TOOLS["read_notes"](task)
    if memory_result["status"] == "error":
        return f"Memory error: {memory_result['error_message']}"

    if memory_result["data"] is not None:
        # We had notes already – reuse them
        answer = memory_result["data"]
    else:
        # 2) do fresh research
        add_trace_step("research")
        primitive_count += 1
        r = TOOLS["research"](task)
        if r["status"] == "error":
            return f"Research error: {r['error_message']}"

        add_trace_step("summarise")
        primitive_count += 1
        s = TOOLS["summarise"](r["data"])
        if s["status"] == "error":
            return f"Summarise error: {s['error_message']}"
        answer = s["data"]

        # 3) save to notes
        add_trace_step("write_note")
        primitive_count += 1
        TOOLS["write_note"](task, answer)

    # 4) record metrics
    record_metrics(primitive_calls=primitive_count, macro_calls=0)
    return answer


q = "Compare high-level strengths and weaknesses of Gemini vs GPT-4.1."
ans1 = research_agent(q)
print("FIRST RUN:\n", ans1[:400], "...\n")

ans2 = research_agent(q)
print("\nSECOND RUN (should hit memory, fewer primitive calls):\n", ans2[:400], "...\n")

print("\nTraces so far:", TRACES)
print("Primitive calls per episode:", METRICS["primitive_calls"])


# Extend metrics with a quality score (0–10) per episode
if "quality_score" not in METRICS:
    METRICS["quality_score"] = []

print("Metrics keys:", METRICS.keys())


import re

def judge_answer(task: str, answer: str) -> dict:
    """
    Judge agent.
    Ask Gemini to score the answer 0–10 for relevance, accuracy, and structure.
    Returns {"status": "success", "score": float} or error dict.
    """
    prompt = f"""
You are an expert evaluator.

Task:
{task}

Answer:
{answer}

Score the answer from 0 to 10 for overall quality
(accuracy, relevance, and structure). Respond with ONLY a number, nothing else.
"""
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        ).text.strip()

        # Extract first number in the response
        m = re.search(r"(\d+(\.\d+)?)", resp)
        if not m:
            return {"status": "error", "error_message": f"Could not parse score from: {resp}"}
        score = float(m.group(1))
        return {"status": "success", "score": score, "raw": resp}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def run_episode(task: str) -> str:
    """
    Run one full episode:
      - research_agent produces an answer
      - judge_agent scores it
      - metrics are updated
    Returns the answer for inspection.
    """
    answer = research_agent(task)
    
    j = judge_answer(task, answer)
    if j["status"] == "success":
        METRICS["quality_score"].append(j["score"])
        print(f"Judge score: {j['score']} (raw: {j['raw']})")
    else:
        METRICS["quality_score"].append(None)
        print(f"Judge error: {j['error_message']}")
    
    return answer


tasks = [
    "Compare high-level strengths and weaknesses of Gemini vs GPT-4.1.",
    "Summarise recent developments in AI agents and tool use.",
    "Compare high-level strengths and weaknesses of Gemini vs GPT-4.1.",  # repeat
    "Explain how compression relates to intelligence in AI models."
]

answers = []
for i, t in enumerate(tasks, start=1):
    print(f"\n=== EPISODE {i} ===")
    print("Task:", t)
    ans = run_episode(t)
    answers.append(ans[:300])  # store a snippet
    print("\nAnswer snippet:\n", ans[:300], "...")
    print("Primitive calls this episode:", METRICS["primitive_calls"][-1])

print("\nAll primitive_calls:", METRICS["primitive_calls"])
print("All quality scores:", METRICS["quality_score"])

baseline_primitive = METRICS["primitive_calls"].copy()
baseline_quality = METRICS["quality_score"].copy()


from collections import Counter

def get_tool_sequences(traces):
    """Return just the list of tool-name sequences from TRACES."""
    return [t["steps"] for t in traces]

def find_candidate_macros(traces, min_len=2, max_len=4, min_count=2):
    """
    Very simple 'compression' pass:
      - look at all subsequences of length 2..max_len
      - count how often they appear across episodes
      - return those that appear at least min_count times
    """
    seqs = get_tool_sequences(traces)
    counter = Counter()
    for seq in seqs:
        n = len(seq)
        for L in range(min_len, min(max_len, n) + 1):
            for i in range(n - L + 1):
                sub = tuple(seq[i:i+L])
                counter[sub] += 1
    candidates = {sub: cnt for sub, cnt in counter.items() if cnt >= min_count}
    return candidates

candidates = find_candidate_macros(TRACES, min_len=2, max_len=4, min_count=2)
print("Candidate subsequences (pattern -> count):")
for pattern, count in candidates.items():
    print(pattern, ":", count)


def research_and_summarise(topic: str) -> dict:
    """
    Macro tool.
    Compresses the common pattern:
      research(topic) -> summarise(result) -> write_note(topic, summary)
    Returns the summary, and still updates memory.
    """
    # Call primitives internally, but from the agent's point of view
    # this is just ONE tool call.
    r = research(topic)
    if r["status"] == "error":
        return r

    s = summarise(r["data"])
    if s["status"] == "error":
        return s

    w = write_note(topic, s["data"])
    if w["status"] == "error":
        return w

    return {"status": "success", "data": s["data"]}


TOOLS["research_and_summarise"] = research_and_summarise
print("Registered macro tool: research_and_summarise")


def reset_state():
    """Clear traces & metrics for a fresh 'after macros' run."""
    TRACES.clear()
    METRICS["primitive_calls"].clear()
    METRICS["macro_calls"].clear()
    METRICS["quality_score"].clear()
    print("State reset for macro-enabled run.")
    print("Current notes keys:", list(RESEARCH_NOTES.keys()))  # debug / documentation


def research_agent_with_macros(task: str) -> str:
    """
    Version of the agent that uses the macro tool when it needs fresh research.

    Behaviour:
      1) Always check memory first.
      2) If memory hit -> reuse note (1 primitive call, 0 macros).
      3) If no memory -> call 'research_and_summarise' macro
         (1 primitive call for read_notes, 1 macro call).
    Also logs traces + metrics.
    """
    begin_trace(task)
    primitive_count = 0
    macro_count = 0

    # 1) check memory
    add_trace_step("read_notes")
    primitive_count += 1
    memory_result = TOOLS["read_notes"](task)
    if memory_result["status"] == "error":
        record_metrics(primitive_calls=primitive_count, macro_calls=macro_count)
        return f"Memory error: {memory_result['error_message']}"

    if memory_result["data"] is not None:
        print("[agent] Hit memory – reusing existing note.")
        answer = memory_result["data"]
    else:
        print("[agent] No memory – using 'research_and_summarise' macro.")
        # 2) use macro instead of three primitives
        add_trace_step("research_and_summarise")
        macro_count += 1
        m = TOOLS["research_and_summarise"](task)
        if m["status"] == "error":
            record_metrics(primitive_calls=primitive_count, macro_calls=macro_count)
            return f"Macro error: {m['error_message']}"
        answer = m["data"]

    record_metrics(primitive_calls=primitive_count, macro_calls=macro_count)
    return answer


def run_episode_with_macros(task: str) -> str:
    """
    Same as run_episode, but uses the macro-enabled agent.
    """
    answer = research_agent_with_macros(task)
    j = judge_answer(task, answer)
    if j["status"] == "success":
        METRICS["quality_score"].append(j["score"])
        print(f"Judge score: {j['score']} (raw: {j['raw']})")
    else:
        METRICS["quality_score"].append(None)
        print(f"Judge error: {j['error_message']}")
    return answer


reset_state()

# First two tasks are OLD (memory hit), last two are NEW (macro needed)
tasks = [
    # these already have notes from the baseline agent
    "Compare high-level strengths and weaknesses of Gemini vs GPT-4.1.",
    "Summarise recent developments in AI agents and tool use.",
    # these are new topics -> no notes -> will use macro
    "AI agents for healthcare decision support in 2025",
    "AI agents for education and tutoring in 2025",
]

answers = []
for i, t in enumerate(tasks, start=1):
    print(f"\n=== MACRO EPISODE {i} ===")
    print("Task:", t)
    ans = run_episode_with_macros(t)
    answers.append(ans[:300])  # store a snippet
    print("Primitive calls this episode:", METRICS["primitive_calls"][-1],
          "| Macro calls:", METRICS["macro_calls"][-1])


# Capture metrics from the macro-enabled run
macro_primitive = METRICS["primitive_calls"].copy()
macro_quality   = METRICS["quality_score"].copy()

print("macro_primitive:", macro_primitive)
print("macro_quality:", macro_quality)


def propose_macros(traces, min_len=2, max_len=4, min_count=2):
    """Mine + ask for human approval of a macro that compresses a common tool pattern."""
    import re
    from collections import Counter

    # 1) gather sequences
    seqs = [t["steps"] for t in traces]
    counter = Counter()

    # 2) mine subsequences
    for seq in seqs:
        n = len(seq)
        for L in range(min_len, min(max_len, n) + 1):
            for i in range(n - L + 1):
                sub = tuple(seq[i:i+L])
                counter[sub] += 1

    # 3) return frequent ones requiring approval
    candidates = {p: c for p, c in counter.items() if c >= min_count}

    if not candidates:
        print("No suitable macro found.")
        return None

    # 4) ask human to approve only the best candidate
    best = max(candidates, key=candidates.get)
    choice = input(f"Approve this pattern {best} as macro? (y/n): ").strip().lower()

    if choice == "y":
        METRICS["macro_calls"].append(1)
        print("Macro approved ✅")
        return best
    else:
        print("Macro rejected ❌")
        return None


def propose_macros(traces, min_len=2, max_len=4, min_count=2):
    """
    Human-in-the-loop macro discovery.

    1. Mine frequent subsequences from traces.
    2. For each candidate, ask the human whether to approve it as a macro.
    (For now we only implement the 'research -> summarise -> write_note' macro.)
    """
    candidates = find_candidate_macros(traces, min_len=min_len, max_len=max_len, min_count=min_count)
    print("Discovered candidate patterns:")
    for pattern, count in candidates.items():
        print(f"  {pattern}  (count={count})")

    # Simple interactive approval for our main pattern
    target = ("research", "summarise", "write_note")
    if target in candidates:
        print("\nCandidate macro found:", target)
        choice = input("Approve this pattern as 'research_and_summarise' macro? [y/n] ").strip().lower()
        if choice == "y":
            print("✅ Macro 'research_and_summarise' is approved (code already defined and registered).")
        else:
            print("❌ Macro rejected (you can skip using research_and_summarise).")
    else:
        print("\nNo suitable macro pattern found for 'research -> summarise -> write_note'.")


hard_task = (
    "Write a structured 3–4 paragraph brief on the current state of AI agents: "
    "cover typical architectures (tool use, multi-agent setups), key challenges "
    "(observability, evaluation, reliability), and why compression of behaviour "
    "might matter for future agents."
)

print("=== HARD TASK WITH MACRO-ENABLED AGENT ===")
answer = run_episode_with_macros(hard_task)

print("\nAnswer snippet:\n", answer[:800], "...")
print("\nPrimitive calls (this episode):", METRICS["primitive_calls"][-1],
      "| Macro calls:", METRICS["macro_calls"][-1],
      "| Judge score:", METRICS["quality_score"][-1])


print("=== NOTES MEMORY FILE ===")
print(NOTES_PATH.read_text()[:500])


print("Primitive calls:", METRICS["primitive_calls"])
print("Macro calls:", METRICS["macro_calls"])
print("Judge scores:", METRICS["quality_score"])
print("Num traces:", len(TRACES))

macro_primitive = METRICS["primitive_calls"].copy()
macro_quality = METRICS["quality_score"].copy()


import matplotlib.pyplot as plt
import numpy as np

# --- Episodes indices ---
n_base  = len(baseline_primitive)
n_macro = len(macro_primitive)

episodes_baseline = np.arange(1, n_base + 1)
episodes_macro    = np.arange(n_base + 1, n_base + n_macro + 1)

# ==============================
# 1) Line plot: per-episode calls
# ==============================
plt.figure(figsize=(6, 4))

plt.plot(
    episodes_baseline,
    baseline_primitive,
    marker="o",
    linestyle="-",
    label="Before macros",
)

plt.plot(
    episodes_macro,
    macro_primitive,
    marker="o",
    linestyle="-",
    label="After macros (macro tool)",
)

# Mark where macros are introduced
macro_start = n_base + 0.5
plt.axvline(macro_start, linestyle="--")
plt.text(
    macro_start + 0.1,
    max(baseline_primitive + macro_primitive),
    "Macros introduced",
    rotation=90,
    va="top",
)

plt.xlabel("Episode")
plt.ylabel("Primitive tool calls")
plt.title("Primitive tool calls per episode\nbefore and after macros")
plt.xticks(np.arange(1, n_base + n_macro + 1))  # only integer ticks
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 2) Bar chart: average calls
# ==============================
before_mean = np.mean(baseline_primitive)
after_mean  = np.mean(macro_primitive)

plt.figure(figsize=(4, 4))
plt.bar(["Before macros", "After macros"], [before_mean, after_mean])
plt.ylabel("Average primitive tool calls")
plt.title("Average tool calls before vs after macros")

# Annotate exact values on top of bars
for i, v in enumerate([before_mean, after_mean]):
    plt.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom")

plt.ylim(0, max(before_mean, after_mean) + 0.5)
plt.tight_layout()
plt.grid(axis="y", linestyle=":")
plt.show()

