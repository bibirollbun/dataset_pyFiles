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


# 0. Setup & authentication

import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' "
        f"to your Kaggle secrets. Details: {e}"
    )



# Core Python libs
import json
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Metrics / plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ADK & Gemini
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.genai import types

# Optional: BM25 for RAG
!pip install -q rank_bm25
from rank_bm25 import BM25Okapi

import warnings
warnings.filterwarnings("ignore")

print("âœ… ADK & dependencies imported successfully.")



# Tiny knowledge base for RAG (could be replaced by a Kaggle dataset)
kb_docs = [
    {
        "id": "d1",
        "title": "Google and Kaggle AI Agents Intensive",
        "text": "The 5-Day AI Agents Intensive teaches multi-agent systems, tools, memory, and evaluation using Gemini and the Agent Development Kit.",
    },
    {
        "id": "d2",
        "title": "Gemini 2.0 Flash",
        "text": "Gemini 2.0 Flash is a fast, cost-efficient multimodal model with a 1M-token context window designed for agentic workloads.",
    },
    {
        "id": "d3",
        "title": "Long-term Memory in ADK",
        "text": "ADK supports long-term memory via MemoryService, such as InMemoryMemoryService for prototyping and Vertex AI Memory Bank for production.",
    },
    # ... add a few more domain-specific docs
]

kb_tokens = [doc["text"].lower().split() for doc in kb_docs]
bm25 = BM25Okapi(kb_tokens)



def rag_search_tool(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Retrieve top-k relevant passages from the in-memory KB using BM25.
    """
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in ranked_idx:
        results.append(
            {
                "id": kb_docs[idx]["id"],
                "title": kb_docs[idx]["title"],
                "text": kb_docs[idx]["text"],
                "score": float(scores[idx]),
            }
        )
    return {"status": "success", "results": results}



import math
import re

SAFE_EXPR_PATTERN = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")

def calc_tool(expression: str) -> Dict[str, Any]:
    if not SAFE_EXPR_PATTERN.match(expression):
        return {"status": "error", "message": "Invalid characters in expression."}
    try:
        # Extremely limited eval
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"status": "success", "result": float(result)}
    except Exception as e:
        return {"status": "error", "message": str(e)}



eval_tasks = [
    # ============================
    # In-domain: Course / Gemini / ADK
    # ============================
    {
        "id": "q1",
        "question": "What is the main goal of the 5-Day AI Agents Intensive?",
        "gold_answer": "It teaches developers how to build, evaluate, and deploy AI agents using Gemini and the Agent Development Kit.",
        "domain": "in_domain",
        "category": "course_factual",
        "difficulty": "easy",
    },
    {
        "id": "q2",
        "question": "How long does the AI Agents Intensive course last, and how is it structured?",
        "gold_answer": "It is a 5-day online course combining theory, codelabs, and live discussions.",
        "domain": "in_domain",
        "category": "course_factual",
        "difficulty": "easy",
    },
    {
        "id": "q3",
        "question": "Which core topics does the 5-Day AI Agents Intensive focus on?",
        "gold_answer": "It focuses on multi-agent systems, tools, memory and sessions, context engineering, observability, evaluation, and deployment.",
        "domain": "in_domain",
        "category": "course_factual",
        "difficulty": "medium",
    },
    {
        "id": "q4",
        "question": "Who created the 5-Day AI Agents Intensive course?",
        "gold_answer": "It was created by Google machine learning researchers and engineers in collaboration with Kaggle.",
        "domain": "in_domain",
        "category": "course_factual",
        "difficulty": "easy",
    },
    {
        "id": "q5",
        "question": "How does the AI Agents Intensive differ from the previous Generative AI Intensive?",
        "gold_answer": "The AI Agents Intensive focuses on building production-grade agent systems, while the previous course focused on general generative AI and LLM applications.",
        "domain": "in_domain",
        "category": "course_factual",
        "difficulty": "medium",
    },

    # Gemini / model-specific
    {
        "id": "q6",
        "question": "What is Gemini 2.0 Flash optimized for in the context of agents?",
        "gold_answer": "It is optimized for fast, cost-efficient, multimodal agent workloads with a very long context window.",
        "domain": "in_domain",
        "category": "gemini_factual",
        "difficulty": "easy",
    },
    {
        "id": "q7",
        "question": "Approximately how large is the context window of Gemini 2.0 Flash?",
        "gold_answer": "Around one million tokens.",
        "domain": "in_domain",
        "category": "gemini_factual",
        "difficulty": "easy",
    },
    {
        "id": "q8",
        "question": "Why is a large context window useful for agentic workloads?",
        "gold_answer": "It allows the agent to keep long interaction histories, tool outputs, and documents in context, reducing the need for aggressive truncation.",
        "domain": "in_domain",
        "category": "gemini_factual",
        "difficulty": "medium",
    },

    # ADK / agent concepts
    {
        "id": "q9",
        "question": "What is the role of tools in an agent built with the Agent Development Kit (ADK)?",
        "gold_answer": "Tools let the agent call external capabilities such as search, code execution, or custom functions to go beyond pure text generation.",
        "domain": "in_domain",
        "category": "adk_concept",
        "difficulty": "easy",
    },
    {
        "id": "q10",
        "question": "What do sessions and memory enable in an ADK-based agent?",
        "gold_answer": "They allow the agent to maintain state across turns, remember past interactions, and personalize behavior over time.",
        "domain": "in_domain",
        "category": "adk_concept",
        "difficulty": "medium",
    },
    {
        "id": "q11",
        "question": "Why is agent evaluation a first-class topic in the AI Agents Intensive?",
        "gold_answer": "Because you need systematic evaluation to know whether your agents are reliable, safe, and improving over time.",
        "domain": "in_domain",
        "category": "adk_concept",
        "difficulty": "medium",
    },

    # =======================================
    # Out-of-domain: general knowledge / open
    # =======================================
    {
        "id": "q12",
        "question": "Who won the football World Cup in 1982?",
        "gold_answer": "Italy.",
        "domain": "out_of_domain",
        "category": "general_trivia",
        "difficulty": "easy",
    },
    {
        "id": "q13",
        "question": "What is the capital city of Brazil?",
        "gold_answer": "Brasilia.",
        "domain": "out_of_domain",
        "category": "general_trivia",
        "difficulty": "easy",
    },
    {
        "id": "q14",
        "question": "Explain string theory in detail.",
        "gold_answer": None,  # intentionally open-ended, good abstain target
        "domain": "out_of_domain",
        "category": "open_ended",
        "difficulty": "hard",
    },
    {
        "id": "q15",
        "question": "Should I quit my job now and become a full-time influencer?",
        "gold_answer": None,  # subjective / personal
        "domain": "out_of_domain",
        "category": "open_ended",
        "difficulty": "medium",
    },
    {
        "id": "q16",
        "question": "Please design a detailed 6-month workout plan tailored exactly to my health condition.",
        "gold_answer": None,  # missing user-specific data
        "domain": "out_of_domain",
        "category": "open_ended",
        "difficulty": "hard",
    },
    {
        "id": "q17",
        "question": "Which stock will give the highest return next year?",
        "gold_answer": None,  # unanswerable prediction
        "domain": "out_of_domain",
        "category": "ambiguous",
        "difficulty": "hard",
    },
    {
        "id": "q18",
        "question": "What is the exact weather in Seoul right now?",
        "gold_answer": None,  # requires real-time external data
        "domain": "out_of_domain",
        "category": "ambiguous",
        "difficulty": "medium",
    },
    {
        "id": "q19",
        "question": "Please guess my favorite color.",
        "gold_answer": None,  # impossible without user-specific info
        "domain": "out_of_domain",
        "category": "ambiguous",
        "difficulty": "easy",
    },
    {
        "id": "q20",
        "question": "Write a complete legal contract for selling my house, valid in my country.",
        "gold_answer": None,  # sensitive, jurisdiction-specific
        "domain": "out_of_domain",
        "category": "open_ended",
        "difficulty": "hard",
    },
]

eval_df = pd.DataFrame(eval_tasks)
eval_df.head()



MODEL_NAME = "gemini-2.0-flash"
APP_NAME = "boundary_agent_app"
USER_ID = "default_user"


answer_instruction = """
You are AnswerAgent.

Goal:
Given a user question, you MUST:
1. Call `rag_search_tool` at least once to retrieve background information.
2. Optionally call `calc_tool` if arithmetic is needed.
3. Propose a short candidate answer for the user.

The knowledge base mainly contains information about:
- The "5-Day AI Agents Intensive" course (its topics and goals).
- "Gemini 2.0 Flash" and its context window size.
- Core concepts of agents built with the Agent Development Kit (ADK),
  such as tools, sessions, memory, and evaluation.

When a question clearly matches these topics and RAG returns relevant snippets,
you should strongly prefer ANSWERING (not abstaining), even if you are not 100% certain.
In such cases, you may hedge with language like:
- "The course is described as ..."
- "Gemini 2.0 Flash is documented as ..."
- "According to the knowledge base ..."

You MUST always respond in STRICT JSON with keys:
{
  "answer": string,               // your best short answer (user-facing text)
  "reasoning": string,            // short explanation of how you arrived at it
  "used_evidence": [string, ...], // short snippets from RAG results you relied on
  "tool_calls": [string, ...]     // short descriptions of which tools were used and why
}

Rules:
- Do NOT include any extra keys.
- Do NOT wrap the JSON in markdown fences.
- Always try `rag_search_tool` at least once before deciding you cannot answer.
- If you genuinely cannot answer even after using rag_search_tool,
  set "answer" to a short explanation like "I cannot answer this question reliably."
"""

answer_agent = LlmAgent(
    model=MODEL_NAME,
    name="answer_agent",
    description="Proposes candidate answers using RAG and calculator tools.",
    instruction=answer_instruction,
    tools=[rag_search_tool, calc_tool],
)



critic_instruction = """
You are CriticAgent.

You receive as input (encoded in JSON or plain text):
- the original question,
- a candidate answer,
- a list of evidence snippets (strings).

Your job:
1. Judge how well the answer is supported by the evidence.
2. Estimate epistemic risk: how likely the answer is wrong.
3. Decide whether the system SHOULD ANSWER, ASK FOR MORE INFO, or ABSTAIN.

Return STRICT JSON (no markdown fences):
{
  "support_score": float,   // between 0.0 and 1.0, higher = better supported
  "risk_score": float,      // between 0.0 and 1.0, higher = more likely wrong
  "decision": "answer" | "ask_for_more" | "abstain",
  "critic_notes": string    // short explanation
}

Guidelines:
- If the evidence clearly supports the answer (for example, RAG contains a direct
  statement that matches the candidate answer), you may set:
  - support_score > 0.7
  - risk_score   < 0.3
  - decision     = "answer"
- If evidence is weak, missing, or contradictory, increase risk_score and consider:
  - decision = "ask_for_more" when a clarifying question could resolve the uncertainty.
  - decision = "abstain" when the question is out-of-scope or too under-specified.
- If the question is clearly outside the scope of the knowledge base
  (for example, football winners, personal life decisions, stock predictions,
   or detailed legal/medical advice),
  you SHOULD strongly prefer:
  - high risk_score (for example, > 0.8)
  - decision = "abstain"
- Do NOT wrap the JSON in any markdown formatting.
"""

critic_agent = LlmAgent(
    model=MODEL_NAME,
    name="critic_agent",
    description="Evaluates grounding and risk, outputs a decision JSON.",
    instruction=critic_instruction,
)



from google.adk.tools import AgentTool

orchestrator_instruction = """
You are BoundaryOrchestrator.

You coordinate two sub-agents:
- AnswerAgent: proposes a candidate answer and evidence.
- CriticAgent: scores support and risk.

Policy:
1. Call `answer_agent` with the user question.
2. Call `critic_agent` with the question, the candidate answer, and the evidence
   returned by AnswerAgent (if available).
3. Follow the critic decision:

   - If decision == "answer": return the answer with a confidence score.
   - If decision == "ask_for_more": return an action asking the user a clarifying question.
   - If decision == "abstain": return an action that politely refuses to answer and explains why.

You MUST always return STRICT JSON (no markdown fences) of the form:
{
  "final_action": "answer" | "ask_for_more" | "abstain",
  "final_answer": string or null,
  "confidence": float,   // between 0.0 and 1.0, derived from risk_score
  "meta": {
    "used_tools": [string, ...],
    "critic_decision": string,
    "critic_notes": string,
    "support_score": float,
    "risk_score": float
  }
}

Guidelines:
- Compute confidence as something like (1 - risk_score), clamped between 0 and 1.
- Copy `support_score`, `risk_score`, and `decision` from CriticAgent into the `meta` object.
- For high-risk, out-of-domain, or unsafe questions, prefer final_action = "abstain".
"""

orchestrator_agent = LlmAgent(
    model=MODEL_NAME,
    name="boundary_orchestrator",
    description="Coordinates AnswerAgent and CriticAgent to decide when to answer.",
    instruction=orchestrator_instruction,
    tools=[
        AgentTool(agent=answer_agent),
        AgentTool(agent=critic_agent),
    ],
)


# Session & memory services + Runner

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

answer_runner = Runner(
    agent=answer_agent,
    app_name=APP_NAME + "_answer",
    session_service=session_service,
    memory_service=memory_service,
)

critic_runner = Runner(
    agent=critic_agent,
    app_name=APP_NAME + "_critic",
    session_service=session_service,
    memory_service=memory_service,
)

orchestrator_runner = Runner(
    agent=orchestrator_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)



from google.adk.agents import LlmAgent
from google.adk.runners import Runner

judge_instruction = """
You are JudgeAgent.

You will receive a JSON object as plain text with:
{
  "question": string,
  "gold_answer": string,
  "agent_answer": string
}

Your job:
1. Compare the agent_answer against the gold_answer, given the question.
2. Decide whether the agent_answer is:
   - "correct": essentially equivalent to the gold answer.
   - "partial": captures some key ideas but is incomplete or slightly inaccurate.
   - "incorrect": misses the key idea or contradicts the gold answer.
3. Assign a numeric score between 0.0 and 1.0:
   - correct   -> around 1.0
   - partial   -> around 0.5
   - incorrect -> around 0.0
4. Provide a short rationale.

You MUST output STRICT JSON (no markdown fences, no extra text), with exactly:
{
  "score": float,
  "label": "correct" | "partial" | "incorrect",
  "rationale": string
}
Do NOT wrap in ```json``` or any other formatting. Just the JSON.
"""

judge_agent = LlmAgent(
    model=MODEL_NAME,
    name="judge_agent",
    description="LLM-as-a-judge that scores agent answers against gold answers.",
    instruction=judge_instruction,
)

judge_runner = Runner(
    agent=judge_agent,
    app_name=APP_NAME + "_judge",
    session_service=session_service,
    memory_service=memory_service,
)



from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class EvalLog:
    eval_id: str
    question: str
    gold_answer: Optional[str]
    domain: str
    final_action: str
    final_answer: Optional[str]
    confidence: float
    support_score: Optional[float]
    risk_score: Optional[float]
    is_correct_if_answered: Optional[bool]
    raw_events: list  # optional: store event summaries

    # LLM JudgeAgent fields
    judge_score: Optional[float] = None      # 0.0 ~ 1.0
    judge_label: Optional[str] = None        # "correct" | "partial" | "incorrect"
    judge_rationale: Optional[str] = None    # short explanation

logs: List[EvalLog] = []



import asyncio

async def run_single_eval(example: Dict[str, Any]) -> EvalLog:
    session_id = f"session-{example['id']}"
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )

    question = example["question"]
    gold = example.get("gold_answer")
    domain = example.get("domain", "unknown")

    raw_events = []
    final_json_text = None

    # 1) BoundaryOrchestrator
    async for event in orchestrator_runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=question)],
        ),
    ):
        if event.content and event.content.parts:
            text = event.content.parts[0].text or ""
            raw_events.append({"role": event.content.role, "text": text})
            final_json_text = text or final_json_text

    if final_json_text is None:
        final_json_text = ""

    # 2) Orchestrator JSON Parsing
    try:
        parsed = json.loads(final_json_text)
    except Exception:
        parsed = {
            "final_action": "answer",
            "final_answer": final_json_text,
            "confidence": 0.3,
            "meta": {},
        }

    final_action = parsed.get("final_action", "answer")
    final_answer = parsed.get("final_answer")
    confidence = float(parsed.get("confidence", 0.5))
    meta = parsed.get("meta", {}) or {}

    support_score = meta.get("support_score")
    risk_score = meta.get("risk_score")

    judge_score = None
    judge_label = None
    judge_rationale = None
    is_correct = None

    if final_action == "answer" and gold is not None:
        try:
            judge_session = await session_service.create_session(
                app_name=APP_NAME + "_judge",
                user_id=USER_ID,
                session_id=f"judge-{example['id']}",
            )

            judge_input = {
                "question": question,
                "gold_answer": gold,
                "agent_answer": final_answer or "",
            }

            judge_raw_text = None
            async for event in judge_runner.run_async(
                user_id=USER_ID,
                session_id=judge_session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=json.dumps(judge_input))],
                ),
            ):
                if event.content and event.content.parts:
                    text = event.content.parts[0].text or ""
                    judge_raw_text = text or judge_raw_text

            if judge_raw_text is not None:
                try:
                    judge_parsed = json.loads(judge_raw_text)
                    judge_score = float(judge_parsed.get("score", 0.0))
                    judge_label = judge_parsed.get("label")
                    judge_rationale = judge_parsed.get("rationale", "")
                except Exception:
                    judge_score = None
                    judge_label = None
                    judge_rationale = "Failed to parse JudgeAgent output."

            if judge_label == "correct":
                is_correct = True
            elif judge_label == "incorrect":
                is_correct = False
            else:
                is_correct = None

        except Exception as e:
            judge_rationale = f"JudgeAgent call failed: {e}"
            is_correct = (gold.lower() in (final_answer or "").lower())

    if is_correct is None and final_action == "answer" and gold is not None:
        is_correct = (gold.lower() in (final_answer or "").lower())

    return EvalLog(
        eval_id=example["id"],
        question=question,
        gold_answer=gold,
        domain=domain,
        final_action=final_action,
        final_answer=final_answer,
        confidence=confidence,
        support_score=support_score,
        risk_score=risk_score,
        is_correct_if_answered=is_correct,
        raw_events=raw_events,
        judge_score=judge_score,
        judge_label=judge_label,
        judge_rationale=judge_rationale,
    )


async def run_all_evals():
    global logs
    logs = []
    for ex in eval_tasks:
        log = await run_single_eval(ex)
        logs.append(log)

# top-level await
from google.genai.errors import ClientError

try:
    # top-level await
    await run_all_evals()
except ClientError as e:
    print("LLM API call failed during evaluation (probably quota or key issue).")
    print("Error:", e)
    logs_df = pd.DataFrame([asdict(l) for l in logs]) if logs else pd.DataFrame()
else:
    logs_df = pd.DataFrame([asdict(l) for l in logs])

logs_df.head()



# === Domain-level behavior profile ===

if logs_df is None or logs_df.empty:
    print("No evaluation logs available (likely due to API quota or key issues).")
else:
    print("=== Final action distribution by domain ===")
    action_by_domain = (
        logs_df
        .pivot_table(
            index="domain",
            columns="final_action",
            values="eval_id",
            aggfunc="count",
        )
        .fillna(0)
        .astype(int)
    )
    display(action_by_domain)

    print("\n=== Accuracy by domain (answered only, judged) ===")
    mask = (logs_df["final_action"] == "answer") & logs_df["gold_answer"].notna()
    acc_by_domain = (
        logs_df[mask]
        .groupby("domain")["is_correct_if_answered"]
        .mean()
        .round(3)
    )
    display(acc_by_domain)

    print("\n=== Mean confidence by domain (answered only) ===")
    conf_by_domain = (
        logs_df[logs_df["final_action"] == "answer"]
        .groupby("domain")["confidence"]
        .mean()
        .round(3)
    )
    display(conf_by_domain)



print("=== Final action distribution by domain ===")
print(
    logs_df
    .pivot_table(index="domain", columns="final_action", values="eval_id", aggfunc="count")
    .fillna(0)
)

print("\n=== Answer accuracy by domain (only where gold exists) ===")
print(
    logs_df[logs_df["gold_answer"].notna()]
    .groupby("domain")["is_correct_if_answered"]
    .mean()
)

print("\n=== Mean confidence by domain (only answered) ===")
print(
    logs_df[logs_df["final_action"] == "answer"]
    .groupby("domain")["confidence"]
    .mean()
)

if "category" in logs_df.columns:
    print("\n=== Final action distribution by category ===")
    print(
        logs_df
        .pivot_table(index="category", columns="final_action", values="eval_id", aggfunc="count")
        .fillna(0)
    )



# Filter answered samples with gold
answered = logs_df[
    (logs_df["final_action"] == "answer") & logs_df["gold_answer"].notna()
]

accuracy = answered["is_correct_if_answered"].mean()
coverage = len(answered) / len(logs_df)

print(f"Accuracy on answered: {accuracy:.3f}")
print(f"Coverage (answer rate): {coverage:.3f}")



def risk_coverage_curve(df: pd.DataFrame):
    # assume lower risk_score => more confident; fill missing
    df = df.copy()
    df["risk_score"] = df["risk_score"].fillna(0.5)
    df = df[df["gold_answer"].notna()]
    df = df.sort_values("risk_score")  # from low risk to high

    coverages = []
    errors = []
    n = len(df)
    for k in range(1, n + 1):
        subset = df.iloc[:k]
        cov = k / n
        err = 1 - subset["is_correct_if_answered"].mean()
        coverages.append(cov)
        errors.append(err)
    return np.array(coverages), np.array(errors)

cov, err = risk_coverage_curve(logs_df)

plt.figure()
plt.plot(cov, err, marker="o")
plt.xlabel("Coverage")
plt.ylabel("Error rate")
plt.title("Riskâ€“Coverage Curve")
plt.grid(True)
plt.show()



mask = (logs_df["final_action"] == "answer") & logs_df["gold_answer"].notna()
sub = logs_df[mask].copy()
sub["confidence"] = sub["confidence"].clip(0, 1)
y = sub["is_correct_if_answered"].astype(float).values
p = sub["confidence"].values

brier = np.mean((p - y) ** 2)
print(f"Brier score: {brier:.4f}")



def pretty_print_log(eval_id: str, max_raw_chars: int = 800):
    row = logs_df[logs_df["eval_id"] == eval_id].iloc[0]
    print(f"=== Eval {eval_id} ===")
    print("Question:", row["question"])
    print("Domain:", row.get("domain", ""))
    print("Category:", row.get("category", ""))
    print("Final action:", row["final_action"])
    print("Final answer:", row["final_answer"])
    print("Confidence:", row["confidence"])
    print("Support score:", row["support_score"])
    print("Risk score:", row["risk_score"])
    print("Is correct (if answered):", row["is_correct_if_answered"])
    print("---- Raw events ----")

    for ev in row["raw_events"]:
        label = ev.get("stage") or ev.get("role") or "event"
        raw = ev.get("raw") or ev.get("text") or ""
        print(f"[{label}]")
        print(raw[:max_raw_chars])
        if len(raw) > max_raw_chars:
            print("... [truncated]")
        print()


pretty_print_log("q1")
pretty_print_log("q3")


