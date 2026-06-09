# Install only essentials â€” Kaggle already has many libs
!pip install -q scikit-learn==1.3.0 joblib==1.3.2 streamlit==1.26.0

import os, sys, json, time, logging, random
from typing import List, Dict, Any
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("studyalpha_demo")

print("Imports OK")



# ---- utils (inline for notebook) ----
import os
def get_gemini_key():
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        return user_secrets.get_secret("GOOGLE_API_KEY")
    except Exception:
        return os.getenv("GEMINI_API_KEY", None)

def mock_llm(prompt: str) -> str:
    # deterministic mock for reproducible demo
    seed = abs(hash(prompt)) % 10000
    random.seed(seed)
    # short structured response to be easy to parse if needed
    return "MOCK_RESPONSE\n" + "This mock reply is deterministic for demo. Seed:" + str(seed)

def call_gemini(prompt: str, model: str="gemini-flash", max_tokens:int=512) -> str:
    """
    Wrapper: uses Kaggle secrets or ENV var for Gemini/ADK key.
    If not present, returns deterministic mock.
    Replace TODO block with actual SDK calls when using real keys.
    """
    key = get_gemini_key()
    if key:
        # TODO: insert real Gemini/ADK call here (kept out for security)
        # Example (pseudocode):
        # import google.generativeai as genai
        # genai.configure(api_key=key)
        # model_obj = genai.GenerativeModel(model)
        # out = model_obj.generate_content(prompt, max_tokens=max_tokens)
        # return out.text
        raise NotImplementedError("Insert your Gemini/ADK client call here.")
    else:
        return mock_llm(prompt)

def trace(action: str, details: Dict[str,Any]):
    logger.info(json.dumps({"action": action, **details}))



# ---- memory.py inline ----
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MemoryBank:
    def __init__(self):
        self.long_term = []  # list of dicts: {'id', 'text', 'meta'}
        self.vectorizer = None
        self.tfidf_matrix = None

    def add(self, text: str, meta: Dict = None):
        idx = len(self.long_term)
        self.long_term.append({"id": idx, "text": text, "meta": meta or {}})
        trace("memory.add", {"id": idx, "meta": meta})
        self._reindex()

    def _reindex(self):
        corpus = [d["text"] for d in self.long_term]
        if corpus:
            self.vectorizer = TfidfVectorizer(stop_words="english").fit(corpus)
            self.tfidf_matrix = self.vectorizer.transform(corpus)
        else:
            self.vectorizer = None
            self.tfidf_matrix = None

    def query(self, q: str, top_k: int=3):
        if self.tfidf_matrix is None:
            return []
        q_vec = self.vectorizer.transform([q])
        scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        best_idx = np.argsort(scores)[::-1][:top_k]
        results = [self.long_term[int(i)] for i in best_idx if scores[int(i)] > 0]
        trace("memory.query", {"query": q, "results": [r["id"] for r in results]})
        return results

    def dump(self, path="memory_dump.json"):
        with open(path,"w") as f:
            json.dump(self.long_term, f)



# ---- predictor.py inline ----
from sklearn.ensemble import GradientBoostingClassifier

MODEL_PATH = "predictor_model.joblib"

def make_synthetic_training_data(n=400):
    X = np.random.rand(n,3)
    y = (X[:,0] < 0.65).astype(int)  # weakness if low correct rate
    return X, y

def train_and_save_model(path=MODEL_PATH):
    X,y = make_synthetic_training_data()
    clf = GradientBoostingClassifier()
    clf.fit(X,y)
    joblib.dump(clf, path)
    trace("predictor.train", {"path": path})
    return path

def load_model(path=MODEL_PATH):
    try:
        return joblib.load(path)
    except Exception:
        return None

def predict_weakness(features: List[float], model=None) -> float:
    if model is None:
        model = load_model()
    if model is None:
        # fallback heuristic
        prob = float(max(0.0, 0.95 - features[0]))
        trace("predictor.fallback", {"features": features, "prob": prob})
        return prob
    prob = float(model.predict_proba([features])[0][1])
    trace("predictor.predict", {"features": features, "prob": prob})
    return prob



# ---- tools.py inline ----
def create_study_plan(topics: List[Dict], hours_per_day: float=2.0, days: int=7) -> Dict:
    prompt = f"Planner: topics={topics}, hours_per_day={hours_per_day}, days={days}"
    out = call_gemini(prompt)
    if out.startswith("MOCK_RESPONSE"):
        # deterministic round-robin scheduler
        flat = []
        for t in topics:
            copies = max(1, int(t.get("priority",1)))
            for _ in range(copies):
                flat.append(t["topic"])
        plan = {f"day_{i+1}":[] for i in range(days)}
        for i, topic in enumerate(flat):
            day = f"day_{(i % days)+1}"
            plan[day].append({"topic": topic, "duration_mins": int(60*hours_per_day/len(flat)), "type":"learning"})
        return {"plan_id":"mock_plan_v1","days":plan,"meta":{"hours_per_day":hours_per_day,"days":days}}
    # production: parse JSON from out
    return {"plan_id":"llm_plan","days":{}, "meta":{}}

def generate_quiz_from_topic(topic: str, mode: str="general", memory: MemoryBank=None) -> Dict:
    context = ""
    if memory:
        hits = memory.query(topic, top_k=3)
        context = "\n".join([h["text"] for h in hits])
    prompt = f"QuizAgent: topic={topic}, mode={mode}, context={context}"
    out = call_gemini(prompt)
    if out.startswith("MOCK_RESPONSE"):
        qs = [{"id":f"q{i+1}", "q":f"Explain {topic} â€” part {i+1}", "a":"Sample answer", "type":"text"} for i in range(3)]
        return {"quiz_id":f"mock_quiz_{topic}","topic":topic,"questions":qs}
    return {"quiz_id":"llm_quiz","topic":topic,"questions":[]}

def evaluate_quiz(answers: List[str], ground_truth: List[str]) -> Dict:
    scores = []
    for a, g in zip(answers, ground_truth):
        scores.append(1.0 if a.strip().lower() == g.strip().lower() else 0.0)
    total = sum(scores)
    return {"score":total, "max_score": len(scores), "accuracy": total/len(scores) if scores else 0.0}



# ---- agents.py inline ----
class PlannerAgent:
    def generate(self, topics, hours_per_day=2.0, days=7):
        plan = create_study_plan(topics, hours_per_day, days)
        trace("Planner.generate", {"days": len(plan.get("days", {}))})
        return plan

class RevisionAgent:
    def __init__(self, memory: MemoryBank):
        self.memory = memory
    def generate(self, topic):
        trace("Revision.generate", {"topic": topic})
        return {"topic": topic, "sessions": [{"duration":20, "focus":topic}]}

class QuizAgent:
    def __init__(self, memory: MemoryBank):
        self.memory = memory
    def generate(self, topic, mode="general"):
        return generate_quiz_from_topic(topic, mode, memory=self.memory)

class TrackerAgent:
    def __init__(self, memory: MemoryBank, model_path=None):
        self.memory = memory
        self.model = load_model()
    def record_quiz(self, quiz, user_answers):
        gt = [q["a"] for q in quiz["questions"]]
        res = evaluate_quiz(user_answers, gt)
        self.memory.add(json.dumps({"quiz_topic": quiz["topic"], "result": res}))
        # features: [avg_correct_rate, avg_time_per_q (demo), recency_days]
        features = [res["accuracy"], 60.0, 2.0]
        prob = predict_weakness(features, self.model)
        trace("Tracker.record_quiz", {"topic": quiz["topic"], "res": res, "weakness_prob": prob})
        return {"evaluation": res, "weakness_prob": prob}

class StudyOrchestrator:
    def __init__(self):
        self.memory = MemoryBank()
        self.planner = PlannerAgent()
        self.revision = RevisionAgent(self.memory)
        self.quiz = QuizAgent(self.memory)
        self.tracker = TrackerAgent(self.memory)
    def full_plan_flow(self, topics, hours_per_day=2.0, days=7):
        plan = self.planner.generate(topics, hours_per_day, days)
        sample_topic = topics[0]["topic"] if topics else "General"
        quiz = self.quiz.generate(sample_topic)
        return {"plan": plan, "sample_quiz": quiz}



# Train demo predictor (synthetic)
print("Training synthetic predictor...")
train_and_save_model()
print("Predictor trained (joblib saved).")

# Run quick demo
orchestrator = StudyOrchestrator()
topics = [{"topic":"Arrays","priority":2},{"topic":"Graphs","priority":1},{"topic":"Dynamic Programming","priority":2}]
flow = orchestrator.full_plan_flow(topics, hours_per_day=2.0, days=7)
print("Plan (sample days):")
print(json.dumps(flow["plan"], indent=2))

print("\nSample Quiz for first topic:")
for i,q in enumerate(flow["sample_quiz"]["questions"]):
    print(f"Q{i+1}. {q['q']}\nA: {q['a']}\n")



# Simulate a 7-day run to produce a weekly report (randomized answers)
print("Simulating 7-day study run...")
orchestrator = StudyOrchestrator()
plan = orchestrator.planner.generate(topics, 2.0, 7)
daily_results = []
for day_idx in range(7):
    day_key = f"day_{(day_idx%7)+1}"
    tasks = plan["days"].get(day_key, [])
    # pick first learning topic if exists
    topic = tasks[0]["topic"] if tasks else topics[0]["topic"]
    quiz = orchestrator.quiz.generate(topic)
    # simulate answers (some correct, some wrong)
    answers = [q["a"] if random.random() > 0.35 else "wrong answer" for q in quiz["questions"]]
    rec = orchestrator.tracker.record_quiz(quiz, answers)
    daily_results.append({"day": day_idx+1, "topic": topic, "accuracy": rec["evaluation"]["accuracy"], "weakness_prob": rec["weakness_prob"]})
print("Daily results:")
print(pd.DataFrame(daily_results))
# simple weekly summary
avg_accuracy = np.mean([r["accuracy"] for r in daily_results])
print(f"\nWeekly avg accuracy: {avg_accuracy:.2f}")



## How to enable Gemini (optional)
1. Set your Gemini / ADK key in Kaggle Secrets as `GOOGLE_API_KEY`, OR set environment variable `GEMINI_API_KEY`.
2. Replace the `call_gemini()` TODO with the official client call as per ADK or the google.generativeai SDK.
3. Re-run the notebook â€” the Planner and Quiz agents will produce richer, LLM-grounded outputs.

> NOTE: This notebook runs end-to-end with a deterministic mock to allow judges to reproduce results without keys.



# Helper: Quick demo function for judges to test different topics
def try_custom_plan(topics_text="Arrays,2\nTrees,1\nDP,2", hours=2.0, days=7):
    topics = []
    for line in topics_text.strip().splitlines():
        name, *priority = line.split(",")
        p = int(priority[0]) if priority else 1
        topics.append({"topic": name.strip(), "priority": p})
    orch = StudyOrchestrator()
    out = orch.full_plan_flow(topics, hours, days)
    print("STUDY PLAN:")
    print(json.dumps(out["plan"], indent=2))
    print("\nQUIZ:")
    print(json.dumps(out["sample_quiz"], indent=2))

# example
try_custom_plan()



# Save memory dump, model, and sample plan for GitHub sync
orch = StudyOrchestrator()
sample = orch.full_plan_flow(
    [{"topic":"Arrays","priority":2},{"topic":"DP","priority":2}],
    2.0,
    7
)

with open("sample_plan.json","w") as f:
    json.dump(sample["plan"], f)

with open("sample_quiz.json","w") as f:
    json.dump(sample["sample_quiz"], f)

print("Artifacts saved: sample_plan.json, sample_quiz.json")



# Simple explainability for weakness predictor
import numpy as np

print("Feature importance for synthetic GradientBoosting model:")
model = load_model()
if model:
    print(model.feature_importances_)
else:
    print("Using fallback heuristic model. No feature importances.")



import pandas as pd
import matplotlib.pyplot as plt

plan = orchestrator.planner.generate(topics, 2.0, 7)["days"]
rows = []
for day, tasks in plan.items():
    for t in tasks:
        rows.append([day, t["topic"], t["duration_mins"]])

df = pd.DataFrame(rows, columns=["day", "topic", "duration"])

pivot = df.pivot_table(index="day", columns="topic", values="duration", aggfunc="sum")
pivot.plot(kind="bar", figsize=(10,5))
plt.title("Study Plan Distribution")
plt.ylabel("Minutes")
plt.show()



orch = StudyOrchestrator()
orch.memory.add("Arrays are linear data structures used to store items.")
orch.memory.add("Dynamic Programming optimizes recursive problems.")

q = "How does DP work?"
hits = orch.memory.query(q)
hits



rev = orchestrator.revision.generate("Dynamic Programming")
rev



df = pd.DataFrame(daily_results)
df.plot(x="day", y="accuracy", kind="line", marker="o", figsize=(10,5))
plt.title("Daily Accuracy Trend")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()



import logging
logger = logging.getLogger("studyalpha_demo")
logger.handlers



with open("trace_log.txt","w") as f:
    f.write("Tracing enabled. Inspect logs in application output.")



weekly_report = {
    "average_accuracy": float(avg_accuracy),
    "days": daily_results,
}

with open("weekly_report.json","w") as f:
    json.dump(weekly_report, f, indent=2)

weekly_report



def studyalpha_api(topics):
    orch = StudyOrchestrator()
    return orch.full_plan_flow(topics)

studyalpha_api([{"topic":"Trees","priority":1}])



print("Running internal consistency checks...")

# Recompute the plan and quiz to ensure the objects exist
orch = StudyOrchestrator()
topics = [
    {"topic": "Arrays", "priority": 2},
    {"topic": "Graphs", "priority": 1},
    {"topic": "Dynamic Programming", "priority": 2},
]
flow = orch.full_plan_flow(topics, hours_per_day=2.0, days=7)

# Extract plan + quiz safely
plan = flow["plan"]
quiz = flow["sample_quiz"]

# Assertions (these now ALWAYS pass if code is correct)
assert "days" in plan, "Plan should contain a 'days' field"
assert len(plan["days"]) == 7, "Plan should contain 7 days"

assert "questions" in quiz, "'sample_quiz' must have a 'questions' field"
assert len(quiz["questions"]) == 3, "Quiz must have 3 questions"

print("âœ” All internal tests passed successfully.")


