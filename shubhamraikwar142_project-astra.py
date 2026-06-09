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


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")

print("Loaded key:", GEMINI_API_KEY[:8] + "...")



# ==========================================================
# Cell 1 â€” SETUP: Imports + Gemini Init + helper
# ==========================================================
!pip install -q google-generativeai faiss-cpu

import os, time, json, uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List

import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# Load GEMINI API key from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
except Exception as e:
    raise RuntimeError("Could not load GEMINI_API_KEY from Kaggle Secrets. Error: " + str(e))

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY empty. Add it to Kaggle Add-on Secrets.")

genai.configure(api_key=GEMINI_API_KEY)
print("ğŸ”� GEMINI key loaded (from Kaggle Secrets).")

# Use gemini-pro (v1beta-friendly model name in Kaggle)
try:
    gemini_model = genai.GenerativeModel("gemini-pro")
    test = gemini_model.generate_content("Hello ASTRA! Please reply OK.")
    print("âœ… Gemini model ready (gemini-pro). Test output:", str(test.text)[:200])
except Exception as e:
    print("â�Œ Gemini init error:", e)
    gemini_model = None

# Safe caller wrapper
def call_gemini(prompt: str, max_retries: int = 3) -> str:
    if gemini_model is None:
        return "(Offline Mode) MOCK RESPONSE:\n" + prompt[:300]
    last_err = None
    for _ in range(max_retries):
        try:
            out = gemini_model.generate_content(prompt)
            return out.text
        except Exception as e:
            last_err = e
            time.sleep(0.8)
    return f"(Fallback) Gemini did not respond. Last error: {last_err}"



# ==========================================================
# Cell 2 â€” Memory + Embeddings + Vector store
# ==========================================================
# Embedding helper using textembedding-gecko (Kaggle friendly)
def embed_text(text: str) -> np.ndarray:
    try:
        resp = genai.embed_content(model="textembedding-gecko", content=text)
        emb = np.array(resp["embedding"], dtype=np.float32)
        return emb
    except Exception:
        # deterministic fallback vector
        v = np.zeros(256, dtype=np.float32)
        for i, ch in enumerate(text[:256]):
            v[i] = ord(ch) / 255.0
        return v

# JSON memory
class JSONMemoryBank:
    def __init__(self, filename="json_memory_bank.json"):
        self.filename = filename
        try:
            with open(self.filename, "r") as f:
                self.memory = json.load(f)
        except Exception:
            self.memory = {}

    def save(self):
        with open(self.filename, "w") as f:
            json.dump(self.memory, f, indent=2)

    def get_user(self, uid):
        return self.memory.get(uid, {
            "profile": {},
            "progress": {},
            "notes": [],
            "quizzes": {},
            "cognitive_profile": {}
        })

    def update_user(self, uid, key, value):
        user = self.get_user(uid)
        user[key] = value
        self.memory[uid] = user
        self.save()

    def append_note(self, uid, note):
        user = self.get_user(uid)
        user["notes"].append({"time": time.time(), "note": note})
        self.memory[uid] = user
        self.save()

    def record_quiz(self, uid, topic, score):
        user = self.get_user(uid)
        user["quizzes"][topic] = {"score": float(score), "time": time.time()}
        self.memory[uid] = user
        self.save()

# Vector memory (FAISS if available)
try:
    import faiss
    _has_faiss = True
except Exception:
    faiss = None
    _has_faiss = False

class VectorMemoryBank:
    def __init__(self, dim=None):
        self.dim = dim
        self.texts = []
        self.index = None

    def _ensure_index(self, vec):
        d = len(vec)
        if self.index is None or self.dim != d:
            self.dim = d
            if _has_faiss:
                self.index = faiss.IndexFlatL2(d)
            else:
                self.index = None

    def add(self, text):
        v = embed_text(text).astype("float32")
        self._ensure_index(v)
        idx = len(self.texts)
        self.texts.append(text)
        if self.index is not None:
            self.index.add(v.reshape(1, -1))

    def search(self, query, top_k=5):
        q = embed_text(query).astype("float32")
        self._ensure_index(q)
        if self.index is not None:
            D, I = self.index.search(q.reshape(1, -1), top_k)
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx >= 0 and idx < len(self.texts):
                    results.append({"text": self.texts[idx], "distance": float(dist)})
            return results
        # fallback python
        scores = []
        for t in self.texts:
            diff = q - embed_text(t)
            dist = float(np.dot(diff, diff))
            scores.append({"text": t, "distance": dist})
        scores.sort(key=lambda x: x["distance"])
        return scores[:top_k]

class HybridMemory:
    def __init__(self, json_mem, vec_mem):
        self.json = json_mem
        self.vec = vec_mem

    def get_user(self, uid):
        return self.json.get_user(uid)

    def save_structured(self, uid, key, val):
        self.json.update_user(uid, key, val)

    def save_note(self, uid, txt):
        self.json.append_note(uid, txt)
        self.vec.add(f"{uid}:{txt}")

    def semantic_search(self, uid, query, top_k=5):
        return self.vec.search(query, top_k)

# Initialize memory
json_mem = JSONMemoryBank()
vec_mem = VectorMemoryBank()
memory = HybridMemory(json_mem, vec_mem)
print("Hybrid memory initialized.")



# ==========================================================
# Cell 2 â€” Memory + Embeddings + Vector store
# ==========================================================
# Embedding helper using textembedding-gecko (Kaggle friendly)
def embed_text(text: str) -> np.ndarray:
    try:
        resp = genai.embed_content(model="textembedding-gecko", content=text)
        emb = np.array(resp["embedding"], dtype=np.float32)
        return emb
    except Exception:
        # deterministic fallback vector
        v = np.zeros(256, dtype=np.float32)
        for i, ch in enumerate(text[:256]):
            v[i] = ord(ch) / 255.0
        return v

# JSON memory
class JSONMemoryBank:
    def __init__(self, filename="json_memory_bank.json"):
        self.filename = filename
        try:
            with open(self.filename, "r") as f:
                self.memory = json.load(f)
        except Exception:
            self.memory = {}

    def save(self):
        with open(self.filename, "w") as f:
            json.dump(self.memory, f, indent=2)

    def get_user(self, uid):
        return self.memory.get(uid, {
            "profile": {},
            "progress": {},
            "notes": [],
            "quizzes": {},
            "cognitive_profile": {}
        })

    def update_user(self, uid, key, value):
        user = self.get_user(uid)
        user[key] = value
        self.memory[uid] = user
        self.save()

    def append_note(self, uid, note):
        user = self.get_user(uid)
        user["notes"].append({"time": time.time(), "note": note})
        self.memory[uid] = user
        self.save()

    def record_quiz(self, uid, topic, score):
        user = self.get_user(uid)
        user["quizzes"][topic] = {"score": float(score), "time": time.time()}
        self.memory[uid] = user
        self.save()

# Vector memory (FAISS if available)
try:
    import faiss
    _has_faiss = True
except Exception:
    faiss = None
    _has_faiss = False

class VectorMemoryBank:
    def __init__(self, dim=None):
        self.dim = dim
        self.texts = []
        self.index = None

    def _ensure_index(self, vec):
        d = len(vec)
        if self.index is None or self.dim != d:
            self.dim = d
            if _has_faiss:
                self.index = faiss.IndexFlatL2(d)
            else:
                self.index = None

    def add(self, text):
        v = embed_text(text).astype("float32")
        self._ensure_index(v)
        idx = len(self.texts)
        self.texts.append(text)
        if self.index is not None:
            self.index.add(v.reshape(1, -1))

    def search(self, query, top_k=5):
        q = embed_text(query).astype("float32")
        self._ensure_index(q)
        if self.index is not None:
            D, I = self.index.search(q.reshape(1, -1), top_k)
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx >= 0 and idx < len(self.texts):
                    results.append({"text": self.texts[idx], "distance": float(dist)})
            return results
        # fallback python
        scores = []
        for t in self.texts:
            diff = q - embed_text(t)
            dist = float(np.dot(diff, diff))
            scores.append({"text": t, "distance": dist})
        scores.sort(key=lambda x: x["distance"])
        return scores[:top_k]

class HybridMemory:
    def __init__(self, json_mem, vec_mem):
        self.json = json_mem
        self.vec = vec_mem

    def get_user(self, uid):
        return self.json.get_user(uid)

    def save_structured(self, uid, key, val):
        self.json.update_user(uid, key, val)

    def save_note(self, uid, txt):
        self.json.append_note(uid, txt)
        self.vec.add(f"{uid}:{txt}")

    def semantic_search(self, uid, query, top_k=5):
        return self.vec.search(query, top_k)

# Initialize memory
json_mem = JSONMemoryBank()
vec_mem = VectorMemoryBank()
memory = HybridMemory(json_mem, vec_mem)
print("Hybrid memory initialized.")



# ==========================================================
# Cell 3 â€” Agents (Planner / Tutor / Retriever / Evaluator)
# ==========================================================
@dataclass
class AgentResponse:
    text: str
    metadata: dict = field(default_factory=dict)

class BaseAgent:
    def __init__(self, name, memory):
        self.name = name
        self.memory = memory

class PlannerAgent(BaseAgent):
    def respond(self, uid, payload, session):
        system = "You are an adaptive DS/ML curriculum planner.\n"
        prompt = system + "\n" + payload
        out = call_gemini(prompt)
        user = self.memory.get_user(uid)
        prog = user.get("progress", {})
        prog["plan"] = out
        self.memory.json.update_user(uid, "progress", prog)
        self.memory.save_note(uid, "PLAN_CREATED")
        return AgentResponse(text=out)

class TutorAgent(BaseAgent):
    def respond(self, uid, question, session):
        profile = self.memory.get_user(uid).get("profile", {})
        tone = profile.get("preferred_tone", "adaptive")
        prompt = f"You are a DS/ML tutor. Reply in tone={tone}.\nAnswer clearly with examples.\n\nQuestion: {question}"
        out = call_gemini(prompt)
        self.memory.save_note(uid, f"TUTOR_QA:{question[:60]}")
        # attempt to update cognitive signal; ignore if not available
        try:
            cognitive_engine.update_from_tutor(uid, question[:80], signal="curious")
        except Exception:
            pass
        return AgentResponse(text=out)

class RetrieverAgent(BaseAgent):
    def respond(self, uid, topic, session):
        prompt = f"Provide 5 curated DS resources for: {topic}"
        out = call_gemini(prompt)
        self.memory.save_note(uid, f"RESOURCES:{topic}")
        return AgentResponse(text=out)

class EvaluatorAgent(BaseAgent):
    def generate(self, uid, topic):
        prompt = f"Create 3 conceptual MCQs about {topic}."
        out = call_gemini(prompt)
        self.memory.save_note(uid, f"QUIZ_GEN:{topic}")
        return out

    def grade(self, uid, topic, answer):
        prompt = (
            "Grade this answer 0â€“1.\n"
            "Return JSON: {\"score\": float, \"feedback\": \"...\"}\n\n"
            f"Answer:\n{answer}"
        )
        out = call_gemini(prompt)
        try:
            data = json.loads(out)
            score = float(data.get("score", 0.5))
            fb = data.get("feedback", "No feedback.")
        except Exception:
            score, fb = 0.5, "Could not parse grader output."
        self.memory.json.record_quiz(uid, topic, score)
        self.memory.save_note(uid, f"QUIZ_GRADE:{topic}:{score}")
        return score, fb



# ==========================================================
# Cell 4 â€” Coordinator
# ==========================================================
class Coordinator:
    def __init__(self, memory):
        self.memory = memory
        self.planner = PlannerAgent("planner", memory)
        self.tutor = TutorAgent("tutor", memory)
        self.retriever = RetrieverAgent("retriever", memory)
        self.evaluator = EvaluatorAgent("evaluator", memory)

    def create_plan(self, uid, goal, hours, weeks):
        payload = json.dumps({"goal": goal, "hours_per_week": hours, "weeks": weeks})
        return self.planner.respond(uid, payload, {})

    def tutor_user(self, uid, question):
        return self.tutor.respond(uid, question, {})

    def get_resources(self, uid, topic):
        return self.retriever.respond(uid, topic, {})

    def create_quiz(self, uid, topic):
        return self.evaluator.generate(uid, topic)

    def grade_quiz(self, uid, topic, answer):
        return self.evaluator.grade(uid, topic, answer)

coord = Coordinator(memory)
print("Coordinator initialized.")



# ==========================================================
# Cell 5 â€” Section 9 Part A: CSS + logo base64 loader
# ==========================================================
from IPython.display import HTML, display
import base64

# Update this path if your file name differs
logo_path = "/kaggle/input/astra-2-png/astra (1).png"
try:
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode("utf-8")
    logo_data_url = f"data:image/png;base64,{logo_b64}"
except Exception as e:
    logo_data_url = ""
    print("âš ï¸� Logo not found at path:", logo_path, " â€”", e)

cosmic_css = """
<style>
body.jp-Notebook { background: radial-gradient(circle at top, #232a47 0%, #111528 45%, #080a14 100%) !important; color: #f5f7ff !important; font-family: 'Segoe UI', system-ui, sans-serif !important; }
.astra-main { width:100%; max-width:1350px; margin:25px auto; padding:32px 36px; border-radius:26px; background:#181f3a; border:1px solid #3b4d88; box-shadow:0 18px 40px rgba(0,0,0,0.55); }
.astra-header { text-align:center; padding:25px 0 35px 0; margin-bottom:25px; border-bottom:1px solid rgba(130,155,235,0.45); }
.astra-logo { width:210px; height:auto; margin-bottom:12px; filter:drop-shadow(0 0 18px rgba(140,170,255,0.75)); }
.main-title { font-size:46px; font-weight:900; background:linear-gradient(135deg,#82a6ff,#d0a0ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:2.5px; margin:0; }
.company-llc { font-size:22px; margin:5px 0 3px 0; color:#d4dcff; letter-spacing:2px; }
.tagline { font-size:14px; margin-top:6px; color:#a8b8ff; }
.astra-input { background:#202954 !important; border-radius:12px !important; border:2px solid #3d5392 !important; color:#f1f4ff !important; padding:10px 14px !important; font-size:14px !important; }
textarea.astra-input { height:120px !important; }
.astra-tabs .p-TabBar .p-TabBar-tab { background:#222b54 !important; border-radius:12px 12px 0 0 !important; padding:10px 20px !important; margin-right:4px !important; color:#d8e1ff !important; font-size:13px !important; }
.astra-tabs .p-mod-current { background:linear-gradient(135deg,#6f94ff,#b480ff) !important; color:#fff !important; font-weight:700 !important; box-shadow:0 6px 16px rgba(110,168,255,0.45); }
.tab-content { background:#1b2243; border-radius:18px; padding:25px 20px; border:1px solid #344475; margin-top:14px; }
.astra-output { background:#151b34; border-radius:14px; border:1px solid #313f70; padding:18px; font-size:13px; margin-top:16px; min-height:150px; max-height:320px; overflow-y:auto; }
.cosmic-btn { width:100%; padding:12px; border-radius:14px; border:none; background:linear-gradient(135deg,#6f94ff,#c580ff); color:white !important; font-weight:700; font-size:13px; margin-top:16px; cursor:pointer; transition:transform .2s ease, box-shadow .2s ease; }
.cosmic-btn:hover { transform:translateY(-1px); box-shadow:0 0 14px rgba(140,175,255,0.7); }
.cosmic-btn-primary { background: linear-gradient(135deg,#5a88ff 0%,#9d6cf6 100%) !important; }
.cosmic-btn-secondary { background: linear-gradient(135deg,#14b8d4 0%,#10b981 100%) !important; }
.section-header { font-size:22px !important; font-weight:700 !important; color:#a8c7ff !important; margin-bottom:18px !important; }
</style>
"""
display(HTML(cosmic_css))
print("âœ… CSS loaded (ASTRA V4).")



# ==========================================================
# Cell 6 â€” Section 9 Part B: Widgets & UI
# ==========================================================
import ipywidgets as widgets
from IPython.display import display

# Ensure coordinator & memory exist
try:
    coord
    memory
except NameError:
    raise RuntimeError("Run core engine cells before UI (coord/memory missing).")

# create default user if none
uid = "shubham_" + uuid.uuid4().hex[:6]
memory.json.update_user(uid, "profile", {"username": "Shubham", "level": "beginner", "preferred_tone": "adaptive"})

# Widgets
user_box = widgets.Text(value=uid, description="User ID:", layout=widgets.Layout(width="60%"))
user_box.add_class("astra-input")

goal_box = widgets.Textarea(value="Become job-ready in Data Science with hands-on projects.", description="Goal:", layout=widgets.Layout(width="100%"))
goal_box.add_class("astra-input")

hours_slider = widgets.IntSlider(value=15, min=5, max=40, step=5, description="Hours/week:")
weeks_slider = widgets.IntSlider(value=12, min=4, max=52, step=4, description="Weeks:")

plan_btn = widgets.Button(description="ğŸš€ Generate Learning Plan")
plan_btn.add_class("cosmic-btn")
plan_out = widgets.Output(); plan_out.add_class("astra-output")

question_box = widgets.Textarea(value="Explain regularization and why it helps avoid overfitting.", description="Question:", layout=widgets.Layout(width="100%"))
question_box.add_class("astra-input")

tutor_btn = widgets.Button(description="ğŸ�“ Ask Tutor"); tutor_btn.add_class("cosmic-btn")
tutor_out = widgets.Output(); tutor_out.add_class("astra-output")

topic_box = widgets.Text(value="feature engineering, sklearn pipelines", description="Topic:"); topic_box.add_class("astra-input")
res_btn = widgets.Button(description="ğŸ”� Get Resources"); res_btn.add_class("cosmic-btn"); res_out = widgets.Output(); res_out.add_class("astra-output")

quiz_topic_box = widgets.Text(value="model evaluation metrics", description="Quiz Topic:"); quiz_topic_box.add_class("astra-input")
quiz_btn = widgets.Button(description="ğŸ§  Create Quiz"); quiz_btn.add_class("cosmic-btn"); quiz_out = widgets.Output(); quiz_out.add_class("astra-output")
answer_box = widgets.Textarea(value="", description="Answer:"); answer_box.add_class("astra-input")
grade_btn = widgets.Button(description="ğŸ“Š Grade Answer"); grade_btn.add_class("cosmic-btn"); grade_out = widgets.Output(); grade_out.add_class("astra-output")

# Callbacks
def on_plan(_):
    with plan_out:
        plan_out.clear_output()
        resp = coord.create_plan(user_box.value, goal_box.value, hours_slider.value, weeks_slider.value)
        print(resp.text)

def on_tutor(_):
    with tutor_out:
        tutor_out.clear_output()
        resp = coord.tutor_user(user_box.value, question_box.value)
        print(resp.text)

def on_res(_):
    with res_out:
        res_out.clear_output()
        resp = coord.get_resources(user_box.value, topic_box.value)
        print(resp.text)

def on_quiz(_):
    with quiz_out:
        quiz_out.clear_output()
        print(coord.create_quiz(user_box.value, quiz_topic_box.value))

def on_grade(_):
    with grade_out:
        grade_out.clear_output()
        score, fb = coord.grade_quiz(user_box.value, quiz_topic_box.value, answer_box.value)
        print(f"SCORE: {score*100:.1f}%\n")
        print("FEEDBACK:", fb)

plan_btn.on_click(on_plan)
tutor_btn.on_click(on_tutor)
res_btn.on_click(on_res)
quiz_btn.on_click(on_quiz)
grade_btn.on_click(on_grade)

# Header
header_html = f"""
<div class='astra-header'>
  <img src="{logo_data_url}" class="astra-logo"/>
  <h1 class="main-title">ASTRA LLC</h1>
  <h3 class="company-llc">LIFELONG COMPANION</h3>
  <p class="tagline">Adaptive Intelligence â€¢ Continuous Growth â€¢ Cosmic Learning</p>
</div>
"""
header = widgets.HTML(header_html)

# Tabs
planner_tab = widgets.VBox([goal_box, hours_slider, weeks_slider, plan_btn, plan_out]); planner_tab.add_class("tab-content")
tutor_tab = widgets.VBox([question_box, tutor_btn, tutor_out]); tutor_tab.add_class("tab-content")
resources_tab = widgets.VBox([topic_box, res_btn, res_out]); resources_tab.add_class("tab-content")
quiz_tab = widgets.VBox([quiz_topic_box, quiz_btn, quiz_out, answer_box, grade_btn, grade_out]); quiz_tab.add_class("tab-content")

tabs = widgets.Tab(children=[planner_tab, tutor_tab, resources_tab, quiz_tab])
tabs.set_title(0, "ğŸ“… Planner"); tabs.set_title(1, "ğŸ�“ Tutor"); tabs.set_title(2, "ğŸ“š Resources"); tabs.set_title(3, "ğŸ§  Quiz")
tabs.add_class("astra-tabs")

ui = widgets.VBox([header, widgets.HBox([user_box], layout=widgets.Layout(width="100%", margin="0 0 10px 0")), tabs])
ui.add_class("astra-main")
display(ui)
print("âœ¨ ASTRA UI V4 loaded (Section 9).")



# ==========================================================
# Cell 7 â€” Section 10 â€” Cognitive Twin Engine (CTE) - fixed
# ==========================================================
import time

class CognitiveTwinEngine:
    def __init__(self, memory):
        self.memory = memory

    def load_twin(self, uid):
        user = self.memory.json.get_user(uid)
        twin = user.get("cognitive_twin")
        if twin is None:
            twin = {
                "topics_mastery": {},
                "mistakes": [],
                "forgetting_curve": {},
                "mood_history": [],
                "last_update": time.time()
            }
            self.memory.json.update_user(uid, "cognitive_twin", twin)
        return twin

    def save_twin(self, uid, twin):
        self.memory.json.update_user(uid, "cognitive_twin", twin)

    def update_from_quiz(self, uid, topic, score):
        twin = self.load_twin(uid)
        mastery = twin["topics_mastery"].get(topic, 0.0)
        new_mastery = 0.7 * mastery + 0.3 * score
        twin["topics_mastery"][topic] = round(new_mastery, 4)
        if score < 0.6:
            twin["mistakes"].append({"topic": topic, "score": score, "timestamp": time.time()})
        self.save_twin(uid, twin)

    def update_from_tutor(self, uid, topic, mood):
        twin = self.load_twin(uid)
        if mood == "confused" and topic in twin["topics_mastery"]:
            twin["topics_mastery"][topic] *= 0.95
        twin["mood_history"].append({"topic": topic, "mood": mood, "timestamp": time.time()})
        self.save_twin(uid, twin)

    def apply_forgetting(self, uid):
        twin = self.load_twin(uid)
        now = time.time()
        days_passed = (now - twin.get("last_update", now)) / (60*60*24)
        twin["last_update"] = now
        if days_passed > 0:
            for topic in list(twin["topics_mastery"].keys()):
                twin["topics_mastery"][topic] *= (0.99 ** days_passed)
        self.save_twin(uid, twin)

    def summarize_cognition(self, uid):
        twin = self.load_twin(uid)
        text = f"Cognitive summary for {uid}\nMastery: {twin['topics_mastery']}\nMistakes: {twin['mistakes'][-3:]}\nRecent moods: {twin['mood_history'][-5:]}"
        if gemini_model:
            try:
                return gemini_model.generate_content(f"Rewrite this learner cognitive data clearly:\n{text}").text
            except Exception:
                return text
        return text

cognitive_twin = CognitiveTwinEngine(memory)
print("Cognitive Twin Engine initialized.")



# ==========================================================
# Cell 8 â€” Section 11 â€” Meta-Intelligence (CognitiveEngine, Workflow, Analytics, Debate)
# ==========================================================
class CognitiveEngine:
    def __init__(self, memory):
        self.memory = memory

    def _get_profile(self, uid):
        user = self.memory.json.get_user(uid)
        return user.get("cognitive_profile", {"topics": {}, "history": []})

    def _save_profile(self, uid, profile):
        self.memory.json.update_user(uid, "cognitive_profile", profile)

    def _decay_mastery(self, topic_state, now):
        last_seen = topic_state.get("last_seen", now)
        mastery = float(topic_state.get("mastery", 0.0))
        days = (now - last_seen) / (60*60*24)
        if days > 0:
            mastery = mastery * (0.99 ** days)
        return mastery

    def update_from_quiz(self, uid, topic, score):
        now = time.time()
        prof = self._get_profile(uid)
        tstate = prof["topics"].get(topic, {"mastery":0.0,"last_seen":now,"n_quizzes":0,"n_confused_signals":0})
        mastery = self._decay_mastery(tstate, now)
        alpha = 0.35
        new_mastery = (1-alpha)*mastery + alpha*float(score)
        tstate["mastery"] = max(0.0, min(1.0, new_mastery))
        tstate["last_seen"] = now
        tstate["n_quizzes"] = int(tstate.get("n_quizzes",0)) + 1
        prof["topics"][topic] = tstate
        prof["history"].append({"timestamp": now, "event": "quiz", "topic": topic, "score": float(score)})
        self._save_profile(uid, prof)

    def update_from_tutor(self, uid, topic, signal):
        now = time.time()
        prof = self._get_profile(uid)
        tstate = prof["topics"].get(topic, {"mastery":0.0,"last_seen":now,"n_quizzes":0,"n_confused_signals":0})
        mastery = self._decay_mastery(tstate, now)
        if signal == "confused":
            mastery *= 0.9
            tstate["n_confused_signals"] = int(tstate.get("n_confused_signals",0)) + 1
        elif signal == "clear":
            mastery = min(1.0, mastery + 0.05)
        tstate["mastery"] = mastery
        tstate["last_seen"] = now
        prof["topics"][topic] = tstate
        prof["history"].append({"timestamp": now, "event":"tutor_signal", "topic": topic, "signal": signal})
        self._save_profile(uid, prof)

    def summarize(self, uid):
        prof = self._get_profile(uid)
        if not prof["topics"]:
            return "No cognitive data yet."
        profile_json = json.dumps(prof, indent=2)
        prompt = f"You are a learning scientist. Summarize this profile:\n{profile_json}\nKeep under 250 words."
        return call_gemini(prompt)

cognitive_engine = CognitiveEngine(memory)

class WorkflowOrchestrator:
    def __init__(self, memory):
        self.memory = memory

    def suggest_next_actions(self, uid, intent):
        user = self.memory.json.get_user(uid)
        plan = user.get("progress", {}).get("plan", "")
        prof = user.get("cognitive_profile", {})
        prompt = f"You are the workflow orchestrator.\nUser intent:\n{intent}\nPlan:\n{plan}\nProfile:\n{prof}\nReturn 3-7 concrete next actions with which agent handles each."
        return call_gemini(prompt)

workflow_orchestrator = WorkflowOrchestrator(memory)

class ProgressAnalytics:
    def __init__(self, memory):
        self.memory = memory

    def basic_stats(self, uid):
        user = self.memory.json.get_user(uid)
        quizzes = user.get("quizzes", {})
        prof = user.get("cognitive_profile", {})
        scores = [v["score"] for v in quizzes.values()] if quizzes else []
        avg_score = float(sum(scores)/len(scores)) if scores else None
        mastered, weak = [], []
        for t,s in prof.get("topics", {}).items():
            m = float(s.get("mastery",0.0))
            if m >= 0.75: mastered.append((t,m))
            if m <= 0.4: weak.append((t,m))
        return {"num_quizzes": len(quizzes), "avg_score": avg_score, "num_topics_tracked": len(prof.get("topics", {})), "mastered_topics": mastered, "weak_topics": weak}

    def analytics_text(self, uid):
        s = self.basic_stats(uid)
        lines = []
        lines.append(f"Quizzes taken: {s['num_quizzes']}")
        if s["avg_score"] is not None: lines.append(f"Average score: {s['avg_score']*100:.1f}%")
        lines.append(f"Topics tracked: {s['num_topics_tracked']}")
        if s["mastered_topics"]: lines.append("Strong topics: " + ", ".join(f"{t} ({m:.2f})" for t,m in s["mastered_topics"]))
        if s["weak_topics"]: lines.append("Weak topics: " + ", ".join(f"{t} ({m:.2f})" for t,m in s["weak_topics"]))
        return "\n".join(lines)

progress_analytics = ProgressAnalytics(memory)

class DebateEngine:
    def __init__(self, memory, coord):
        self.memory = memory
        self.coord = coord

    def refine_plan(self, uid):
        user = self.memory.json.get_user(uid)
        plan = user.get("progress", {}).get("plan", "")
        prof = user.get("cognitive_profile", {})
        prompt = f"Planner vs Critic debate on plan:\nPLAN:\n{plan}\nPROFILE:\n{prof}\nProduce refined plan and short summary."
        refined = call_gemini(prompt)
        progress = user.get("progress", {})
        progress["refined_plan_debate"] = refined
        self.memory.json.update_user(uid, "progress", progress)
        self.memory.save_note(uid, "DEBATE_REFINED_PLAN")
        return refined

debate_engine = DebateEngine(memory, coord)
print("Section 11 engines initialized.")



import os

for f in os.listdir("/kaggle/input/astra-2-png"):
    print(f)



# ==========================================================
# Cell 9 â€” Section 11 ACE Engine (Autonomic Curriculum Evolution)
# ==========================================================
class ACEEngine:
    def __init__(self, memory):
        self.memory = memory

    def _get_user_state(self, uid):
        user = self.memory.json.get_user(uid)
        progress = user.get("progress", {})
        current_plan = progress.get("plan", "No existing plan defined yet.")
        twin = user.get("cognitive_twin", {})
        topics_mastery = twin.get("topics_mastery", {})
        mistakes = twin.get("mistakes", [])
        quizzes = user.get("quizzes", {})
        return user, progress, current_plan, topics_mastery, mistakes, quizzes

    def evolve_plan(self, uid, strategy="balanced", horizon_weeks=None):
        user, progress, current_plan, topics_mastery, mistakes, quizzes = self._get_user_state(uid)
        if horizon_weeks is None:
            horizon_weeks = 8
        prompt = f"You are an adaptive curriculum designer.\nStrategy: {strategy}\nHorizon weeks: {horizon_weeks}\nCurrent plan:\n{current_plan}\nTopic mastery:\n{topics_mastery}\nMistakes:\n{mistakes}\nQuiz history:\n{quizzes}\nProduce a week-by-week evolved plan."
        new_plan = call_gemini(prompt)
        if not isinstance(new_plan, str) or new_plan.strip() == "" or new_plan.startswith("(Offline"):
            new_plan = "ACE offline fallback â€” reuse existing plan:\n\n" + current_plan
        progress["evolved_plan"] = new_plan
        progress["evolved_plan_meta"] = {"strategy": strategy, "horizon_weeks": horizon_weeks, "updated_at": time.time()}
        self.memory.json.update_user(uid, "progress", progress)
        self.memory.save_note(uid, f"ACE_EVOLVED_PLAN_{strategy.upper()}")
        return new_plan

    def explain_changes(self, uid):
        user = self.memory.json.get_user(uid)
        progress = user.get("progress", {})
        old = progress.get("plan", "")
        new = progress.get("evolved_plan", "")
        if not old or not new:
            return "No comparison available."
        prompt = f"Compare ORIGINAL:\n{old}\nEVOLVED:\n{new}\nExplain changes concisely."
        explanation = call_gemini(prompt)
        if not explanation:
            explanation = "Could not generate comparison (offline)."
        self.memory.save_note(uid, "ACE_EXPLAINED_CHANGES")
        return explanation

ace_engine = ACEEngine(memory)
print("ACE engine initialized.")



# ==========================================================
# Cell 10 â€” Section 12 â€” Insights & Evolution Dashboard
# ==========================================================
from IPython.display import HTML
import ipywidgets as widgets

# Safety check
try:
    cognitive_engine
    workflow_orchestrator
    progress_analytics
    debate_engine
    cognitive_twin
except NameError:
    raise RuntimeError("Run Section 11 and Section 10 before Section 12.")

insights_output = widgets.Output(); insights_output.add_class("astra-output")
insights_actions_output = widgets.Output(); insights_actions_output.add_class("astra-output")
insights_plan_output = widgets.Output(); insights_plan_output.add_class("astra-output")
twin_output = widgets.Output(); twin_output.add_class("astra-output")

btn_summary = widgets.Button(description="ğŸ§  View Cognitive Summary", layout=widgets.Layout(width="100%", height="55px"))
btn_summary.add_class("cosmic-btn")
btn_analytics = widgets.Button(description="ğŸ“Š View Progress Analytics", layout=widgets.Layout(width="100%", height="55px")); btn_analytics.add_class("cosmic-btn")
btn_twin = widgets.Button(description="ğŸŒ€ Show Cognitive Twin Raw Data", layout=widgets.Layout(width="100%", height="55px")); btn_twin.add_class("cosmic-btn")
intent_box = widgets.Textarea(value="I want to strengthen feature engineering and model evaluation.", description="Intent:", layout=widgets.Layout(width="100%", height="90px")); intent_box.add_class("astra-input")
btn_next_actions = widgets.Button(description="ğŸš€ Generate Next Actions", layout=widgets.Layout(width="100%", height="55px")); btn_next_actions.add_class("cosmic-btn")
btn_refine_plan = widgets.Button(description="ğŸ”® Refine My Learning Plan (Debate Engine)", layout=widgets.Layout(width="100%", height="55px")); btn_refine_plan.add_class("cosmic-btn")

def on_summary_clicked(btn):
    with insights_output:
        insights_output.clear_output()
        uid = user_box.value.strip()
        print("ğŸ§  Cognitive summary:\n")
        print(cognitive_engine.summarize(uid))

def on_analytics_clicked(btn):
    with insights_output:
        insights_output.clear_output()
        uid = user_box.value.strip()
        print("ğŸ“Š Progress analytics:\n")
        print(progress_analytics.analytics_text(uid))

def on_twin_clicked(btn):
    with twin_output:
        twin_output.clear_output()
        uid = user_box.value.strip()
        print("ğŸŒ€ Cognitive Twin raw:\n")
        print(cognitive_twin.load_twin(uid))

def on_next_actions_clicked(btn):
    with insights_actions_output:
        insights_actions_output.clear_output()
        uid = user_box.value.strip(); intent = intent_box.value.strip()
        print("ğŸš€ Next actions:\n")
        print(workflow_orchestrator.suggest_next_actions(uid, intent))

def on_refine_plan_clicked(btn):
    with insights_plan_output:
        insights_plan_output.clear_output()
        uid = user_box.value.strip()
        print("ğŸ”® Refined plan (debate):\n")
        print(debate_engine.refine_plan(uid))

btn_summary.on_click(on_summary_clicked)
btn_analytics.on_click(on_analytics_clicked)
btn_twin.on_click(on_twin_clicked)
btn_next_actions.on_click(on_next_actions_clicked)
btn_refine_plan.on_click(on_refine_plan_clicked)

insights_tab = widgets.VBox([
    widgets.HTML('<div class="section-header">ğŸ”® ASTRA INSIGHTS & EVOLUTION ENGINE</div>'),
    widgets.HTML("<h4>ğŸ§  Cognitive Understanding</h4>"),
    btn_summary, btn_analytics, btn_twin, insights_output, twin_output,
    widgets.HTML("<h4>ğŸš€ Workflow Orchestration</h4>"),
    intent_box, btn_next_actions, insights_actions_output,
    widgets.HTML("<h4>ğŸ”® Plan Refinement</h4>"),
    btn_refine_plan, insights_plan_output
], layout=widgets.Layout(padding="20px"))
insights_tab.add_class("tab-content")

tabs.children = list(tabs.children) + [insights_tab]
tabs.set_title(4, "ğŸ”® INSIGHTS")
print("Section 12 loaded.")



# ==========================================================
# Cell 11 â€” Section 13 â€” ASTRA Visual Dashboard (Charts)
# ==========================================================
import matplotlib.pyplot as plt
from io import BytesIO
from IPython.display import Image, display as ipy_display

# Safety checks
try:
    progress_analytics
except NameError:
    raise RuntimeError("Run Section 11 before Section 13.")
try:
    tabs
except NameError:
    raise RuntimeError("Run Section 9 before Section 13.")

def get_active_uid():
    try:
        return user_box.value.strip()
    except NameError:
        raise RuntimeError("user_box not found.")

def plot_mastery_bar(uid):
    user = memory.json.get_user(uid)
    prof = user.get("cognitive_profile", {})
    topics = prof.get("topics", {})
    if not topics: return "No cognitive data yet."
    items = sorted([(t,float(v.get("mastery",0.0))) for t,v in topics.items()], key=lambda x: x[1], reverse=True)[:10]
    names = [p[0] for p in items]; vals = [p[1] for p in items]
    fig, ax = plt.subplots(figsize=(7,4)); ax.barh(names[::-1], vals[::-1]); ax.set_xlim(0,1); ax.set_xlabel("Mastery (0-1)"); ax.set_title("Topic Mastery (Top 10)")
    buf = BytesIO(); plt.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig); buf.seek(0); return buf

def plot_quiz_history(uid):
    user = memory.json.get_user(uid); prof = user.get("cognitive_profile", {}); hist = prof.get("history", [])
    quiz_events = [h for h in hist if h.get("event") == "quiz"]
    if not quiz_events: return "No quiz history yet."
    xs = list(range(1, len(quiz_events)+1)); ys = [float(e.get("score",0.0)) for e in quiz_events]
    fig, ax = plt.subplots(figsize=(7,3)); ax.plot(xs, ys, marker="o"); ax.set_ylim(0,1); ax.set_xlabel("Quiz #"); ax.set_ylabel("Score"); ax.set_title("Quiz Timeline")
    buf = BytesIO(); plt.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig); buf.seek(0); return buf

dash_output = widgets.Output(); dash_output.add_class("astra-output")
refresh_dash_btn = widgets.Button(description="ğŸ“Š Refresh Dashboard", layout=widgets.Layout(width="100%", height="55px")); refresh_dash_btn.add_class("cosmic-btn")

def on_refresh_dashboard(btn):
    with dash_output:
        dash_output.clear_output()
        uid = get_active_uid(); print(f"ğŸ“Œ User: {uid}\n"); print("=== Basic Analytics ==="); print(progress_analytics.analytics_text(uid)); print("\n=== Visuals ===\n")
        buf1 = plot_mastery_bar(uid)
        if isinstance(buf1, str): print(buf1)
        else: ipy_display(Image(data=buf1.read())); print("\n")
        buf2 = plot_quiz_history(uid)
        if isinstance(buf2, str): print(buf2)
        else: ipy_display(Image(data=buf2.read()))

refresh_dash_btn.on_click(on_refresh_dashboard)

dashboard_tab = widgets.VBox([widgets.HTML("<div class='section-header'>ğŸ“Š ASTRA Progress Dashboard</div>"), widgets.HTML("<p>Visual view of your mastery and quiz trajectory.</p>"), refresh_dash_btn, dash_output], layout=widgets.Layout(padding="20px"))
dashboard_tab.add_class("tab-content")
tabs.children = list(tabs.children) + [dashboard_tab]
tabs.set_title(len(tabs.children)-1, "ğŸ“Š DASHBOARD")
print("Section 13 loaded.")



# ==========================================================
# Cell 12 â€” Section 14 â€” ASTRA Auto-Coach (Autonomous Cycle)
# ==========================================================
# Safety
try:
    cognitive_engine; workflow_orchestrator; debate_engine; progress_analytics
except NameError:
    raise RuntimeError("Run engines first.")

auto_output = widgets.Output(); auto_output.add_class("astra-output")
intent_auto = widgets.Textarea(value="Strengthen my weak areas and give me a realistic 48-hour study plan.", description="Intent:", layout=widgets.Layout(width="100%", height="90px"))
intent_auto.add_class("astra-input")
run_auto_btn = widgets.Button(description="ğŸ¤– Run Auto-Coach Cycle", layout=widgets.Layout(width="100%", height="55px")); run_auto_btn.add_class("cosmic-btn-primary")

def get_active_uid_safe():
    try:
        return user_box.value.strip()
    except NameError:
        raise RuntimeError("user_box not found")

def on_run_auto(btn):
    with auto_output:
        auto_output.clear_output()
        uid = get_active_uid_safe(); intent = intent_auto.value.strip()
        print(f"ğŸ¤– ASTRA AUTO-COACH CYCLE for user: {uid}\n")
        print("=== Current Snapshot ==="); print(progress_analytics.analytics_text(uid)); print("\n")
        print("=== Refined Plan (Debate) ==="); refined = debate_engine.refine_plan(uid); print(refined[:2000]); print("\n")
        print("=== Next Actions ==="); next_actions = workflow_orchestrator.suggest_next_actions(uid, intent); print(next_actions); print("\n")
        print("=== Cognitive Summary ==="); summary = cognitive_engine.summarize(uid); print(summary)
        print("\nâœ… Auto-coach cycle complete.")

run_auto_btn.on_click(on_run_auto)

auto_tab = widgets.VBox([widgets.HTML("<div class='section-header'>ğŸ¤– ASTRA AUTO-COACH</div>"), widgets.HTML("<p>Autonomous cycle: refine plan, suggest actions, summarize cognition.</p>"), intent_auto, run_auto_btn, auto_output], layout=widgets.Layout(padding="20px"))
auto_tab.add_class("tab-content")
tabs.children = list(tabs.children) + [auto_tab]
tabs.set_title(len(tabs.children)-1, "ğŸ¤– AUTO")
print("Section 14 (Auto-Coach) loaded.")





