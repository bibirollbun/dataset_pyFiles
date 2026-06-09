print("AgentAid â€” Multi-Agent Student Support System (Corrected Version)")


import os, json, random, math, logging
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_aid")


CONFIG = {
    "api_mode": "simulation",
    "followup_days": 7
}

def llm_chain(prompt):
    """Simulated LLM â€” safe for Kaggle."""
    if "tutor" in prompt.lower():
        return "3 sessions/week, 30 mins each, fundamentals + practice quizzes."
    if "wellbeing" in prompt.lower():
        return "Daily breathing, weekly reflections, maintain sleep hygiene."
    if "email" in prompt.lower():
        return "Dear Parent, we detected some learning concerns. Let's schedule support."
    return "Simulated response."


def generate_synthetic_students(n=800, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    rows = []

    for sid in range(1, n+1):
        gpa = round(max(0.5, min(4.0, np.random.normal(3.0, 0.6))), 2)
        attendance = round(max(60, min(100, np.random.normal(92, 6))), 1)
        mood = random.choice(["happy","neutral","stressed","sad"])

        risk = 0
        if gpa < 2.5: risk += 0.3
        if attendance < 90: risk += 0.3
        if mood in ["sad","stressed"]: risk += 0.2

        at_risk = 1 if risk > 0.35 else 0

        rows.append({
            "student_id": sid,
            "name": f"Student_{sid}",
            "gpa": gpa,
            "attendance": attendance,
            "mood": mood,
            "last_quiz_score": round(max(0, min(100, np.random.normal(75 - (risk*20), 12))), 1),
            "at_risk": at_risk
        })
    return pd.DataFrame(rows)

df = generate_synthetic_students()
df.head()


features = ["gpa", "attendance", "last_quiz_score"]
X = df[features]
y = df["at_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

clf = RandomForestClassifier(n_estimators=120, random_state=42)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print(classification_report(y_test, pred))



student_store = df.set_index("student_id").to_dict(orient="index")
progress_memory = {}



def get_student_record(student_id):
    return student_store.get(student_id, None)

def predict_risk(student):
    X = np.array([[student["gpa"], student["attendance"], student["last_quiz_score"]]])
    prob = clf.predict_proba(X)[0][1]
    level = "CRITICAL" if prob>0.80 else "HIGH" if prob>0.50 else "MODERATE" if prob>0.25 else "LOW"
    return {"risk_score": float(prob), "risk_level": level}

def generate_tutor_plan(student, level):
    plan = llm_chain("tutor plan")
    return {"type": "Tutoring", "sessions_per_week": 3 if level in ["HIGH","CRITICAL"] else 1,
            "details": plan}

def generate_wellbeing_plan(student, level):
    details = llm_chain("wellbeing plan")
    return {"tasks": ["breathing", "reflection", "sleep hygiene"], "details": details}

def send_notification(recipient, student, message):
    record = {
        "time": str(datetime.now()),
        "recipient": recipient,
        "student_id": student.get("student_id", None),
        "message_preview": message[:120]
    }
    logger.info("Notification:", record)
    return {"sent": True, "record": record}

def schedule_followup(student_id, days=7):
    date = str((datetime.now() + timedelta(days=days)).date())
    if student_id not in progress_memory:
        progress_memory[student_id] = []
    progress_memory[student_id].append({"action": "scheduled_followup", "followup": date})
    return {"followup_date": date}


def a2a(sender, receiver, payload):
    return {"from": sender, "to": receiver, "payload": payload, "timestamp": str(datetime.now())}


class BaseAgent:
    def __init__(self, name): self.name = name

class DataIngestAgent(BaseAgent):
    def fetch(self, student_id):
        rec = get_student_record(student_id)
        if rec:
            rec = rec.copy()
            rec["student_id"] = student_id   # <-- FIXED
        return a2a(self.name, "RiskAgent", {"student": rec})

class RiskAgent(BaseAgent):
    def analyze(self, student):
        return a2a(self.name, "Orchestrator", {"student": student, "risk": predict_risk(student)})

class TutorAgent(BaseAgent):
    def plan(self, student, level):
        return a2a(self.name, "Orchestrator", {"tutor_plan": generate_tutor_plan(student, level)})

class WellbeingAgent(BaseAgent):
    def plan(self, student, level):
        return a2a(self.name, "Orchestrator", {"wellbeing_plan": generate_wellbeing_plan(student, level)})

class OutreachAgent(BaseAgent):
    def notify(self, student, level):
        msg = llm_chain("email")
        return a2a(self.name, "Orchestrator", {"notification": send_notification("parent", student, msg)})

class OrchestratorAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)
        self.logs = []

    def run(self, student_id, followup_days=7):
        di = DataIngestAgent("DataIngest")
        ra = RiskAgent("RiskAgent")
        ta = TutorAgent("TutorAgent")
        wa = WellbeingAgent("WellbeingAgent")
        oa = OutreachAgent("OutreachAgent")

        msg1 = di.fetch(student_id)
        student = msg1["payload"]["student"]
        if not student:
            return {"status": "no_data"}

        msg2 = ra.analyze(student)
        risk = msg2["payload"]["risk"]

        msg3 = ta.plan(student, risk["risk_level"])
        msg4 = wa.plan(student, risk["risk_level"])

        notification = None
        follow = None

        if risk["risk_level"] in ["HIGH","CRITICAL"]:
            msg5 = oa.notify(student, risk["risk_level"])
            notification = msg5["payload"]["notification"]
            follow = schedule_followup(student_id, followup_days)

        # update memory
        if student_id not in progress_memory:
            progress_memory[student_id] = []
        progress_memory[student_id].append({
            "risk": risk,
            "date": str(datetime.now().date())
        })

        entry = {
            "student": student,
            "risk": risk,
            "tutor": msg3["payload"]["tutor_plan"],
            "wellbeing": msg4["payload"]["wellbeing_plan"],
            "notification": notification,
            "followup": follow
        }

        self.logs.append(entry)
        logger.info(f"Orchestrator logged: {risk['risk_level']}")
        return {"status": "ok", "entry": entry}


orch = OrchestratorAgent("Orchestrator")

sample_ids = df.sample(20, random_state=42)["student_id"].tolist()

results = []
for sid in sample_ids:
    results.append(orch.run(sid, CONFIG["followup_days"]))


from collections import Counter

risk_levels = []
for r in results:
    if r.get("status") == "ok":
        risk_levels.append(r["entry"]["risk"]["risk_level"])

print("Risk distribution:", Counter(risk_levels))


os.makedirs("/kaggle/working/agentaid_output", exist_ok=True)

with open("/kaggle/working/agentaid_output/progress_memory.json","w") as f:
    json.dump(progress_memory, f, indent=2)

with open("/kaggle/working/agentaid_output/logs.json","w") as f:
    json.dump(orch.logs, f, indent=2)

print("Saved output files.")


def print_student_report(sid):
    rec = get_student_record(sid)
    rec2 = rec.copy()
    rec2["student_id"] = sid
    print("=== Student Report ===")
    print(rec2)
    print("History:", progress_memory.get(sid, []))

print_student_report(sample_ids[0])


import pandas as pd

log_df = pd.DataFrame([
    {
        "student_id": log["student"]["student_id"],
        "risk_level": log["risk"]["risk_level"],
        "risk_score": round(log["risk"]["risk_score"], 2),
        "tutor_sessions": log["tutor"]["sessions_per_week"],
        "notification_sent": log["notification"] is not None
    }
    for log in orch.logs
])

log_df.head()



def color_risk(level):
    colors = {"LOW":"ðŸŸ¢ LOW", "MODERATE":"ðŸŸ¡ MODERATE", "HIGH":"ðŸŸ  HIGH", "CRITICAL":"ðŸ”´ CRITICAL"}
    return colors[level]

for r in results:
    if r["status"] == "ok":
        print(
            f"Student {r['entry']['student']['student_id']} â†’ "
            f"Risk: {color_risk(r['entry']['risk']['risk_level'])}"
        )



def display_student_summary(entry):
    student = entry["student"]
    risk = entry["risk"]
    tutor = entry["tutor"]
    well = entry["wellbeing"]
    
    print("â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    print(f"Student: {student['name']} ({student['student_id']})")
    print(f"Risk Level: {color_risk(risk['risk_level'])}")
    print(f"Risk Score: {risk['risk_score']:.2f}")
    print(f"Tutoring: {tutor['sessions_per_week']} sessions/week")
    print("Wellbeing Tasks:", ", ".join(well["tasks"]))
    if entry["notification"]:
        print("ðŸ“© Notification Sent to Parent")
    print("â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€\n")

# show dashboard for first 5 students
for x in results[:5]:
    if x["status"] == "ok":
        display_student_summary(x["entry"])



risk_counts = df["at_risk"].value_counts()
plt.figure(figsize=(5,4))
plt.bar(["Not at Risk", "At Risk"], risk_counts.values)
plt.title("Risk Class Distribution")
plt.show()



from collections import Counter
pred_risks = Counter(risk_levels)
plt.bar(pred_risks.keys(), pred_risks.values())
plt.title("Predicted Risk Levels")
plt.show()





