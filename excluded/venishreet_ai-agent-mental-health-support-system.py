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





# Optional â€” Kaggle might block external installs
# !pip install google-generativeai adk



# Imports + fallback ADK simulation (keeps notebook runnable anywhere)
import json
from datetime import datetime
from typing import Dict, Any, List

# If a real Agent class is available from ADK, keep it. Otherwise we'll define a simple fallback Agent/Tool.
try:
    # try typical ADK import (your environment may or may not have this)
    from adk import Agent, Tool, run as adk_run  # if this succeeds, we'll use those classes
    ADK_AVAILABLE = True
except Exception:
    ADK_AVAILABLE = False

# Fallback simple Tool & Agent implementations (used only when ADK not present)
if not ADK_AVAILABLE:
    class Tool:
        def __init__(self, fn, name=None):
            self.fn = fn
            self.name = name or fn.__name__
        @classmethod
        def from_function(cls, fn):
            return cls(fn, name=fn.__name__)
        def call(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    class Agent:
        def __init__(self, name, instructions="", tools=None):
            self.name = name
            self.instructions = instructions or ""
            self.tools = {t.name: t for t in (tools or [])}
        def run(self, prompt: str) -> Dict[str, Any]:
            """
            A simple rule-based runner for the fallback environment.
            For SupportAgent we will rely on a custom behavior added later.
            """
            # Default fallback: basic empathetic reply (will be overridden for SupportAgent)
            return {"reply": "I'm here to listen. Could you tell me more about how you're feeling?"}

    def adk_run(agent, user_input):
        # unify interface
        if hasattr(agent, 'run'):
            return agent.run(user_input)
        raise RuntimeError("Agent has no run method")

print("ADK available:", ADK_AVAILABLE)



# Tools and in-memory journal
mood_history: List[Dict[str,str]] = []

def emotion_tool(text: str) -> str:
    keywords = {
        'anxiety': ['anxious', 'anxiety', 'nervous', 'scared', 'overwhelmed'],
        'stress': ['stressed', 'overwhelmed', 'pressure', 'deadline'],
        'sad': ['sad', 'down', 'depressed', 'hopeless', 'cry'],
        'angry': ['angry', 'mad', 'irritated', 'frustrated'],
        'neutral': []
    }
    t = text.lower()
    for emo, keys in keywords.items():
        for k in keys:
            if k in t:
                return emo
    return 'neutral'

def resource_tool(emotion: str) -> dict:
    db = {
        'anxiety': ['4-4-6 breathing (inhale 4s, hold 4s, exhale 6s)', 'Grounding: 5-4-3-2-1 exercise'],
        'stress': ['Short walk and stretch', 'Break tasks into 15-min chunks (Pomodoro)'],
        'sad': ['Write 3 things you feel grateful for', 'Short journaling prompt: "What helped me today?"'],
        'angry': ['Step back for 2 minutes and breathe', 'Physical release: shake out arms, short walk'],
        'neutral': ['Take a short mindful pause', 'Drink water and breathe']
    }
    return {'recommended': db.get(emotion, db['neutral'])}

def journal_tool(entry: str) -> dict:
    timestamp = datetime.utcnow().isoformat()
    rec = {'entry': entry, 'time': timestamp}
    mood_history.append(rec)
    return {'status': 'saved', 'entry': rec}

def safety_check_tool(text: str) -> str:
    crisis_words = ['suicide', 'kill myself', 'end my life', 'want to die', 'want to die']
    t = text.lower()
    return 'crisis' if any(k in t for k in crisis_words) else 'safe'

# Wrap into Tool objects (ADK or fallback)
emotion_action = Tool.from_function(emotion_tool)
resource_action = Tool.from_function(resource_tool)
journal_action = Tool.from_function(journal_tool)
safety_action = Tool.from_function(safety_check_tool)

print("Tools defined")

# ---------------------------------------
# Long-Running Operation (Simulated LRO)
# ---------------------------------------

class CrisisEscalationLRO:
    """
    Simulated long-running operation for crisis escalation.
    Works without ADK, safe for Kaggle.
    """

    def start(self):
        return {
            "status": "paused",
            "message": "â�³ Crisis detected. Escalation paused â€” waiting for human approval.",
            "next_step": "approve_or_cancel"
        }

    def resume(self, approval: str):
        approval = approval.strip().lower()
        if approval == "approve":
            return {
                "status": "approved",
                "message": "âœ… Escalation approved. Emergency protocol activated."
            }
        else:
            return {
                "status": "cancelled",
                "message": "â�Œ Escalation cancelled by human."
            }

# Create instance
crisis_escalation_lro = CrisisEscalationLRO()



# -------------------------
# SupportAgent (fixed to always produce the Option 2 style reply)
# -------------------------
support_instructions = """
You are a Mental Health Support Agent.

When a user shares how they feel, ALWAYS produce a clear, complete, helpful reply following this structure:

1) A short empathetic acknowledgement of the user's feeling.
2) A brief explanation referencing the detected emotion (if available).
3) One practical, actionable coping step (e.g., a breathing exercise, grounding, brief journaling prompt, or study tip).
4) A warm reassurance to close.

Responses should be 3-5 sentences. Use tools when available for emotion detection and resource suggestions. Do not reply with a single-line prompt asking for more information.
"""

# If ADK Agent class supports custom logic via tools, we still attach tools.
support_agent = Agent(
    name='SupportAgent',
    instructions=support_instructions,
    tools=[emotion_action, resource_action, journal_action]
)

# Fallback: if ADK is not available, override the run method to build the full Option 2 reply.
if not ADK_AVAILABLE:
    def support_run_override(self, prompt: str):
        # 1) emotion detection
        emo = None
        if 'emotion_action' in globals():
            try:
                emo = self.tools['emotion_tool'].call(prompt) if 'emotion_tool' in self.tools else emotion_tool(prompt)
            except Exception:
                emo = emotion_tool(prompt)
        if not emo:
            emo = emotion_tool(prompt)

        # 2) pick resources
        resources = resource_tool(emo)
        first_rec = resources.get('recommended', [])[0] if isinstance(resources, dict) else resources[0]

        # 3) craft Option 2 style reply
        # empathetic acknowledgement
        ack = f"Itâ€™s completely understandable to feel {emo} right now." if emo != 'neutral' else "Thank you for sharing how you're feeling."
        # short explanation
        expl = "This often happens when we have pressure or uncertainty." if emo in ('anxiety','stress') else "That feeling can impact your focus and energy."
        # practical step
        step = f"Try this: {first_rec}."
        # reassurance
        closing = "You're doing your best â€” small steps help and you're not alone."

        reply = " ".join([ack, expl, step, closing])
        # also include structured fields
        return {"reply": reply, "emotion": emo, "resources": resources, "tool_calls": [('emotion_tool', emo)]}

    # attach override
    support_agent.run = support_run_override.__get__(support_agent, Agent)

# -------------------------
# Safety Agent (simple)
# -------------------------
safety_agent = Agent(
    name='SafetyAgent',
    instructions="Detect crisis language and flag escalation.",
    tools=[safety_action]
)

print("Agents instantiated")



# -----------------------------------------------------
# Multi-Agent Coordinator + Simulated Long-Running Operation (LRO)
# -----------------------------------------------------

def mental_health_system(user_input: str, require_human_approval: bool = True):
    """
    Coordinator that:
    1) Runs SafetyAgent
    2) If crisis â†’ triggers Long-Running Operation (pause)
    3) If no crisis â†’ runs SupportAgent
    Returns:
        - reply (normal cases)
        - emotion, resources (normal cases)
        - status + message (crisis cases)
    """

    # -------------------------
    # 1. SAFETY CHECK
    # -------------------------
    safety_result = safety_action.call(user_input)

    if "crisis" in safety_result:
        # Trigger Simulated LRO Tool (PAUSE)
        lro_state = crisis_escalation_lro.start()

        return {
            "status": "paused",
            "message": lro_state["message"],
            "next_step": "Use crisis_escalation_lro.resume('approve') to continue."
        }

    # -------------------------
    # 2. NORMAL CASE â†’ SUPPORT AGENT
    # -------------------------
    try:
        out = support_agent.run(user_input)
    except Exception:
        out = {"reply": "I'm here to support you.", "tool_calls": []}

    # Normalize SupportAgent output
    if isinstance(out, dict):
        reply = out.get("reply", "")
        tool_calls = out.get("tool_calls", [])
    else:
        reply = str(out)
        tool_calls = []

    # -------------------------
    # 3. EMOTION DETECTION
    # -------------------------
    try:
        emotion = emotion_tool(user_input)
    except Exception:
        emotion = "neutral"

    # -------------------------
    # 4. RESOURCE SUGGESTIONS
    # -------------------------
    try:
        resources = resource_action.call(emotion)
    except Exception:
        resources = {}

    # -------------------------
    # RETURN FINAL RESPONSE
    # -------------------------
    return {
        "reply": reply,
        "emotion": emotion,
        "resources": resources,
        "tool_calls": tool_calls
    }

print("Coordinator ready")



# Demo: run a few example inputs to show outputs (non-interactive)
def demo_conversations(require_human_approval=False):
    examples = [
        "I am feeling very anxious about my exams",
        "I feel down and like crying",
        "I am tired and overwhelmed with assignments",
        "Sometimes I think about wanting to end my life"
    ]
    for ex in examples:
        print('\n--- User:', ex)
        out = mental_health_system(ex, require_human_approval=require_human_approval)
        print('Agent output:')
        print(json.dumps(out, indent=2))

# run demo (set require_human_approval=False so it won't pause)
demo_conversations(require_human_approval=False)



# Try-It-Yourself (Edit the text manually)
text = "I am feeling very anxious about my exams"

result = mental_health_system(text, require_human_approval=False)

result



# This creates a file called submission.csv for Kaggle
import pandas as pd

# Make a small dummy file
submission = pd.DataFrame({
    "ID": [1],  # just a placeholder
    "Prediction": ["AI Mental Health Agent ready"]  # placeholder output
})

submission.to_csv("submission.csv", index=False)

print("Submission file created!")

