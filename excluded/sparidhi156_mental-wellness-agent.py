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


# === CELL 1: Imports & Small Activity DB ===
import json
import datetime
from typing import Dict, List, Any
from collections import Counter
import os

# Activity DB (tool simulation)
ACTIVITY_DB = [
    {"id": 1, "type": "breathing", "title": "4-7-8 Breathing", "duration_min": 2, "tags": ["calm","stress"]},
    {"id": 2, "type": "walk", "title": "10-minute Walk", "duration_min": 10, "tags": ["energy","low_mood"]},
    {"id": 3, "type": "music", "title": "Calm Piano Playlist", "duration_min": 20, "tags": ["calm","relax"]},
    {"id": 4, "type": "journaling", "title": "5-min Gratitude Journal", "duration_min": 5, "tags": ["reflect","low_mood"]},
    {"id": 5, "type": "affirmation", "title": "Positive Affirmations", "duration_min": 1, "tags": ["motivation"]},
    {"id": 6, "type": "stretch", "title": "Light Stretching", "duration_min": 5, "tags": ["energy"]},
    {"id": 7, "type": "guided_meditation", "title": "Short Guided Meditation", "duration_min": 8, "tags": ["calm","stress"]},
    {"id": 8, "type": "sleep_tip", "title": "Wind-down Routine", "duration_min": 15, "tags": ["sleep","relax"]}
]


# === CELL 2: Memory Module with Persistence ===
class Memory:
    def __init__(self, path: str = "memory.json"):
        self.path = path
        self.store = {"interactions": []}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.store = json.load(f)
            except Exception:
                # If corrupted, start fresh
                self.store = {"interactions": []}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.store, f, indent=2, ensure_ascii=False)

    def add_interaction(self, user_text: str, mood: str, timestamp: str):
        self.store.setdefault("interactions", []).append({
            "text": user_text,
            "mood": mood,
            "time": timestamp
        })
        self._save()

    def recent_moods(self, n=3) -> List[str]:
        return [x["mood"] for x in self.store.get("interactions", [])[-n:]]

    def summary(self, last_n: int = 10) -> str:
        moods = [x["mood"] for x in self.store.get("interactions", [])[-last_n:]]
        if not moods:
            return "No prior interactions."
        c = Counter(moods)
        most, count = c.most_common(1)[0]
        return f"Recent moods summary: {most} ({count} times in last {len(moods)} interactions)."

# Create memory instance (will load/export to memory.json)
memory = Memory()



# === CELL 3: Emotion Analyzer (Rule-based) ===
def emotion_analyzer(user_text: str) -> Dict[str, Any]:
    s = (user_text or "").lower()
    mood_keywords = {
        "happy": ["happy","good","great","awesome","fine","excited","joy","yay","glad"],
        "sad": ["sad","down","depressed","unhappy","miserable","lonely","low","demotivated"],
        "stressed": ["stressed","anxious","anxiety","overwhelmed","pressure","tensed","panic","nervous"],
        "tired": ["tired","exhausted","sleepy","drained","fatigue","sleepy"],
        "angry": ["angry","mad","furious","annoyed","irritated"],
        "neutral": ["ok","okay","normal","fine"]
    }

    found = {}
    for mood, keys in mood_keywords.items():
        for k in keys:
            if k in s:
                found[mood] = found.get(mood, 0) + 1

    if not found:
        # Simple heuristics fallback
        if any(tok in s for tok in ["!", "love", "yay", "celebrate"]):
            mood = "happy"
        elif any(tok in s for tok in ["don't know", "confused", "idk"]):
            mood = "neutral"
        else:
            mood = "neutral"
    else:
        mood = max(found.items(), key=lambda x: x[1])[0]

    # Simple stress scoring
    stress_score = 3
    if mood == "stressed":
        stress_score = 7
    elif mood in ("sad","tired"):
        stress_score = 5
    elif mood == "happy":
        stress_score = 2
    elif mood == "angry":
        stress_score = 6
    elif mood == "neutral":
        stress_score = 3

    return {"mood": mood, "stress_score": stress_score, "found_keywords": found}



# === CELL 4: Planner Agent (Tool-calling simulation) ===
def recommend_activities(mood: str, stress_score: int, top_k: int = 3) -> List[Dict[str, Any]]:
    tag_map = {
        "happy": ["energy","motivation"],
        "sad": ["low_mood","reflect"],
        "stressed": ["calm","stress","relax"],
        "tired": ["sleep","relax","calm"],
        "angry": ["calm","reflect"],
        "neutral": ["energy","motivation","relax"]
    }
    tags = tag_map.get(mood, ["relax"])
    scored = []
    for item in ACTIVITY_DB:
        score = 0
        # match tags
        for t in item["tags"]:
            if t in tags:
                score += 2
        # prefer short activities under high stress
        if stress_score >= 6 and item["duration_min"] <= 5:
            score += 1
        # deterministic tie-breaker
        score += 0.01 * item["id"]
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]



# === CELL 5: Orchestrator (Multi-Agent Flow) ===
def run_session(user_text: str, mem: Memory, top_k: int = 4) -> Dict[str, Any]:
    timestamp = datetime.datetime.utcnow().isoformat()
    emo = emotion_analyzer(user_text)
    mood = emo["mood"]
    stress = emo["stress_score"]
    mem.add_interaction(user_text=user_text, mood=mood, timestamp=timestamp)
    recs = recommend_activities(mood=mood, stress_score=stress, top_k=top_k)
    response = {
        "timestamp": timestamp,
        "input_text": user_text,
        "mood": mood,
        "stress_score": stress,
        "found_keywords": emo.get("found_keywords", {}),
        "recommendations": recs,
        "memory_summary": mem.summary()
    }
    return response

# Helper: pretty print JSON
def pretty_print(obj: Dict[str, Any]):
    print(json.dumps(obj, indent=2, ensure_ascii=False))



# === CELL 6: Demo Examples (Run this cell to show sample interactions) ===
examples = [
    "I am feeling really anxious and stressed about my exams.",
    "I feel tired and exhausted today.",
    "I am so happy!! I got a good result",
    "I feel low and not motivated to do anything"
]

for ex in examples:
    out = run_session(ex, memory)
    pretty_print(out)
    print("\n---\n")


# === CELL 7: Quick Manual Test Cell ===
# Use this cell to try your own messages. Edit the string and run the cell.
test_input = "I feel nervous and overwhelmed with college work."
result = run_session(test_input, memory)
pretty_print(result)



# === CELL 8: Simple Unit Tests (Optional, demonstrates reproducibility) ===
assert emotion_analyzer("I am very happy today!")["mood"] == "happy"
assert emotion_analyzer("I feel so stressed and anxious")["mood"] == "stressed"
assert emotion_analyzer("I am tired and exhausted")["mood"] == "tired"

# Check planner returns at least one recommendation
_r = recommend_activities("stressed", 7, top_k=2)
assert len(_r) >= 1

print("All quick checks passed.")


# === CELL 9: Export Last Session Result to JSON (for submission / demo) ===
# This writes 'last_session.json' containing the most recent interaction
def export_last_session(mem: Memory, filename: str = "last_session.json"):
    interactions = mem.store.get("interactions", [])
    if not interactions:
        print("No interactions to export.")
        return
    last = interactions[-1]
    # regenerate the full response for the last interaction
    response = run_session(last["text"], mem, top_k=4)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2, ensure_ascii=False)
    print(f"Exported last session to {filename}")

# Run export once (optional)
export_last_session(memory)



# Interactive loop (not supported in Kaggle notebook)
# if __name__ == "__main__":
#     while True:
#         user_text = input("> ").strip()
#         if not user_text: break
#         pretty_print(run_session(user_text, memory))


run_session("I feel happy today", memory)

