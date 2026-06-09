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


# Run once (already ran in your notebook). If you run it again it's okay.
!pip install google-generativeai || true


# --- Load secret from Kaggle Secrets and configure GenAI client ---
from kaggle_secrets import UserSecretsClient
import os
import google.generativeai as genai

user_secrets = UserSecretsClient()
# Replace "GOOGLE_API_KEY" with the exact name you used in the Kaggle UI
secret_value = user_secrets.get_secret("GOOGLE_API_KEY")

# put into env the name the notebook expects
os.environ["GEMINI_API_KEY"] = secret_value

# configure the client
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
print("Configured Gemini from Kaggle secrets (env GEMINI_API_KEY set).")


# --- Imports and helpers ---
from google.generativeai import GenerativeModel   # for model-level calls
import google.generativeai as genai
import os, json, time
from datetime import datetime

# quick check
print("GEN key present:", bool(os.environ.get("GEMINI_API_KEY")))


import os, requests, json
key = os.environ.get("GEMINI_API_KEY")
if not key:
    raise RuntimeError("No GEMINI_API_KEY in env — set your Kaggle secret name to GOOGLE_API_KEY")

url = "https://generativelanguage.googleapis.com/v1beta/models"
r = requests.get(url, params={"key": key}, timeout=30)
print("HTTP", r.status_code)
data = r.json()
# Print a short human-friendly list
models = data.get("models", [])
print(f"Found {len(models)} models\n")
for m in models:
    name = m.get("name") or m.get("model") or m.get("id")
    supported = m.get("supportedMethods") or m.get("supported_methods") or []
    print("NAME:", name)
    if supported:
        print("  supportedMethods:", supported)
    # if there's any extra metadata, print a tiny bit
    desc = m.get("description")
    if desc:
        print("  desc:", desc[:120].replace("\n"," "))
    print()
# Save JSON to a file to paste/share if needed
with open("/tmp/models_list.json","w") as f:
    json.dump(data, f, indent=2)
print("Model list saved to /tmp/models_list.json — attach or screenshot the output if you want me to pick one.")


# --- Robust GenAI caller: tries a list of model names until one works ---
PREFERRED_MODELS = [
    "models/gemini-flash-latest",   # highest chance based on your list
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-pro-latest",
    "models/gemini-2.0-flash",
]

def call_genai(prompt: str, preferred_models=None, max_tokens: int = 300) -> str:
    """
    Try a list of model names with GenerativeModel.generate_content until one works.
    Returns the text output (best effort).
    """
    models_to_try = preferred_models or PREFERRED_MODELS
    last_exc = None
    for mname in models_to_try:
        try:
            model = genai.GenerativeModel(mname)
            # generate_content often accepts a list of strings or a single string; try both
            try:
                resp = model.generate_content(prompt)
            except Exception:
                # some versions expect contents as a list
                resp = model.generate_content([prompt])
            # resp may expose .text or be printable — handle both
            if hasattr(resp, "text") and resp.text:
                return resp.text
            # sometimes resp is a dict-like or object; convert robustly
            s = str(resp)
            if s:
                return s
        except Exception as e:
            last_exc = e
            # try next model
            # (do not crash here so we can attempt fallback models)
    # if none worked, raise a helpful error showing last exception
    raise RuntimeError(f"No compatible GenAI model worked. Last error: {last_exc}")


def summarize_text(text: str) -> str:
    prompt = f"Summarize this text into concise bullet points for review:\n\n{text}"
    return call_genai(prompt, max_tokens=250)


def generate_flashcards(text: str, n_cards: int = 5) -> list:
    prompt = (
        f"Read the study text below and generate {n_cards} short flashcards (Q/A).\n"
        "Return as a JSON list of objects each with keys 'q' and 'a'.\n\n"
        f"{text}"
    )
    raw = call_genai(prompt, max_tokens=400)
    try:
        cards = json.loads(raw)
        # ensure a list of dicts
        if isinstance(cards, list):
            return cards
    except Exception:
        pass
    # fallback: naive parse -> return single-card with whole text
    return [{"q": "Main idea", "a": raw[:300]}]


def generate_quiz_questions(text: str, n_questions: int = 3) -> list:
    prompt = (
        f"Create {n_questions} practice questions from the text below. "
        "Include 1 MCQ (4 options) and the rest short answer. "
        "Return a JSON array of objects with keys: type, question, options (opt), answer.\n\n"
        f"{text}"
    )
    raw = call_genai(prompt, max_tokens=400)
    try:
        qs = json.loads(raw)
        if isinstance(qs, list):
            return qs
    except Exception:
        pass
    # fallback single short question
    return [{"type":"short","question":"Summarize main idea","answer": raw[:200]}]


# --- pipeline runner and a tiny memory store ---
MEMORY = []

def run_study_helper_pipeline(text: str) -> dict:
    """Runs summary -> flashcards -> quiz and stores a small memory record."""
    out = {}
    out['summary'] = summarize_text(text)
    out['flashcards'] = generate_flashcards(out['summary'])
    out['quiz'] = generate_quiz_questions(out['summary'])
    # store a small memory record
    MEMORY.append({
        "timestamp": time.time(),
        "input_snippet": text[:300],
        "summary": out['summary']
    })
    return out


# --- Demo run (replace sample_notes with your lecture notes) ---
sample_notes = """
Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water.
Key steps include light absorption, splitting water, electron transport, and the Calvin cycle.
"""

demo_out = run_study_helper_pipeline(sample_notes)
print("Summary:\n", demo_out['summary'][:2000], "\n\n")
print("Flashcards (first 3):", demo_out['flashcards'][:3], "\n\n")
print("Quiz (first 3):", demo_out['quiz'][:3], "\n\n")
print("Memory (last):", MEMORY[-1])


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import os
print(os.listdir("/kaggle/input/photosynthesis-study-notes"))


file_path = "/kaggle/input/photosynthesis-study-notes/photosynthesis_notes.txt"

with open(file_path, "r") as f:
    notes = f.read()

print(notes[:500])


output = run_study_helper_pipeline(notes)
output


# Save full pipeline output and flashcards/quiz to files
import json
import csv
from pathlib import Path

out = run_study_helper_pipeline(notes)   # keep using your `notes` variable from earlier
out_path = Path("/kaggle/working/study_helper_output.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("Saved:", out_path)

# Save flashcards as CSV
cards = out.get("flashcards", [])
csv_path = Path("/kaggle/working/flashcards.csv")
if cards:
    keys = ["q","a"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for c in cards:
            writer.writerow({k: c.get(k, "") for k in keys})
    print("Flashcards CSV saved:", csv_path)
else:
    print("No flashcards to save.")


import os
print(os.listdir("/kaggle/input"))


print("### STUDY HELPER AGENT — FINAL OUTPUT ###")

print("\n--- Summary ---\n")
print(output['summary'])

print("\n--- Flashcards ---\n")
for card in output['flashcards']:
    print(card)

print("\n--- Quiz Questions ---\n")
for q in output['quiz']:
    print(q)

print("\n--- Memory Records (Last 3) ---\n")
for m in MEMORY[-3:]:
    print(m)


# --- Load the uploaded file and run the pipeline ---
file_path = "/kaggle/input/photosynthesis-study-notes/photosynthesis_notes.txt"

with open(file_path, "r") as f:
    notes = f.read()

# preview (optional)
print("Preview (first 500 chars):\n", notes[:500], "\n\n---\n")

# run the pipeline on the file contents
output = run_study_helper_pipeline(notes)

# show results nicely
print("Summary:\n", output.get("summary", "")[:2000], "\n\n")
print("Flashcards (first 3):", output.get("flashcards", [])[:3], "\n\n")
print("Quiz (first 3):", output.get("quiz", [])[:3], "\n\n")


from ipywidgets import Textarea, Button, Output, VBox

# UI elements
text_input = Textarea(
    value="",
    placeholder="Paste your study notes here...",
    description="Notes:",
    layout={'width': '100%', 'height': '200px'}
)

run_button = Button(
    description="Generate Summary / Flashcards / Quiz",
    button_style='success',
    layout={'width': '300px'}
)

ui_output = Output()

def on_run_button_clicked(b):
    with ui_output:
        ui_output.clear_output()
        notes = text_input.value
        
        if not notes.strip():
            print("⚠️ Please paste some notes before generating.")
            return
        
        result = run_study_helper_pipeline(notes)
        
        print("Summary:\n", result["summary"], "\n")
        print("Flashcards:\n", result["flashcards"], "\n")
        print("Quiz:\n", result["quiz"], "\n")

run_button.on_click(on_run_button_clicked)

display(VBox([text_input, run_button, ui_output]))


file_path = "/kaggle/input/photosynthesis-study-notes/photosynthesis_notes.txt"
with open(file_path,"r") as f:
    notes = f.read()
print("Preview:", notes[:300])
out = run_study_helper_pipeline(notes)
print(out['summary'])

