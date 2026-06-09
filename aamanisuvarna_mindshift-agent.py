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


# MindShift Agent â€” Simple Local Demo (No API keys)
#This notebook demonstrates a small multi-agent pipeline:
#Creator â†’ Refiner â†’ Guardrail â†’ Formatter.
#It runs fully locally using a small model (no external API keys needed).



# Install required packages
!pip install --quiet transformers sentence-transformers scikit-learn



# 1 - Imports and helper utilities
import os, json, uuid, datetime
from typing import List, Dict, Any

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

# Make folders
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)



# 2 - Load a small local model for generation
from transformers import pipeline, set_seed

MODEL_NAME = "distilgpt2"  # small and Kaggle-friendly
generator = pipeline("text-generation", model=MODEL_NAME)
set_seed(42)  # reproducible outputs



# 3 - Simple wrapper to ask our local model for a reply
def ask_agent(prompt: str, max_new_tokens: int = 60) -> str:
    # The generator returns a list of outputs; we take the text field
    out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.8)
    text = out[0]["generated_text"]
    # The model sometimes repeats the prompt; remove prompt prefix if present
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text.strip()



# 4 - Memory and simple logger
MEMORY_PATH = "data/memory.json"
if not os.path.exists(MEMORY_PATH):
    json.dump({"users": {"demo_user": {"summary":"wants growth", "tone_pref":"empowering", "goals":["confidence"], "ratings":[]}}}, open(MEMORY_PATH,"w"), indent=2)

def load_memory():
    return json.load(open(MEMORY_PATH))

def save_memory(mem):
    json.dump(mem, open(MEMORY_PATH,"w"), indent=2)

def write_log(entry: Dict[str,Any]):
    trace = entry.get("trace_id", str(uuid.uuid4()))
    fname = f"logs/{trace}.json"
    open(fname,"w").write(json.dumps(entry, indent=2))
    return fname



# 5 - Creator Agent
class CreatorAgent:
    def generate_candidates(self, theme: str, user_ctx: str, n: int = 4) -> List[Dict[str,str]]:
        prompt = (f"Write {n} short inspirational quote ideas (<= 25 words) on the theme: {theme}.\n"
                  f"Keep the style uplifting and present tense. Output each quote on a new line.")
        raw = ask_agent(prompt, max_new_tokens=80)
        # Split into lines and clean
        lines = [l.strip(" \"-") for l in raw.split("\n") if l.strip()]
        candidates = []
        for i in range(min(n, len(lines))):
            candidates.append({"quote": lines[i], "rationale": "Auto-generated rationale.", "tone":"empowering"})
        # fallback if model gave fewer lines
        while len(candidates) < n:
            candidates.append({"quote": f"Choose to grow today. #{len(candidates)+1}", "rationale":"Simple catalyst", "tone":"empowering"})
        return candidates

# quick test
creator = CreatorAgent()
print("Sample candidates:\n", creator.generate_candidates("self-growth", "demo user context"))



# 6 - Refiner Agent
class RefinerAgent:
    def refine(self, quote_obj: Dict[str,str], user_ctx: str) -> Dict[str,str]:
        q = quote_obj["quote"]
        prompt = (f"Edit the short quote to make it clearer and more moving, keep <= 20 words.\n"
                  f"Preserve the meaning: \"{q}\"")
        out = ask_agent(prompt, max_new_tokens=40)
        # Sometimes the model returns multiple sentences; keep first sentence
        refined = out.split("\n")[0].strip()
        return {"quote": refined if refined else q, "rationale": "Refined to increase clarity and emotion.", "tone": quote_obj.get("tone","empowering")}

# quick test
refiner = RefinerAgent()
print(refiner.refine({"quote":"You can change your life by choosing new thoughts."}, ""))



# 7 - Guardrail Agent (very simple heuristic checks)
class GuardrailAgent:
    def safety_check(self, refined_obj: Dict[str,str]) -> Dict[str,Any]:
        q = refined_obj["quote"].lower()
        issues = []
        # Simple checks for medical claims or guarantees
        bad_words = ["cure","diagnose","guarantee","heal instantly","medical"]
        for w in bad_words:
            if w in q:
                issues.append(f"Contains risky word: {w}")
        # If issues, mark not ok
        ok = len(issues) == 0
        return {"ok": ok, "issues": issues, "notes":"heuristic checks"}

guard = GuardrailAgent()
print(guard.safety_check({"quote":"This will cure all your problems"}))



# 8 - Formatter Agent
class FormatAgent:
    def format_package(self, final_obj: Dict[str,str]) -> Dict[str,str]:
        q = final_obj["quote"]
        rationale = final_obj.get("rationale","")
        return {
            "short_quote": q,
            "instagram_caption": f"{q}\n\n{rationale}\n\n#mindshift #dailyquote",
            "tweet": q if len(q) <= 280 else q[:277]+"...",
            "journal_prompt": f"What small step today aligns with: '{q}'?"
        }

formatter = FormatAgent()
print(formatter.format_package({"quote":"A tiny choice becomes a new you.", "rationale":"Encourages small consistent action."}))



# 9 - Evaluation (simple heuristics)
class EvaluationAgent:
    def automated_metrics(self, text: str) -> Dict[str,float]:
        words = text.split()
        positivity = 0.7 + 0.01 * max(0, 10 - len(words))  # tiny heuristic
        clarity = 0.8 if len(words) <= 20 else 0.6
        resonance = (positivity + clarity) / 2
        return {"positivity": round(min(1.0, positivity),2), "clarity": round(clarity,2), "resonance": round(resonance,2)}

evaluator = EvaluationAgent()
print(evaluator.automated_metrics("Choose to grow through small choices today."))



# 10 - Coordinator that runs all agents
class Coordinator:
    def __init__(self):
        self.creator = CreatorAgent()
        self.refiner = RefinerAgent()
        self.guard = GuardrailAgent()
        self.formatter = FormatAgent()
        self.evaluator = EvaluationAgent()
        self.memory = load_memory()

    def run_once(self, user_id="demo_user", theme="self-growth"):
        trace_id = str(uuid.uuid4())
        user = self.memory["users"].get(user_id, {})
        user_ctx = f"{user.get('summary','')} Goals: {', '.join(user.get('goals',[]))}"
        # 1. create candidates
        candidates = self.creator.generate_candidates(theme, user_ctx, n=4)
        # 2. refine each candidate and check safety
        refined_bucket = []
        for c in candidates:
            refined = self.refiner.refine(c, user_ctx)
            verdict = self.guard.safety_check(refined)
            refined_bucket.append({"original":c, "refined":refined, "verdict":verdict})
        # 3. choose first that passes guardrail
        chosen = None
        for item in refined_bucket:
            if item["verdict"]["ok"]:
                chosen = item["refined"]
                break
        if not chosen:
            chosen = refined_bucket[0]["refined"]
        # 4. format & evaluate
        packaged = self.formatter.format_package(chosen)
        metrics = self.evaluator.automated_metrics(chosen["quote"])
        # 5. log and save simple rating placeholder
        log = {
            "trace_id": trace_id,
            "timestamp": now_iso(),
            "theme": theme,
            "user_context": user_ctx,
            "candidates": candidates,
            "refined_bucket": refined_bucket,
            "chosen": chosen,
            "packaged": packaged,
            "metrics": metrics
        }
        write_log(log)
        # append rating placeholder to memory
        r = {"id": str(uuid.uuid4()), "quote": chosen["quote"], "score": 0, "timestamp": now_iso()}
        self.memory["users"].setdefault(user_id, {"summary":"demo", "goals":[],"ratings":[]})
        self.memory["users"][user_id]["ratings"].append(r)
        save_memory(self.memory)
        return {"trace_id":trace_id, "chosen":chosen, "packaged":packaged, "metrics":metrics}

# demo run
coord = Coordinator()
result = coord.run_once(theme="breaking limiting beliefs")
print("Quote:", result["chosen"]["quote"])
print("Instagram caption:\n", result["packaged"]["instagram_caption"])
print("Metrics:", result["metrics"])



# 11 - Pretty print
print("\nğŸŒŸ MindShift Quote of the Day ğŸŒŸ\n")
print("â€œ" + result["chosen"]["quote"] + "â€�\n")
print("Tone:", result["chosen"].get("tone","empowering"))
print("\nRationale:", result["chosen"].get("rationale",""))
print("\nInstagram caption:\n", result["packaged"]["instagram_caption"])



# 12 - Rate the quote (type a number and run this cell)
rating = 5  # change to 1-5
mem = load_memory()
# update last rating entry
mem["users"]["demo_user"]["ratings"][-1]["score"] = rating
save_memory(mem)
print("Saved rating:", rating)



# 13 - Run multiple themes quickly
themes = ["gratitude", "focus", "discipline", "self-love"]
for t in themes:
    out = coord.run_once(theme=t)
    print(f"\n--- Theme: {t} ---")
    print(out["chosen"]["quote"])
    print("Metrics:", out["metrics"])



from transformers import pipeline
import os

# Force CPU (ignore GPU warnings)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 1ï¸�âƒ£ Creator Agent - generates raw quote
creator_agent = pipeline("text-generation", model="distilgpt2", max_new_tokens=50)

# 2ï¸�âƒ£ Refiner Agent - polishes quote
refiner_agent = pipeline("text-generation", model="distilgpt2", max_new_tokens=50)

# 3ï¸�âƒ£ Guardrail Agent - simple heuristic for demo
class GuardrailAgent:
    def check(self, quote):
        risky_words = ["cure", "guarantee", "heal instantly"]
        issues = [w for w in risky_words if w in quote.lower()]
        return {"ok": len(issues)==0, "issues": issues, "notes": "heuristic checks"}

guardrail_agent = GuardrailAgent()

# 4ï¸�âƒ£ Formatter Agent - adds hashtags / formatting
class FormatterAgent:
    def format_quote(self, quote, theme="motivation"):
        return f"{quote}\n#{theme} #dailyquote #mindshift"

formatter_agent = FormatterAgent()

# 5ï¸�âƒ£ Evaluator Agent - simple placeholder metrics
class EvaluatorAgent:
    def evaluate(self, quote):
        return {"positivity": 0.7, "clarity": 0.7, "resonance": 0.7}

evaluator_agent = EvaluatorAgent()



def run_mindshift_pipeline(theme, num_quotes=3):
    results = []

    for i in range(num_quotes):
        # 1ï¸�âƒ£ Creator generates raw quote
        prompt = f"Write a short motivational quote about {theme} in an inspiring tone:"
        raw = creator_agent(prompt, max_new_tokens=50, num_return_sequences=1)[0]['generated_text']
        print(f"\nRaw quote {i+1}:", raw)

        # 2ï¸�âƒ£ Refiner polishes the quote
        refined_prompt = f"Refine this quote to make it more uplifting and clear: {raw}"
        refined = refiner_agent(refined_prompt, max_new_tokens=50, num_return_sequences=1)[0]['generated_text']
        print(f"Refined quote {i+1}:", refined)

        # 3ï¸�âƒ£ Guardrail (optional: can comment for testing)
        guard_result = guardrail_agent.check(refined)
        if not guard_result['ok']:
            print(f"Guardrail flagged: {guard_result['issues']}")
            continue  # skip unsafe quotes

        # 4ï¸�âƒ£ Formatter
        formatted = formatter_agent.format_quote(refined, theme)

        # 5ï¸�âƒ£ Metrics
        metrics = evaluator_agent.evaluate(refined)

        results.append({
            "theme": theme,
            "quote": refined,
            "formatted": formatted,
            "metrics": metrics
        })

    return results



themes = ["gratitude", "focus", "discipline", "self-love"]
all_quotes = []

for theme in themes:
    print(f"\n--- Generating quotes for theme: {theme} ---")
    quotes = run_mindshift_pipeline(theme, num_quotes=5)
    all_quotes.extend(quotes)



import pandas as pd

# Display results
for q in all_quotes:
    print(f"\nTheme: {q['theme']}")
    print(f"Quote: {q['quote']}")
    print(f"Formatted: {q['formatted']}")
    print(f"Metrics: {q['metrics']}")

# Save to CSV and JSON
df = pd.DataFrame(all_quotes)
df.to_csv("mindshift_quotes.csv", index=False)
df.to_json("mindshift_quotes.json", orient="records")
print("\nâœ… Saved quotes to CSV and JSON.")



## The mini App to run the Agent
# Interactive demo
def interactive_mindshift():
    print("Welcome to MindShift Agent! Type a theme to get a motivational quote (or 'exit' to quit).")
    
    while True:
        theme = input("\nEnter theme: ").strip()
        if theme.lower() == "exit":
            print("Goodbye! Stay inspired âœ¨")
            break
        
        quotes = run_mindshift_pipeline(theme, num_quotes=1)
        if len(quotes) == 0:
            print("Oops! Could not generate a safe quote. Try another theme.")
            continue
        
        q = quotes[0]
        print("\n--- Your MindShift Quote ---")
        print(f"Theme: {q['theme']}")
        print(f"Quote: {q['quote']}")
        print(f"Formatted: {q['formatted']}")
        print(f"Metrics: {q['metrics']}")
        print("----------------------------")

# Run the interactive demo
interactive_mindshift()


