# ONE CELL → RUN THIS → DOWNLOAD → SUBMIT TO YOUR COURSE
import google.generativeai as genai
from IPython.display import display, Markdown, HTML
import os

# === ONE-TIME KEY (just paste when asked) ===
if not os.getenv("GOOGLE_API_KEY"):
    key = input("Paste your Gemini API key → ")
    os.environ["GOOGLE_API_KEY"] = key.strip()
    print("Key saved!")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# === YOUR PROFILE (change this to yours) ===
profile = "16 years old, India, 97% in 10th, love AI & robotics, no research yet"
dreams  = "Stanford CS, MIT, Caltech"

prompt = f"""
You are UniMentor-AI – 7 elite admissions experts powered by Gemini 2.5 Flash.
Student: {profile}
Dream: {dreams}

Return exactly:

UNI-MENTOR-AI REPORT
Profile Strength: ?/10 | Chance: ?%

1. BRUTALLY HONEST ANALYSIS (150 words)

2. 6-MONTH ROADMAP
Month 1 → ...
...
Month 6 → ...

3. 3 KILLER PROJECTS
• Title → Why Stanford loves it

4. 3 ESSAY TOPICS + HOOKS
• Topic → "Opening line"

5. 3 ACTIONS THIS WEEK
"""

response = model.generate_content(prompt, generation_config={"temperature":0.8, "max_output_tokens":4096})

print("\n" + "═"*80)
print("UNI-MENTOR-AI FINAL CAPSTONE REPORT (Gemini 2.5 Flash)".center(80))
print("═"*80 + "\n")
display(Markdown(response.text))

# AUTO-DOWNLOAD BUTTON (click it → saves the notebook with your report forever)
display(HTML('''
<br><br>
<button onclick="google.colab.kernel.invokeFunction('notebook.download')" 
        style="background:#1a73e8;color:white;padding:15px 30px;font-size:18px;border:none;border-radius:8px;cursor:pointer">
DOWNLOAD THIS NOTEBOOK NOW (Your Complete Capstone!)
</button>
<script>
function downloadNotebook(){google.colab.notebook.download()}
IPython.notebook.kernel.invokeFunction = downloadNotebook;
</script>
'''))

