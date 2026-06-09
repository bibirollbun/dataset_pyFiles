# -------------------------------
# Gemini Hackathon Notebook
# -------------------------------

import pandas as pd
import time
from tqdm.auto import tqdm
import google.generativeai as genai

# -------------------------------
# Step 0: Configure API
# -------------------------------
API_KEY = "AIzaSyC1OBO6Jj31RT401zb_EN5aokX-3qqYm7U"

if not API_KEY or "YOUR_API_KEY_HERE" in API_KEY:
    raise ValueError(
        "⚠️ API key is missing or still the placeholder! "
        "Please paste your real API key in API_KEY before running the notebook."
    )

genai.configure(api_key=API_KEY)
print("✅ API key configured successfully!")

# -------------------------------
# Step 1: Load Test Data
# -------------------------------
df = pd.read_csv("/kaggle/input/test-csv/test.csv")  # adjust path if needed
df.head()

# -------------------------------
# Step 2: Configure Model
# -------------------------------
model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",  # or "models/gemini-3-pro-preview"
    generation_config={
        "temperature": 0.1,
        "top_p": 0.85,
        "top_k": 30,
        "max_output_tokens": 2048,
        "candidate_count": 1,
        "response_mime_type": "text/plain"
    }
)

# -------------------------------
# Step 3: Task Classification
# -------------------------------
def classify_task(prompt):
    prompt_lower = prompt.lower()
    if any(x in prompt_lower for x in ["explain", "define", "what is"]):
        return "definition"
    if any(x in prompt_lower for x in ["write", "compose", "generate", "story", "poem"]):
        return "generation"
    if any(x in prompt_lower for x in ["summarize", "shorten", "condense"]):
        return "summary"
    if any(x in prompt_lower for x in ["list", "enumerate"]):
        return "list"
    return "generic"

TEMPLATES = {
    "definition": "Provide a precise and concise explanation.",
    "generation": "Create an original, coherent response that fits the instruction.",
    "summary": "Summarize with clarity and compact wording.",
    "list": "Return a clean, logically ordered list.",
    "generic": "Respond clearly and accurately."
}

# -------------------------------
# Step 4: Helper Functions
# -------------------------------
def self_refine(response_text):
    refine_prompt = f"""
    You are improving a generated answer. Original: {response_text}
    Improve it by:
    - Removing noise and unnecessary words.
    - Increasing clarity.
    - Making tone neutral and consistent.
    - Ensuring it directly answers the task.
    - Eliminating repetition.
    Return only the improved text.
    """
    try:
        return model.generate_content(refine_prompt).text.strip()
    except:
        return response_text

def chunk_prompt(prompt, max_len=250):
    words = prompt.split()
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len)]

def semantic_rewrite(prompt, output):
    boost = f"""
    Task: {prompt}
    Generated Output: {output}
    Rewrite the output for:
    - Higher semantic accuracy
    - Cleaner structure
    - Precise alignment with the task
    Return only the rewritten output.
    """
    try:
        return model.generate_content(boost).text.strip()
    except:
        return output

# -------------------------------
# Step 5: Final Answer with Auto-Retry
# -------------------------------
def generate_final_answer(prompt, retries=3, delay=60):
    for attempt in range(retries):
        try:
            task_type = classify_task(prompt)
            task_instruction = TEMPLATES[task_type]
            chunks = chunk_prompt(prompt)
            reasoning_text = ""
            for c in chunks:
                reasoning_text += model.generate_content(f"Process this chunk: {c}").text + " "
            base_prompt = f"""
            System rules:
            - Avoid markdown, bullets, or emojis unless needed.
            - No self-reference.
            - Match tone of few-shot examples: neutral, compact, precise.
            - No hallucinations.
            Task instruction: {task_instruction}
            User prompt: {prompt}
            Processed reasoning: {reasoning_text}
            Generate the best final answer.
            """
            first_pass = model.generate_content(base_prompt).text.strip()
            refined = self_refine(first_pass)
            boosted = semantic_rewrite(prompt, refined)
            return boosted.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"⚠️ Quota exceeded, retrying in {delay}s (Attempt {attempt+1}/{retries})...")
                time.sleep(delay)
            else:
                print(f"⚠️ Error generating output: {e}. Using placeholder.")
                return "Test output"
    # After retries, fallback
    return "Test output"

# -------------------------------
# Step 6: Run Predictions
# -------------------------------
preds = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    time.sleep(1)  # small delay to reduce hitting free tier limits
    preds.append(generate_final_answer(row["prompt"]))

# -------------------------------
# Step 7: Save Submission
# -------------------------------
sub = pd.DataFrame({
    "id": df["id"],
    "output": preds
})

sub.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ submission.csv ready! Check Output tab to download.")
sub.head()


