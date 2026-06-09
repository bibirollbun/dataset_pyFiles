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


# quick: set the key for the current notebook session
import os
os.environ["GEMINI_API_KEY"] = "your_real_api_key_here"
print("GEMINI_API_KEY set for this session.")



resume_content = """
Snehal Jakhale
AI & Machine Learning Enthusiast
Email: snehal@example.com | Phone: 9876543210

SUMMARY
Motivated AI/ML student with strong foundations in Python, Machine Learning, Deep Learning,
Data Preprocessing, and model deployment. Experience with TensorFlow, PyTorch, LangChain,
vector databases, and RAG-based LLM systems.

SKILLS
- Python, NumPy, Pandas, Matplotlib
- Machine Learning: Regression, Classification, Clustering
- Deep Learning: CNN, RNN, NLP, Transformers
- Tools: TensorFlow, PyTorch, LangChain, LlamaIndex
- Version Control: Git, GitHub

PROJECTS
1. AI-Powered Diet Plan Generator  
   Built a personalized Indian diet plan recommender using BMI, BMR, TDEE, and food preferences.
   Tech: Python, Pandas, Gradio.

2. ADmyBRAND Insights Dashboard  
   Built a Next.js dashboard with analytics, charts, dark/light mode, and reusable components.

3. Cyclone Prediction Model  
   Developed an ML pipeline using meteorological satellite data (MOSDAC) with preprocessing
   and random forest–based prediction.

EDUCATION
B.Tech in Artificial Intelligence, 2021–2025
"""

with open("resume.txt", "w") as f:
    f.write(resume_content.strip())

print("resume.txt created!")



job_description_content = """
Position: Machine Learning Engineer – Entry Level

We are looking for a Machine Learning Engineer with strong understanding of
ML fundamentals, data preprocessing, Python, and model training.

Responsibilities:
- Build and train ML models using Python.
- Work with deep learning architectures like CNN or Transformers.
- Preprocess and clean large datasets.
- Collaborate with the AI research and engineering team.
- Deploy models in production environments.

Required Skills:
- Strong Python programming skills.
- Knowledge of ML algorithms (linear regression, SVM, decision trees).
- Experience with TensorFlow or PyTorch.
- Understanding of NLP, embeddings, and vector databases is a plus.
- Familiarity with Git and version control.

Preferred Skills:
- Experience with LLMs, LangChain, or RAG systems.
"""

with open("job_description.txt", "w") as f:
    f.write(job_description_content.strip())

print("job_description.txt created!")



"""
ai_job_assistant_kaggle.py

- Works in Kaggle notebooks (auto-loads secrets) and CLI.
- Loads resume.txt and job_description.txt, calls Gemini via google-genai / google-generativeai SDK,
  and writes output.json containing: match_score, suggestions, cover_letter.

Instructions:
1) In Kaggle: store your API key under Settings -> Secrets with a name like "GOOGLE_API_KEY".
2) Put resume.txt and job_description.txt in the working directory, or provide paths when prompted.
3) Run this cell / script.

"""

import os
import sys
import json
import time
import getpass
from typing import Dict, Optional

# Name of the env var the rest of the code expects
API_KEY_ENV = "GEMINI_API_KEY"

# Candidate model names (edit if needed)
DEFAULT_MODEL_NAMES = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-prob",
    "gemini-2.0-pro",
]


# ---------------- Utilities ----------------
def load_text_file(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_json(data: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------- GenAI wrapper ---------------
class GenAIWrapper:
    """
    Compatibility wrapper for older `google.generativeai` and newer `google.genai`.
    Use generate_text(prompt, model, max_output_tokens) -> str
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._sdk_name: Optional[str] = None
        self._genai = None
        self._client = None

        # Try older package
        try:
            import google.generativeai as genai  # type: ignore
            self._sdk_name = "google.generativeai"
            self._genai = genai
            try:
                if hasattr(genai, "configure"):
                    genai.configure(api_key=self.api_key)
            except Exception:
                pass
            return
        except Exception:
            pass

        # Try newer official package
        try:
            from google import genai as genai_client  # type: ignore
            self._sdk_name = "google.genai"
            self._genai = genai_client
            try:
                if hasattr(genai_client, "Client"):
                    self._client = genai_client.Client(api_key=self.api_key)
            except Exception:
                self._client = None
            return
        except Exception:
            pass

        raise ImportError(
            "No supported Google GenAI SDK found. Install `google-generativeai` or `google-genai`."
        )

    def generate_text(self, prompt: str, model: str, max_output_tokens: int = 1024) -> str:
        if not prompt:
            raise ValueError("Empty prompt")
        if not model:
            raise ValueError("Empty model")

        if self._sdk_name == "google.generativeai":
            genai = self._genai
            try:
                # Prefer GenerativeModel if available
                if hasattr(genai, "GenerativeModel"):
                    try:
                        model_obj = genai.GenerativeModel(model_name=model)
                        resp = model_obj.generate_content(prompt)
                        text = self._extract_text_from_old_sdk_response(resp)
                        if text:
                            return text
                    except Exception:
                        pass
                # Fallback top-level
                if hasattr(genai, "generate_content"):
                    resp = genai.generate_content(model=model, prompt=prompt)
                    text = self._extract_text_from_old_sdk_response(resp)
                    if text:
                        return text
                raise RuntimeError("google.generativeai did not return usable text.")
            except Exception as e:
                raise RuntimeError(f"Older SDK generation error: {e}")

        elif self._sdk_name == "google.genai":
            genai_client = self._genai
            client = self._client or (genai_client.Client(api_key=self.api_key) if hasattr(genai_client, "Client") else None)
            if client is None:
                raise RuntimeError("Could not create google.genai Client instance; check SDK version.")
            try:
                if hasattr(client, "responses") and hasattr(client.responses, "create"):
                    resp = client.responses.create(model=model, input=prompt, max_output_tokens=max_output_tokens)
                else:
                    resp = client.create(model=model, input=prompt, max_output_tokens=max_output_tokens)

                # Safe extraction: resp.output may be None
                text_out = ""
                out_attr = getattr(resp, "output", None)
                if out_attr:
                    try:
                        for part in out_attr:
                            content = getattr(part, "content", None) or (part.get("content") if isinstance(part, dict) else None)
                            if not content:
                                continue
                            for p in content:
                                if hasattr(p, "text"):
                                    text_out += p.text
                                elif isinstance(p, dict) and "text" in p:
                                    text_out += p["text"]
                                else:
                                    text_out += str(p)
                    except Exception:
                        # Fallback: serialize and search for 'text' keys
                        try:
                            j = json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
                            def find_text(obj):
                                s = ""
                                if isinstance(obj, dict):
                                    for k, v in obj.items():
                                        if k == "text" and isinstance(v, str):
                                            s += v
                                        else:
                                            s += find_text(v)
                                elif isinstance(obj, list):
                                    for item in obj:
                                        s += find_text(item)
                                return s
                            text_out = find_text(j)
                        except Exception:
                            text_out = ""

                if text_out and text_out.strip():
                    return text_out

                # Common fallbacks
                for fld in ["output_text", "text", "content"]:
                    val = getattr(resp, fld, None) or (resp.get(fld) if isinstance(resp, dict) else None)
                    if isinstance(val, str) and val.strip():
                        return val

                raise RuntimeError("Model returned empty output (resp.output is empty or None).")
            except Exception as e:
                raise RuntimeError(f"New SDK generation error: {e}")
        else:
            raise RuntimeError("Unsupported GenAI SDK configuration")

    @staticmethod
    def _extract_text_from_old_sdk_response(resp) -> str:
        if resp is None:
            return ""
        if hasattr(resp, "text"):
            return getattr(resp, "text") or ""
        if isinstance(resp, dict):
            if "candidates" in resp and isinstance(resp["candidates"], (list, tuple)) and resp["candidates"]:
                c0 = resp["candidates"][0]
                if isinstance(c0, dict) and "content" in c0:
                    return c0["content"] or ""
            for key in ["output", "result", "content"]:
                val = resp.get(key)
                if isinstance(val, str) and val.strip():
                    return val
            return str(resp)
        try:
            j = json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
            if isinstance(j, dict) and "text" in j and isinstance(j["text"], str):
                return j["text"]
        except Exception:
            pass
        return ""


# --------------- Prompt ---------------
PROMPT_TEMPLATE_JSON = """
You are an assistant that analyzes a candidate resume against a job description.
Return EXACTLY a JSON object with the following keys:
- match_score: integer between 0 and 100 (higher is better)
- suggestions: array of short strings (3-6 items) with concrete resume bullet improvements
- cover_letter: string, a personalized cover letter addressed to "Hiring Manager" (3-5 short paragraphs)

Resume:
{resume}

Job description:
{job}

Format: Respond ONLY with valid JSON (no explanation).
"""


# --------------- Runner ---------------
def run(resume_path: str, job_path: str, output_path: str = "output.json", model_hint: Optional[str] = None):
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(f"Please set {API_KEY_ENV} environment variable to your Gemini/Google API key")

    resume = load_text_file(resume_path)
    job = load_text_file(job_path)

    wrapper = GenAIWrapper(api_key=api_key)
    prompt = PROMPT_TEMPLATE_JSON.format(resume=resume, job=job)

    candidates = [model_hint] + DEFAULT_MODEL_NAMES if model_hint else DEFAULT_MODEL_NAMES
    # dedupe preserving order
    seen = set()
    models_to_try = []
    for m in candidates:
        if not m or m in seen:
            continue
        seen.add(m)
        models_to_try.append(m)

    last_exception = None
    for model_name in models_to_try:
        try:
            print(f"[INFO] Requesting model '{model_name}' ...")
            out_text = wrapper.generate_text(prompt=prompt, model=model_name, max_output_tokens=1024)
            if out_text is None:
                raise RuntimeError("Received None from wrapper.generate_text")
            out_text = out_text.strip()
            if not out_text:
                raise RuntimeError("Model returned empty string")

            # Try parse JSON directly
            try:
                parsed = json.loads(out_text)
                save_json(parsed, output_path)
                print(f"[OK] JSON parsed and saved to {output_path}")
                return parsed
            except json.JSONDecodeError:
                start = out_text.find("{")
                end = out_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    snippet = out_text[start:end+1]
                    try:
                        parsed = json.loads(snippet)
                        save_json(parsed, output_path)
                        print(f"[OK] Extracted JSON saved to {output_path}")
                        return parsed
                    except json.JSONDecodeError:
                        pass

            print(f"[WARN] Model '{model_name}' returned text but not valid JSON. Preview:")
            print(out_text[:1200])
        except Exception as e:
            last_exception = e
            print(f"[ERROR] Model '{model_name}' failed: {repr(e)}")
            time.sleep(0.4)

    raise RuntimeError(f"All model attempts failed. Last error: {last_exception}")


# ------------- Helpers for notebook-friendly arg parsing -------------
def _find_file_args_from_sysargv():
    """
    Return (resume_path, job_path, out_file, model_hint)
    Strategy:
      - Look through sys.argv and collect existing files (ignore flags like -f).
      - If none found, return (None, None, None, None) to allow interactive prompt.
    """
    args = sys.argv[1:]
    file_args = []
    model_hint = None
    out_file = None

    for i, a in enumerate(args):
        if not a or a.startswith("-"):
            if a.startswith("--model="):
                model_hint = a.split("=", 1)[1]
            elif a in ("--model", "-m") and i + 1 < len(args) and not args[i + 1].startswith("-"):
                model_hint = args[i + 1]
            continue
        if os.path.exists(os.path.expanduser(a)):
            file_args.append(os.path.expanduser(a))
        else:
            if len(file_args) >= 2 and out_file is None:
                out_file = a

    resume = file_args[0] if len(file_args) >= 1 else None
    job = file_args[1] if len(file_args) >= 2 else None
    return resume, job, out_file, model_hint


# ------------- Auto-load Kaggle secret utility -------------
def _load_kaggle_secret_to_env(secret_name_candidates=("GOOGLE_API_KEY", "GEMINI_API_KEY", "YOUR_API_KEY_GOES_HERE")):
    """
    If running in Kaggle and kaggle_secrets is available, load the first matching secret
    name into GEMINI_API_KEY env var and return True. Otherwise return False.
    """
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception:
        return False

    try:
        usc = UserSecretsClient()
        for secret_name in secret_name_candidates:
            try:
                val = usc.get_secret(secret_name)
                if val:
                    os.environ[API_KEY_ENV] = val
                    print(f"Loaded secret '{secret_name}' into {API_KEY_ENV}")
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


# ------------- Main entry (works in notebook) -------------
def main_interactive():
    try:
        # Try loading from Kaggle secrets automatically
        _load_kaggle_secret_to_env()

        # If no env var, prompt securely
        api_key = os.getenv(API_KEY_ENV)
        if not api_key:
            print("GEMINI API key not found in environment. You can paste it now (input hidden).")
            api_key = getpass.getpass("Enter GEMINI_API_KEY (input hidden): ").strip()
            if api_key:
                os.environ[API_KEY_ENV] = api_key
            else:
                raise EnvironmentError("No GEMINI_API_KEY provided. Exiting.")

        resume_path, job_path, out_file, model_hint = _find_file_args_from_sysargv()

        if not resume_path or not job_path:
            print("No valid resume/job file paths were detected on the command line.")
            resume_path = resume_path or input("Path to resume text file (e.g., resume.txt): ").strip() or "resume.txt"
            job_path = job_path or input("Path to job description file (e.g., job_description.txt): ").strip() or "job_description.txt"
            out_file = out_file or input("Output file name [output.json]: ").strip() or "output.json"
        else:
            out_file = out_file or "output.json"

        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume file not found: {resume_path}")
        if not os.path.exists(job_path):
            raise FileNotFoundError(f"Job description file not found: {job_path}")

        result = run(resume_path, job_path, out_file, model_hint=model_hint)
        print("Result preview:")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
        return result

    except Exception as e:
        print("Fatal error:", repr(e))
        print("\nTroubleshooting tips:")
        print(" 1) Ensure GEMINI_API_KEY is set in Kaggle Secrets (key name like GOOGLE_API_KEY) or paste it when prompted.")
        print(" 2) Install one of the SDKs in an earlier cell: `pip install google-generativeai` OR `pip install google-genai`")
        print(" 3) Ensure resume.txt and job_description.txt exist in the working directory (or give correct paths).")
        print(" 4) To force a specific model, add `--model model_name` in sys.argv or pass it in CLI.")
        return None


# If module executed directly, run main_interactive()
if __name__ == "__main__":
    main_interactive()





