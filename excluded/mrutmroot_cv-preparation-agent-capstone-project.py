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


pip install google-adk


!pip install pymupdf


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import logging
import os

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ðŸ§¹ Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("âœ… Logging configured")


# ================================
# 1. Setup & Imports
# ================================
import re
import json
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import fitz  # PyMuPDF for PDF CV parsing
import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)
MODEL = genai.GenerativeModel(model_name='models/gemini-2.5-flash-lite')


# ================================
# 2. Memory + Session
# ================================

class MemoryBank:
    """Long-term user memory (persisted as JSON)."""

    def __init__(self, path="memory_bank.json"):
        self.path = path
        try:
            with open(path, "r") as f:
                self.data = json.load(f)
        except:
            self.data = {}

    def save_preference(self, key, value):
        self.data[key] = value
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key, default=None):
        return self.data.get(key, default)


class SessionState:
    """In-memory session for a single optimization run."""

    def __init__(self):
        self.job_posting_text = None
        self.resume_text = None
        self.keywords = []
        self.results = {}

session = SessionState()
memory = MemoryBank()


# ================================
# 3. Custom Tools
# ================================
# 3.1 CV Parser Tool (PDF/TXT â†’ Text)
# ================================
class ResumeParserTool:
    """Convert PDF or plaintext CV into raw text."""
    
    @staticmethod
    def parse_pdf(path: str) -> str:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    @staticmethod
    def parse_text(text: str) -> str:
        return text


# ================================
# 3.2 Keyword Extractor Tool (via Gemini)
# ================================

def extract_keywords_with_gemini(job_text: str) -> List[str]:
    """Extract simple ATS keywords from job posting."""
    prompt = f"""
    Extract ONLY a Python list of keywords from the job posting.
    Do NOT include explanations.
    Output example: ["python", "sql", "docker"]

    Job posting:
    {job_text}
    """

    response = MODEL.generate_content(prompt).text.strip()

    # Cleanup unsafe characters
    clean = response.replace("\n", "").replace("'", '"')

    try:
        keywords = json.loads(clean)
        if isinstance(keywords, list):
            return [kw.lower().strip() for kw in keywords]
        else:
            return []
    except:
        # Fallback: split by commas (robust mode)
        return [k.strip().lower() for k in re.split(r"[,\n]", response) if len(k.strip()) > 1]


# ================================
# 4. Agents
# ================================
# 4.1 JobParserAgent
# ================================
class JobParserAgent:
    """Extracts job requirements + keywords from the posting."""
    @staticmethod
    def run(job_text: str):
        keywords = extract_keywords_with_gemini(job_text)

        summary = MODEL.generate_content(
            "Summarize the job posting in 5 lines:\n\n" + job_text
        ).text

        return {
            "summary": summary,
            "keywords": keywords,
            "keywords_count": len(keywords)
        }


# ================================
# 4.2 SkillMatcherAgent
# ================================
class SkillMatcherAgent:
    """Matches job keywords with CV text."""
    @staticmethod
    def run(resume_text: str, keywords: List[str]):
        if not isinstance(keywords, list):
            keywords = []

        resume_lower = resume_text.lower()

        matched = [kw for kw in keywords if kw in resume_lower]
        missing = [kw for kw in keywords if kw not in resume_lower]

        score = int((len(matched) / max(len(keywords), 1)) * 100)

        return {
            "score": score,
            "matched": matched,
            "missing": missing
        }


# ================================
# 4.3 CV ImproverAgent (Gemini rewriting)
# ================================
class CVImproverAgent:
    """Improves bullet points by adding missing keywords."""
    @staticmethod
    def rewrite(resume_text: str, missing_keywords: List[str]):
        prompt = f"""
        You are an ATS optimization engine.

        TASK:
        - Improve the following CV text.
        - Integrate missing keywords naturally and professionally.
        - Keep structure and clarity.
        - DO NOT invent fake jobs.
        - Add impactful bullet points.

        Missing keywords:
        {missing_keywords}

        Resume text:
        {resume_text}

        Return ONLY the improved resume.
        """

        rewritten = MODEL.generate_content(prompt).text
        return rewritten


# ================================
# 4.4 CoordinatorAgent (Pipeline)
# ================================
class CoordinatorAgent:
    """Controls overall pipeline."""
    @staticmethod
    def run(job_text: str, cv_text: str):
        job_info = JobParserAgent.run(job_text)

        keywords = job_info.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []

        match_info = SkillMatcherAgent.run(cv_text, keywords)

        missing = match_info.get("missing", [])
        
        improved_cv = CVImproverAgent.rewrite(cv_text, missing)

        output = {
            "job_info": job_info,
            "matching": match_info,
            "optimized_cv": improved_cv
        }

        session.results = output
        return output


# ================================
# 5. Input: Load Job Posting + CV
# ================================
# Paste job posting
#######
session.job_posting = """
We are hiring a Senior Backend Developer with Python, APIs, SQL, Docker, 
cloud deployment, monitoring, and CI/CD experience. Knowledge of scalable 
distributed systems is required. Python and REST APIs appear multiple times.
"""

session.cv_text = """
Jan Nowicki â€“ Backend Developer

Experience:
- Developed REST APIs
- Maintained SQL databases
- Built automation scripts
"""
# session.cv_text = ResumeParserTool.from_pdf("/kaggle/input/cv.pdf")



# ================================
# 6. Run Full Pipeline
# ================================
results = CoordinatorAgent.run(session.job_posting, session.cv_text)
results


# ================================
# 7. Display Results in Notebook
# ================================
# 7.1 Matching Score
# ================================
print("MATCHING SCORE:", results["matching"]["score"], "/ 100")


# ================================
# 7.2 Missing Keywords
# ================================
#print("Missing keywords:", results["matching"]["missing_keywords"])

results["matching"]["missing"]


# ================================
# 7.3 Improved CV (Preview)
# ================================
print("\n\n==== IMPROVED CV ====\n")
print(results["optimized_cv"])


# ================================
# 7.4 Changelog
# ================================
#print("\n\n==== CHANGES MADE ====\n")
#for c in results["changes"]:
#    print("-", c)



# ================================
# 8. Export Results to Files
# ================================
with open("optimized_cv.txt", "w") as f:
    f.write(results["optimized_cv"])

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved: optimized_cv.txt, results.json")

