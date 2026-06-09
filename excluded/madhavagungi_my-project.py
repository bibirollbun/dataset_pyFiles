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


!pip install -U google-generativeai
!pip install -U pypdf
!pip install -U faiss-cpu
!pip install -U langchain
!pip install -U langchain-community

# Restart kernel after this cell runs if Kaggle asks



from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

# Load Kaggle secret
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Use supported model
model = genai.GenerativeModel("models/gemini-flash-latest")

print("✅ Gemini model loaded:", model.model_name)



import glob
from pypdf import PdfReader

# Find any uploaded PDF in Kaggle environment
pdf_files = glob.glob("/kaggle/input/**/*.pdf", recursive=True)

if not pdf_files:
    raise FileNotFoundError("❌ No PDF found. Please upload your resume (PDF).")

resume_path = pdf_files[0]   # Take first found pdf
print("✅ Found resume at:", resume_path)

# Read PDF
reader = PdfReader(resume_path)

resume_text = ""
for page in reader.pages:
    if page.extract_text():
        resume_text += page.extract_text()

print("\n✅ Resume successfully loaded\n")
print(resume_text[:700])  # preview first 700 characters



import concurrent.futures

# === AGENTS ===
def resume_reader_agent(text):
    prompt = f"""
    You are a Resume Reader Agent.
    Extract skills, education, experience, projects and certifications.

    Resume:
    {text}
    """
    return model.generate_content(prompt).text


def ats_scorer_agent(text):
    prompt = f"""
    You are an ATS system.
    Score the resume out of 100.
    Give strengths, weaknesses and missing keywords.

    Resume:
    {text}
    """
    return model.generate_content(prompt).text


def resume_rewriter_agent(text):
    prompt = f"""
    Rewrite this resume to be highly ATS-optimized,
    professional and powerful. Do not change facts.

    Resume:
    {text}
    """
    return model.generate_content(prompt).text


def job_matcher_agent(text):
    prompt = f"""
    Suggest the TOP 5 job roles best suited for this resume.
    Explain why.

    Resume:
    {text}
    """
    return model.generate_content(prompt).text


def hiring_manager_agent(text):
    prompt = f"""
    You are a Senior Hiring Manager.

    Give:
    - Final decision (Shortlist / Consider / Reject)
    - Improvement tips
    - Salary range
    - Feedback

    Resume:
    {text}
    """
    return model.generate_content(prompt).text


# === PARALLEL EXECUTION ===
def run_all_agents_parallel(resume_text):

    with concurrent.futures.ThreadPoolExecutor() as executor:

        future_reader = executor.submit(resume_reader_agent, resume_text)
        future_ats = executor.submit(ats_scorer_agent, resume_text)
        future_rewrite = executor.submit(resume_rewriter_agent, resume_text)
        future_match = executor.submit(job_matcher_agent, resume_text)
        future_hiring = executor.submit(hiring_manager_agent, resume_text)

        return {
            "Resume Analysis": future_reader.result(),
            "ATS Score": future_ats.result(),
            "Improved Resume": future_rewrite.result(),
            "Job Matches": future_match.result(),
            "Hiring Manager Decision": future_hiring.result()
        }


# === RUN ALL AGENTS ===
results = run_all_agents_parallel(resume_text)

for key, value in results.items():
    print(f"\n{'='*15} {key} {'='*15}\n")
    print(value)



import json

with open("Smart_Resume_Auditor_Results.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Results saved as Smart_Resume_Auditor_Results.json")





