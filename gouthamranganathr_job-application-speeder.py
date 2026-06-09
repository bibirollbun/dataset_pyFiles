import os
import google.generativeai as genai
from pypdf import PdfReader

# --- CONFIG ---
# careful with the key, don't commit it to github
# fetching from env or just paste it here for testing
os.environ["GOOGLE_API_KEY"] = "# Key hidden"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- TOOLS ---
def get_pdf_text(filepath):
    # simple helper to pull text out of the resume pdf
    # doing this so we don't have to copy-paste the resume every single time
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except:
        return "Error reading the PDF. Is the path right?"

# --- MAIN AGENT LOGIC ---
def run_agent():
    print("--- JOB APPLICATION SPEEDER ---")
    
    # check if the resume is actually there
    resume_path = "resume.pdf"
    if not os.path.exists(resume_path):
        print("Heads up: You need a file named 'resume.pdf' in this folder.")
        return

    # grabbing the job description from the user
    # using a loop here so it handles multi-line pastes correctly
    print("Paste the Job Description text below (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    job_text = "\n".join(lines)

    print("\n... Reading resume ...")
    resume_text = get_pdf_text(resume_path)

    # using flash because it's faster for this kind of stuff
    model = genai.GenerativeModel('gemini-1.5-flash')

    # STEP 1: Analysis Agent
    # We need to understand the job before writing. 
    # This agent acts like a recruiter extracting keywords.
    print("\n[1/2] Analyzing the job post...")
    
    analysis_prompt = f"""
    Role: Expert Technical Recruiter.
    Task: Extract the top 3 hard skills and 3 soft skills from the job description below.
    Also, identify the company culture/tone (e.g., formal, startup, serious).
    
    JOB DESCRIPTION:
    {job_text}
    """
    analysis = model.generate_content(analysis_prompt)
    
    # STEP 2: Writing Agent
    # taking the analysis from step 1 and combining it with the resume
    # this keeps the cover letter focused on what actually matters
    print("[2/2] Drafting the letter...")
    
    writer_prompt = f"""
    Role: Job Applicant.
    Task: Write a cover letter connecting my resume to the job analysis provided.
    
    Constraints:
    - Keep it under 200 words.
    - Match the tone identified in the analysis.
    - Don't sound generic.
    
    MY RESUME:
    {resume_text}
    
    JOB ANALYSIS:
    {analysis.text}
    """
    result = model.generate_content(writer_prompt)
    
    # Output results
    print("\n" + "="*30)
    print("FINAL DRAFT")
    print("="*30)
    print(result.text)
    
    # saving it to a file so I don't lose it
    with open("cover_letter.txt", "w") as f:
        f.write(result.text)
    print("\nSaved to cover_letter.txt")

if __name__ == "__main__":
    run_agent()# This Python 3 environment comes with many helpful analytics libraries installed
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

