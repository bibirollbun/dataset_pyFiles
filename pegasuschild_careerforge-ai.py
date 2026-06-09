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


# Install necessary libraries
# jobspy to scrap job listings

!pip install -q python-jobspy google-genai
print("âœ… Libraries installed.")


# Kaggle Authentication
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print("ğŸ”‘ Authentication Error: Please add 'GOOGLE_API_KEY' to your Kaggle secrets.")
    raise e


# Imports and Configuration

import re
import time
import json
import asyncio # For handling asynchronous agent runs
from datetime import datetime # For timestamps
from typing import Any, Dict, List
from IPython.display import display, HTML, Markdown


# Google ADK Imports
from google.genai import types
from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool
from google.adk.tools.tool_context import ToolContext

print("âœ… ADK components imported successfully.")


# Import jobspy
try:
    from jobspy import scrape_jobs
    print("\nâœ… SUCCESS! 'jobspy' has been successfully imported.")
except ModuleNotFoundError:
    print("\nâ�Œ FAILURE: Even after manually adding the path, 'jobspy' could not be found.")
    print("   This indicates a severe issue with the Colab environment.")
except ValueError as e:
    print(f"\nâ�Œ FAILURE: An error occurred during import, likely due to a library version conflict: {e}")
    print("   Consider restarting the runtime and ensuring consistent library versions.")


# Retry configuration
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

print("âœ… Retry configuration created.")



# Memory Setup

# ğŸ§  1. LONG-TERM MEMORY
HISTORY_FILE = "job_history_final.json"

def load_db():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(HISTORY_FILE, "w") as f:
        json.dump(db, f, indent=2)
        
# Runtime Memory
JOB_MEMORY: Dict[str, Dict[str, Any]] = {}
APPLICATION_MEMORY: Dict[str, Any] = {} # Holds the Final Output Packages

# ğŸ› ï¸� 2. TOOL DEFINITION

def get_job_details_tool(job_id: str) -> dict:
    """Retrieves full details for a specific job ID."""
    job = JOB_MEMORY.get(job_id)
    if not job: return {"status": "error", "message": "Job not found"}
    return {
        "status": "success", 
        "title": job.get('title'), 
        "company": job.get('company'),
        "description": job.get('description', '')[:3000] 
    }

def save_application_tool(job_id: str, cover_letter: str) -> dict:
    """Saves the generated cover letter to the application package."""
    print(f"   [Tool] ğŸ’¾ Saving Final Package for {job_id}...")
    APPLICATION_MEMORY[job_id] = {
        "status": "Ready", 
        "cover_letter": cover_letter,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return {"status": "success"}


print("âœ… Memory Bank & Tools defined.")



# ğŸ”� Job Banks Scrapping (Real-Time Job Scraper, no Mock Data.)

def JobSearchTool(search_terms_str: str, location: str, seniority_level: str) -> dict:
    search_terms = [s.strip() for s in search_terms_str.split(',')]
    print(f"   [Tool] ğŸš€ Searching for: {search_terms} in '{location}'")
    print(f"          (Target Seniority: {seniority_level})")
   
    all_jobs_dfs = []
   
    # 1. Live Scrape Loop
    for term in search_terms:
        print(f"   [Tool] ğŸ”� Scraping '{term}'...")
        try:
            jobs_df = scrape_jobs(
                site_name=["indeed"], #Limit for assignment, in real life should include other job banks like "Linkedin" & "Glassdoor"
                search_term=term,
                location=location,
                # country_indeed='Canada', # For Canada need to include this line
                results_wanted=30,  # Limit results to prevent overload, in real life suggest to put 50+
                sort_by='date'
            )
            if not jobs_df.empty:
                all_jobs_dfs.append(jobs_df)
            time.sleep(1)
        except Exception as e:
            print(f"   [Tool] âš ï¸� Live Job Scrap Error on '{term}': {e}")

    if not all_jobs_dfs:
        print("   [Tool] â�Œ No jobs found. (Scraper may be blocked or query too specific)")
        return {"status": "empty", "jobs": []}

    # 2. Processing
    combined_df = pd.concat(all_jobs_dfs, ignore_index=True)
    combined_df.drop_duplicates(subset=['job_url'], keep='first', inplace=True)
    combined_df = combined_df.fillna("Unknown")     # This prevents the "ClientError: 400 ... NaN" crash
   
    if 'description' in combined_df.columns:        ## Filter Empty Descriptions
        combined_df = combined_df[combined_df['description'].str.len() > 50]

    # Store in Memory
    jobs_data = combined_df.to_dict(orient='records')
    result_list = []
    
    for job in jobs_data:
        # Robust ID Generation: Company + Title + Location
        comp = job.get('company', 'Unk')
        titl = job.get('title', 'Unk')
        loc = job.get('location', 'Unk')
        
        clean_id = f"{comp}_{titl}_{loc}".replace(" ", "_").lower()
        clean_id = re.sub(r'[^a-z0-9_]', '', clean_id)[:60] # Limit length
        
        job['job_id'] = clean_id
        JOB_MEMORY[clean_id] = job 
        
        result_list.append({
            "job_id": clean_id, 
            "title": titl, 
            "company": comp,
            "location": loc
        })

    # LIMIT RETURN TO 15 TO PREVENT TOKEN OVERFLOW (In real life user can expand based on user's LLM plan)
    print(f"   [Tool] âœ… Passed {len(result_list)} valid candidates to Agent.")
    return {"status": "success", "jobs": result_list[:15]}

print("âœ… JobSearchTool defined.")


# Global Prompts

# --- Agent 1. THE HEADHUNTER (Finds the right opportunities) ---
SCOUT_INSTRUCTION = """
You are an Expert Recruiter.
1. Analyze the User's Resume and Request to determine Seniority (Junior/Mid/Senior).
2. Determine 3 distinct search terms for job bank keyword search.
3. Call `JobSearchTool` ONCE with comma-separated terms.
4. Analyze the results. 
5. Score each job (0-10) based on fit.
   - 0-4: Bad fit (e.g. Senior role for Junior candidate, totally unrelated domains, very little skills or experience matched).
   - 5-8: Good fit (Partial skills match, title close or similar job functions).
   - 9-10: Perfect fit (Exact match).

OUTPUT FORMAT:
"MATCH: [job_id] | SCORE: [Score]"
"""

# --- Agent 2. THE COACH (Resume improvement advisory) ---
COACH_INSTRUCTION = """
You are a Resume Editor. 
Compare the JOB DESCRIPTION vs the RESUME.
Identify ONE specific text block (Summary or Skill) to improve.
Do NOT create or hallucinate new experience for the user. 
Try to enhance user's existing experiences to fit the job opportunity better.

OUTPUT EXACTLY THIS FORMAT:
**PROPOSAL:**
> [Old Text]
**CHANGED TO:**
> [New Text]
**REASON:** [Why?]
"""


# --- Agent 3. THE Cover Letter Writer Agent ---
WRITER_INSTRUCTION = """
Write a professional Cover Letter (300 words).
You MUST call `save_application_tool` with the FULL TEXT of the letter.
DO NOT ASK QUESTIONS. JUST EXECUTE.
"""

print("âœ… Agent Prompts defined.")


# Test case Resume #1 â€” Mid-Level Data Analyst / Data Scientist (5 Years Experience)

RESUME_SENIOR = """
Name: Alex Chen
Location: Toronto, ON, Canada
Email: alex.chen@example.com

Phone: 416-555-1834
LinkedIn: linkedin.com/in/alexchen-da
Portfolio: alexchen-analytics.com

Professional Summary

Data Analyst with 5 years of experience in analytics, data engineering, and applied machine learning across SaaS and e-commerce industries. Skilled in building scalable analytics pipelines, developing predictive models, and partnering with stakeholders to deliver data-driven business decisions. Seeking Senior Data Analyst or Data Scientist roles with opportunities to lead projects and mentor junior analysts.

Technical Skills

Languages: Python, SQL, R
Tools & Platforms: BigQuery, Snowflake, dbt, Airflow, Git, Docker
Libraries: Pandas, NumPy, Scikit-learn, TensorFlow, Matplotlib
Cloud: GCP (preferred), AWS
Other: Data modeling, ETL, A/B testing, dashboarding (Looker, Tableau)

Professional Experience

Data Analyst â†’ Senior Data Analyst (Promotion in 2023)
Nimbus Commerce (E-commerce SaaS), Toronto, ON Aug 2020 â€“ Present

Built automated ETL pipelines using Airflow and dbt that reduced manual reporting time by 80%.

Designed and maintained the companyâ€™s product analytics data model in BigQuery.

Led development of a customer churn prediction model (AUC 0.82), contributing to a 12% improvement in retention.

Conducted A/B testing frameworks for marketing and product teams, including experiment design, statistical analysis, and reporting.

Collaborated with engineering to define data requirements for new product features.

Mentored two junior analysts and established team documentation guidelines.

Produced executive dashboards in Looker used by VP-level leadership for weekly business reviews.

Data Analyst (Intern â†’ Full-Time)
BrightSky Logistics, Vancouver, BC Jul 2019 â€“ Jul 2020

Created SQL-based reporting workflows that improved data accuracy in operational reports by 25%.

Built dashboards in Tableau to track delivery performance, resulting in a 15% reduction in delays.

Supported ad-hoc analysis on routing efficiency and cost optimization.

Education
B.Sc. in Statistics
University of British Columbia 2019

Certifications
Google Data Analytics Professional Certificate
dbt Fundamentals
"""


# Test case Resume #2 â€” Junior Finance / Accounting Candidate (1 Year Experience)

RESUME_JUNIOR = """
Name: Emily Rodriguez
Location: New York, NY
Email: emily.rodriguez@example.com

Phone: 917-555-9843
LinkedIn: linkedin.com/in/emilyrodriguez-fa

Professional Summary

Junior financial analyst with 1 year of experience in corporate banking, supporting budgeting, forecasting, and financial reporting processes. Strong analytical skills with a solid foundation in accounting principles, variance analysis, and Excel modeling. Seeking a Financial Analyst role in banking or financial services.

Technical Skills

Financial Tools: Excel (advanced), Power BI, SAP, Oracle ERP
Skills: Budgeting, forecasting, financial reporting, variance analysis, reconciliations
Other: SQL (basic), Python (basic), dashboard building

Professional Experience
Junior Financial Analyst

Citadel Bank Corp, New York, NY Jul 2024 â€“ Present

Supported monthly budgeting and forecasting cycles for the Corporate Banking unit.

Performed variance analysis comparing actuals vs. budget; identified drivers that improved forecasting accuracy by 8%.

Prepared financial reports for internal management review and regulatory compliance.

Assisted in automating Excel models using formulas, pivot tables, and lookups, reducing processing time by 30%.

Collaborated with accounting to reconcile expenses and validate journal entries.

Accounting Intern

Hudson Financial Services, New York, NY Jan 2024 â€“ Jun 2024

Assisted with accounts payable/receivable processing and month-end close tasks.

Conducted reconciliations and supported the preparation of financial statements.

Improved data accuracy by developing a new Excel template for tracking operating expenses.

Education

B.B.A. in Accounting & Finance
Baruch College, Zicklin School of Business 2024

Certifications

Bloomberg Market Concepts (BMC)

Financial Modeling & Valuation (FMVA) â€” In progress

"""


# ğŸ¤– SYSTEM ORCHESTRATOR

def get_agent_text_safely(turn):
    if not hasattr(turn, "content") or turn.content is None: return ""
    if hasattr(turn.content, "text") and turn.content.text: return turn.content.text
    parts = []
    if hasattr(turn.content, "parts") and turn.content.parts:
        for p in turn.content.parts:
            if hasattr(p, "text") and p.text: parts.append(p.text)
    return "\n".join(parts)

async def run_final_system(user_request, resume_text):

    # STEP 0: RESET SESSION MEMORY
    global APPLICATION_MEMORY
    APPLICATION_MEMORY = {} 
    
    # 1. IDENTITY EXTRACTION
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", resume_text)
    user_email = email_match.group(0) if email_match else "guest@example.com"
    
    # 2. LOAD LTM
    db = load_db()
    if user_email not in db: db[user_email] = {}

    print(f"\nğŸ¤– --- IDENTIFIED USER: {user_email} ---")

    # --- DEFINE AGENTS ---
    scout = LlmAgent(name="Scout", model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config), instruction=SCOUT_INSTRUCTION, tools=[JobSearchTool, get_job_details_tool])
    coach = LlmAgent(name="Coach", model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config), instruction=COACH_INSTRUCTION, tools=[get_job_details_tool])
    writer = LlmAgent(name="Writer", model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config), instruction=WRITER_INSTRUCTION, tools=[save_application_tool, get_job_details_tool])
    
    # --- PHASE 1: SCOUTING ---
    runner = InMemoryRunner(scout)
    print("\nğŸ•µï¸�â€�â™€ï¸� Scouting & Strategy...")
    
    events = await runner.run_debug(f"USER REQUEST: {user_request}\n\nRESUME:\n{resume_text}")
    
    # Parse Matches
    last_response = ""
    for event in reversed(events):
        text = get_agent_text_safely(event)
        if "MATCH:" in text:
            last_response = text
            break
            
    matches = re.findall(r'MATCH:\s*(.*?)\s*\|\s*SCORE:\s*(\d+)', last_response)
    
    if not matches:
        print("\nâ�Œ No matches found today.")
        return

    matches = sorted(matches, key=lambda x: int(x[1]), reverse=True)
    print(f"\nğŸ�¯ Found {len(matches)} Candidates. Processing Top 2...")

    # --- PHASE 2: LOOP ---
    processed_count = 0
    
    for job_id, score in matches:
        if processed_count >= 2: break 
        score = int(score)

        # COMMON SENSE FILTER: Score < 6 is rejected
        if score < 6: continue 

        # LTM Duplicate Check
        if job_id in db[user_email]:
            print(f"   â�­ï¸� Skipping {job_id} (Already recommended previously)")
            continue

        processed_count += 1
        job_info = JOB_MEMORY.get(job_id, {})
        company = job_info.get('company', 'Unknown')
        job_desc = job_info.get('description', '')

        print(f"\n==================================================")
        print(f"âš™ï¸� CANDIDATE: {company} (Score: {score}/10)")
        print(f"   Role: {job_info.get('title')}")
        print(f"==================================================")
        
        resume_update_html = "" 
        
        # --- PATH A: PERFECT MATCH (9-10) ---
        if score >= 9:
            print("âœ… Score is High (9-10). Resume is good to go!")
            resume_update_html = "<em>Resume is a Perfect Match. No changes needed.</em>"
            
            writer_runner = InMemoryRunner(writer)
            prompt = f"Create package for job_id: {job_id}.\nJOB DESC:\n{job_desc}\nRESUME:\n{resume_text}"
            await writer_runner.run_debug(prompt)
            
            db[user_email][job_id] = {"status": "Auto-Generated", "date": datetime.now().isoformat()}
            save_db(db)

        # --- PATH B: GOOD MATCH (5-8) ---
        else:
            print("âš ï¸� Good Match (5-8). Calling Coach...")
            
            coach_runner = InMemoryRunner(coach)
            coach_prompt = f"Tailor resume for job_id: {job_id}.\nJOB DESC:\n{job_desc}\nRESUME:\n{resume_text}"
            coach_events = await coach_runner.run_debug(coach_prompt)
            
            coach_output = ""
            for event in reversed(coach_events):
                text = get_agent_text_safely(event)
                if "**PROPOSAL:**" in text or "**CHANGED TO:**" in text:
                    coach_output = text
                    break
            
            display(Markdown("### ğŸ’¡ Resume Coach Suggestion"))
            display(Markdown(coach_output))
            
            # === REAL HUMAN IN THE LOOP ===
            print(f"\nğŸ‘‰ ACTION REQUIRED: Do you approve this resume change for {company}?")
            user_choice = input("Type 'y' to approve, 'n' to reject: ") 
            
            if user_choice.lower() == 'y':
                print("\nâœ… Approved. Generating Application Package...")
                resume_update_html = f"<strong>Approved Update:</strong><br><pre>{coach_output}</pre>"
                
                writer_runner = InMemoryRunner(writer)
                writer_prompt = f"""
                Create package for job_id: {job_id}.
                JOB DESC: {job_desc}
                RESUME: {resume_text}
                USER APPROVED CHANGE: {coach_output}
                """
                await writer_runner.run_debug(writer_prompt)
                
                db[user_email][job_id] = {"status": "User-Approved", "date": datetime.now().isoformat()}
                save_db(db)
            else:
                print("â�Œ Rejected. Skipping this job.")
                continue
        
        if job_id in APPLICATION_MEMORY:
            APPLICATION_MEMORY[job_id]['resume_status'] = resume_update_html


    
    # --- FINAL HTML REPORT ---
    print("\n\n")
    display(Markdown("# ğŸ“¦ Final Application Packages"))
    
    if not APPLICATION_MEMORY:
        print("No packages generated.")
    else:
        for job_id in APPLICATION_MEMORY:
            pkg = APPLICATION_MEMORY[job_id]
            job = JOB_MEMORY.get(job_id, {})
            
            html_content = f"""
            <div style="border:1px solid #ddd; border-radius:10px; padding:20px; margin-bottom:20px; background-color:#ffffff; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #eee; padding-bottom:10px;">
                    <div>
                        <h2 style="margin:0; color:#2c3e50;">{job.get('title')}</h2>
                        <h4 style="margin:5px 0 0 0; color:#7f8c8d;">{job.get('company')} | {job.get('location')}</h4>
                    </div>
                    <a href="{job.get('job_url')}" target="_blank" style="background-color:#3498db; color:white; padding:10px 15px; text-decoration:none; border-radius:5px; font-weight:bold;">Apply Now â†—</a>
                </div>
                
                <div style="margin-top:15px;">
                    <h4 style="color:#e67e22;">Resume Status</h4>
                    <div style="background:#fff3e0; padding:10px; border-left:4px solid #e67e22;">
                        {pkg.get('resume_status', 'No changes recorded.')}
                    </div>
                </div>

                <div style="margin-top:15px;">
                    <h4 style="color:#27ae60;">Cover Letter</h4>
                    <div style="background:#f9f9f9; padding:15px; border:1px solid #eee; white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 0.9em;">
{pkg.get('cover_letter')}
                    </div>
                </div>
            </div>
            """
            display(HTML(html_content))

    
    # --- PROOF OF LTM ---
    print("\n\n------------------------------------------------")
    print("ğŸ’¾ LONG TERM MEMORY DUMP (Proof of Storage)")
    print("------------------------------------------------")
    print(json.dumps(db, indent=2))



# RESUME_SENIOR
# The agent will auto-detect "alex.chen@example.com" and use that to track job history
# Choose your request
current_resume = RESUME_SENIOR
user_query = "I am looking for a Senior Data Scientist or Data Analytics manager job in Seattle"

await run_final_system(user_query, current_resume)



# RESUME_JUNIOR
# The agent will auto-detect "emily.rodriguez@example.com" and use that to track job history
current_resume = RESUME_JUNIOR
user_query = "I want a Junior Financial Analyst job in New York"

await run_final_system(user_query, current_resume)

