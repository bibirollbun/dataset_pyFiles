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


from typing import Any, Dict, List
import json
import time
import logging

from google.genai import types

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

print("âœ… ADK components imported successfully.")

# Retry configuration
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Session service (conceptual)
session_service = InMemorySessionService()
print("âœ… Session service (conceptual) created.")


def pretty_print_json(data: Any):
    print(json.dumps(data, indent=2, ensure_ascii=False))

print("âœ… Helper pretty printer ready.")


# å…¨å±€æ¨¡æ‹Ÿæ•°æ�®åº“/å�˜é‡�
logging.basicConfig(level=logging.ERROR)

JOB_DATABASE = [
    {
        "id": "JOB-001",
        "title": "Senior Python Backend Engineer",
        "company": "TechGiant Inc.",
        "location": "Beijing",
        "description": "Responsible for developing and maintaining high-concurrency web services. Requires proficiency in Python and common frameworks (Flask/Django). Experience in e-commerce projects is preferred.",
        "skills_required": ["Python", "Flask", "Django", "Distributed Systems", "E-commerce"]
    },
    {
        "id": "JOB-002",
        "title": "Data Analyst",
        "company": "DataCorp Ltd.",
        "location": "Shanghai",
        "description": "Participate in business data analysis. Requires proficiency in SQL and Python for data cleaning and statistical modeling. Experience with data visualization is required.",
        "skills_required": ["SQL", "Python", "Data Analysis", "Statistics", "Visualization"]
    },
    {
        "id": "JOB-003",
        "title": "Frontend Developer",
        "company": "WebStudio",
        "location": "Shanghai",
        "description": "Responsible for company web frontend development. Requires mastery of HTML/CSS/JavaScript and hands-on experience with mainstream frontend frameworks.",
        "skills_required": ["HTML", "CSS", "JavaScript", "React", "Vue"]
    }
]

def search_jobs_database(query: str, location: str = "") -> str:
    """
    Simulates a mock job search engine.
    
    1. Broad Match Strategy:
       - Returns a match if ANY keyword from the query appears in the 'title', 'description', or 'skills'.
       
    2. Fallback Logic (if no matches are found):
       - First, retry using a core keyword (e.g., 'Python').
       - If that fails, return all jobs located in the same city.
       - As a final resort, return the entire dataset.
    """
    try:
        print(f"ğŸ”� æ­£åœ¨æ•°æ�®åº“ä¸­æ�œç´¢: '{query}' @ {location}...")
        time.sleep(0.5)  # æ¨¡æ‹Ÿå»¶è¿Ÿ

        q = query.lower()

        # æŠŠç‰¹åˆ«é•¿çš„ä¸€ä¸² queryï¼Œåˆ‡æˆ�è‹¥å¹²å…³é”®è¯�
        # ä¾‹å¦‚ â€œPython backend developer, CS bachelor's degree...â€� -> ['python','backend','developer',...]
        raw_tokens = [t.strip(" ,.;") for t in q.split()]
        keywords = [t for t in raw_tokens if len(t) > 2] or [q]

        def match_job(job, kws):
            text_blob = " ".join([
                job["title"],
                job["description"],
                " ".join(job.get("skills_required", [])),
            ]).lower()
            return any(kw in text_blob for kw in kws)

        # ç¬¬ä¸€ä¸ªç­›é€‰ï¼šæŒ‰ç”¨æˆ·ç»™çš„å…³é”®è¯�ç²—ç•¥åŒ¹é…�
        results = [
            job for job in JOB_DATABASE
            if match_job(job, keywords)
            and ((not location) or (location in job["location"] or location.lower() == "any"))
        ]

        # å¦‚æ�œæ²¡æœ‰ç»“æ�œï¼Œå°�è¯•ç”¨â€œpythonâ€�è¿™ç§�å¼º signal å†�æ�œä¸€è½®
        if not results:
            core_kws = [kw for kw in keywords if "python" in kw or "å��ç«¯" in kw or "backend" in kw]
            if core_kws:
                results = [
                    job for job in JOB_DATABASE
                    if match_job(job, core_kws)
                    and ((not location) or (location in job["location"] or location.lower() == "any"))
                ]

        # å¦‚æ�œè¿˜æ˜¯æ²¡æœ‰ï¼Œå°±è‡³å°‘ç»™å‡ºå�ŒåŸ�å¸‚çš„å²—ä½�ï¼ˆå±•ç¤ºç³»ç»Ÿèƒ½åŠ›ï¼‰
        if not results and location:
            results = [job for job in JOB_DATABASE if location in job["location"]]

        # å¦‚æ�œæ•°æ�®åº“æœ¬èº«å°±å¾ˆå°�ï¼Œé‚£å°±å¹²è„†è¿”å›�æ‰€æœ‰ï¼Œè®©ä¸Šå±‚ Agent å�»ç­›
        if not results:
            results = JOB_DATABASE

        return json.dumps(results, ensure_ascii=False)

    except Exception as e:
        logging.error("search_jobs_database å‡ºé”™: %s", e)
        # å‡ºé”™æ—¶ä¹Ÿå°½é‡�è¿”å›�ä¸€ä¸ªé��ç©ºçš„å®‰å…¨ç»“æ�œï¼Œåˆ«è®©å¤šAgenté“¾æ–­æ�‰
        return json.dumps(JOB_DATABASE, ensure_ascii=False)


APPLICATION_DB: List[Dict[str, str]] = []  # ç”¨äº�å­˜å‚¨æ±‚è�Œç”³è¯·è®°å½•

def log_application_record(company: str, job_title: str, status: str = "Applied") -> str:
    """
    Saves a user's job application record to the APPLICATION_DB.

    The `status` must be one of 'Applied', 'Interviewing', 'Offer', or 'Rejected'.
    """
    try:
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "company": company,
            "job_title": job_title,
            "status": status
        }
        APPLICATION_DB.append(record)
        return f"âœ… å·²è®°å½•ç”³è¯·: {company} - {job_title} [{status}]"
    except Exception as e:
        return f"â�Œ è®°å½•å¤±è´¥: {e}"

def show_application_records() -> str:
    """
    Returns a JSON string containing the list of all current job application records. 
    If no records exist, returns a notification message.
    """
    try:
        if not APPLICATION_DB:
            return "[]"
        return json.dumps(APPLICATION_DB, ensure_ascii=False)
    except Exception as e:
        logging.error("show_application_records å‡ºé”™: %s", e)
        return "[]"


# log_application_record_tool = FunctionTool(log_application_record)
# show_application_records_tool = FunctionTool(show_application_records)


def get_user_resume() -> str:
    """
    Retrieves the text content of the user's current resume.

    In a real-world scenario, this tool would fetch the latest resume from the user's profile or database.
    """
    return USER_RESUME

print("âœ… å·¥å…·å‡½æ•°å®šä¹‰å®Œæˆ�ã€‚")


# 1. å²—ä½�ä¾¦å¯Ÿå‘˜ Agent - æ�œç´¢å¹¶æ�¨è��è�Œä½�
job_scout = LlmAgent(
    name="JobScout",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction= """
# Role
You are a professional headhunter. Your goal is to find the best job opportunities for users by intelligently querying the database and matching results with their background.

# Tools Available
1. `search_jobs_database`: Search for job listings.
2. `get_user_resume`: Retrieve the user's resume for profile matching.

# Instructions for Tool Usage
When using `search_jobs_database`:
- **Query Construction**:
    - NEVER use the user's raw input as the query.
    - Extract 3-6 core keywords from the user's request (e.g., Job Title, Core Skills, Industry).
    - Format: "Job Title Skill1 Skill2 Industry".
    - Example: "Python Backend Developer Flask Django E-commerce".
- **Location Parameter**:
    - Use ONLY the city name (e.g., "Beijing", "Shanghai").

# Recommendation Logic
If `search_jobs_database` returns multiple results:
1. Call `get_user_resume` to understand the user's background.
2. Cross-reference the resume with the job descriptions.
3. Select and recommend the **top 2-3 matches**.
""",
    tools=[search_jobs_database, get_user_resume]
)

# 2. ç®€å�†å®šåˆ¶æµ�æ°´çº¿ Agent - ä¾�æ¬¡æ‰§è¡Œ Gap åˆ†æ��å’Œæ–‡ä¹¦ç”Ÿæˆ�
# 2.A å·®è·�åˆ†æ��å­�Agent
gap_analyst = LlmAgent(
    name="GapAnalyst",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    output_key="analysis_report",  # å°†è¾“å‡ºå­˜å…¥å�˜é‡�ï¼Œä¾›ä¸‹ä¸€ä¸ªAgentä½¿ç”¨:contentReference[oaicite:42]{index=42}
    instruction= """
# Role
You are a Senior Career Consultant specializing in comparing job requirements with candidate resumes.

# Task
Conduct a detailed comparison between the user's resume and the target Job Description (JD). Identify gaps and output an analysis report.

# Report Requirements
The report must include:
1. **Keywords Gap**: Core skills or keywords missing from the user's profile.
2. **Strengths**: Areas where the user's background highly matches the job requirements.

# Output Format
Please output the result strictly in **JSON format**.
Example: `{"missing_skills": ["Skill A", "Skill B"], "strengths": ["Skill C", "Experience D"]}`

# Constraints
- **Analyze ONLY**: Focus solely on the gap analysis.
- **DO NOT** generate a full cover letter or rewrite the resume content.
"""
)
# 2.B æ–‡ä¹¦æ’°å†™å­�Agent
resume_writer = LlmAgent(
    name="ResumeWriter",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction= """
# Role
You are an experienced Resume Optimization Expert and Professional Copywriter.

# Input Source
You will be provided with a **"Gap Analysis Report"** (containing `missing_skills` and `strengths`) from the previous step.

# Task
Based on the analysis report, generate the following two sections for the user:

1. **Optimized Profile Summary** (for the Resume):
   - Highlight the user's `strengths`.
   - Strategically address or frame the `missing_skills` to show learning potential or transferable skills.
   - Keep it concise and impactful.

2. **Cover Letter**:
   - Target the specific job.
   - Tone: Professional, confident, and sincere.
   - Focus: Emphasize the alignment between the user's background and the job requirements based on the identified strengths.

# Output Format
Please output the response in **Markdown** format, using clear headers to separate the **Profile Summary** and the **Cover Letter**.
"""
)
# SequentialAgent å°† GapAnalyst å’Œ ResumeWriter æŒ‰é¡ºåº�æ‰§è¡Œ
career_tailor = SequentialAgent(
    name="CareerTailor",
    description="Automatically analyze gaps between the resume and the job description to generate tailored cover letters and profile summaries.",
    sub_agents=[gap_analyst, resume_writer]  # é¡ºåº�è°ƒç”¨GapAnalystï¼Œç„¶å��ResumeWriter:contentReference[oaicite:43]{index=43}
)

# # 3. æ¨¡æ‹Ÿé�¢è¯•æ•™ç»ƒ Agent
# interview_coach = LlmAgent(
#     name="InterviewCoach",
#     model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
#     instruction= """
# # Role
# You are a strict but friendly **Interview Coach**, specializing in **Behavioral Interview** training.

# # Workflow
# 1. **Initialization**: 
#    - If this is the start of the session and the user hasn't specified a target role, ask: "Which position would you like to simulate an interview for today?"
# 2. **Questioning**: 
#    - Propose a typical interview question relevant to the target position (behavioral or technical).
# 3. **Evaluation (STAR Method)**: 
#    - After the user answers, briefly critique their response using the **STAR method** (Situation, Task, Action, Result).
#    - Point out strengths and specific areas for improvement.
#    - If the user's answer is too vague, feel free to ask follow-up questions (e.g., "Can you be more specific about your role in that project?") before giving the final STAR feedback.
# 4. **Progression**: 
#    - Ask the next question, focusing on a different competency.
#    - Continue for **2-3 rounds** of Q&A.
# 5. **Conclusion**: 
#    - After the rounds are complete, summarize the user's overall performance, provide final tips for improvement, and politely end the interview.

# # Guidelines
# - **Tone**: Professional, encouraging, yet rigorous.
# - **Interaction**: Engage in a turn-by-turn dialogue. **DO NOT** ask all questions at once. Wait for the user's response before moving to the feedback or the next question.
# """
# )
# ï¼ˆæ³¨ï¼šInterviewCoach å°†åœ¨å¯¹è¯�å¾ªç�¯ä¸­è¢«å¤šæ¬¡è°ƒç”¨ï¼Œä»¥å®�ç�°å¤šè½®é�¢è¯•äº¤äº’ï¼‰

# 4. æ¡£æ¡ˆç®¡å®¶ Agent - ç®¡ç�†ç”³è¯·è®°å½•
archivist = LlmAgent(
    name="Archivist",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction= """
# Role
You are the user's **Job Application Manager**, responsible for tracking and managing job application records.

# Tools & Workflow

## 1. Log or Update Applications
- **Trigger**: When the user informs you about a new application or wants to update a status.
- **Action**: Call the `log_application_record` tool.
- **Required Parameters**:
  - `company_name`: The name of the company.
  - `job_title`: The title of the position.
  - `status`: Must be one of `Applied`, `Interviewing`, `Offer`, or `Rejected`.
- **Handling Missing Info**: If the user's input lacks any of the above details, **do not** call the tool yet. Politely ask the user to provide the missing information first.

## 2. Check Progress
- **Trigger**: When the user wants to check past application progress.
- **Action**: Use the `show_application_records` tool to retrieve data.
- **Output**: Present the records in a concise and clear report.

# Response Guidelines
- Always confirm that a record has been saved or successfully updated.
- When listing applications, keep the layout clean.
- Provide appropriate encouragement or advice on next steps (e.g., "Good luck with the interview!").
""",
    tools=[log_application_record, show_application_records]
)

job_scout_tool = AgentTool(job_scout)
career_tailor_tool = AgentTool(career_tailor)
archivist_tool = AgentTool(archivist)

# 5. æ€»æ�§ Agent - è°ƒåº¦ä¸­å¿ƒï¼Œå°†å­�Agentå°�è£…ä¸ºå·¥å…·ä¾›è°ƒç”¨
coordinator = LlmAgent(
    name="Coordinator",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction= """
You are the Coordinator of the JobSecure multi-agent job search system.

Your job is to **ORCHESTRATE** the following agents as a fixed pipeline.
Do NOT just describe what you will do. You MUST actually call the tools.

PIPELINE (always follow this, unless the user explicitly says they only want a single step):

1. Call the JobScout agent TOOL to search for 2â€“3 matching jobs.
   - Summarize each job in 3â€“5 bullet points.
2. Pick ONE target job and call the CareerTailor agent TOOL.
   - Ask it to generate: (a) tailored resume bullet points, (b) a short cover letter draft.
3. Optionally call Archivist if the user informs you about a new application or wants to update a status.
4. Finally, you MUST return a single Markdown answer to the user that includes:
   - A short summary of the selected job(s)
   - Tailored resume bullets
   - A cover letter draft
   - Clear next-step suggestions (interview prep, application tracking, etc.)

Important:
- Always prefer calling tools/agent tools over answering from your own knowledge.
- Never stop after only explaining the plan. You must **execute** it.
""",
    # å°†å�„å­�Agentå°�è£…ä¸ºå·¥å…·ä¾› Coordinator è°ƒç”¨:contentReference[oaicite:49]{index=49}
    tools=[
        job_scout_tool,
        career_tailor_tool,
        archivist_tool,
    ]
)

print("âœ… æ™ºèƒ½ä½“å®šä¹‰å®Œæˆ�ã€‚")



runner = InMemoryRunner(coordinator)
print("âœ… InMemoryRunner created for JobSecure Coordinator.")

demo_prompt = """
Hi, I'm looking for a Python backend developer job in Beijing.
Here is my background:

- CS bachelor, 2 years Python backend development
- Familiar with Flask/Django
- Worked on an e-commerce system rebuild

Please:
1. Recommend 2â€“3 matching jobs.
2. Then try to help me tailor my resume and cover letter for one of them.
3. Briefly tell me what we can do next (e.g., interview prep, application tracking).
"""

import asyncio  # è™½ç„¶ç”¨ä¸�ä¸€å®šç”¨åˆ°ï¼Œä½†ä¸€èˆ¬å¤§å®¶ä¼š import ä¸Š

response = await runner.run_debug(demo_prompt)

def extract_last_text(response, source_filter=None):
    """
    ä»� ADK run_debug çš„ response ä¸­æ��å�–æœ€å��ä¸€æ�¡æ–‡æœ¬ã€‚
    å¦‚æ�œä¼ å…¥ source_filterï¼ˆæ¯”å¦‚ 'Coordinator'ï¼‰ï¼Œåˆ™å�ªçœ‹æ�¥è‡ªè¯¥ source çš„äº‹ä»¶ã€‚
    """
    last_text = None
    last_source = None

    for i, turn in enumerate(response):
        source = getattr(turn, "source", f"Turn {i}")
        content = getattr(turn, "content", None)

        if source_filter and source != source_filter:
            continue

        if content is None:
            continue

        text = None
        # æœ‰çš„ turn æ˜¯ content.text
        if hasattr(content, "text") and content.text:
            text = content.text
        # æœ‰çš„æ˜¯ content.parts[â€¦] é‡Œæœ‰å¤šæ®µ text
        elif hasattr(content, "parts") and content.parts:
            texts = [getattr(p, "text", "") for p in content.parts if getattr(p, "text", None)]
            if texts:
                text = "\n".join(texts)

        if text:
            last_text = text
            last_source = source

    return last_text, last_source


final_text, who = extract_last_text(response, source_filter=None)  # ä½ ä¹Ÿå�¯ä»¥å¡« Coordinator

print("\n==================== FINAL OUTPUT ====================\n")
print(f"ğŸ”¹ æ�¥è‡ª: {who}")
print(final_text or "[No final text found]")

# # debug
# for i, turn in enumerate(response):
#     who = getattr(turn, "source", f"Turn {i}")
#     content = getattr(turn, "content", None)
#     print(f"--- Turn {i}: source={who} ---")
#     print(content)
#     print()



progress_update_prompt = """
Hi, I'm back with some updates on my job applications.

Here are my current applications:

1) Company: TechGiant Inc.
   Job Title: Senior Python Backend Engineer
   Status: Interviewing

2) Company: DataWave
   Job Title: Python Data Engineer
   Status: Applied
"""

response_tracking = await runner.run_debug(progress_update_prompt)

print("\n==================== SCENARIO 2: APPLICATION TRACKING ====================\n")
for i, turn in enumerate(response_tracking):
    who = getattr(turn, "source", f"Turn {i}")
    content = getattr(turn, "content", None)
    print(f"--- Turn {i}: source={who} ---")
    print(content)
    print()

# ä¾�ç„¶ç”¨ä½ ä¹‹å‰�çš„ extract_last_text æŠ½å‡ºç»™ç”¨æˆ·çœ‹çš„æœ€ç»ˆè¾“å‡º
final_text_tracking, who_tracking = extract_last_text(response_tracking, source_filter=None)
print("\n===== FINAL VISIBLE OUTPUT (SCENARIO 2) =====\n")
print(final_text_tracking)

# å†�æŠŠå½“å‰� APPLICATION_DB æ‰“å�°å‡ºæ�¥ï¼Œè®©è¯„å®¡çœ‹åˆ° â€œçŠ¶æ€�çœŸçš„è¢«è®°å½•äº†â€�
print("\n===== INTERNAL APPLICATION_DB STATE =====\n")
print(json.dumps(APPLICATION_DB, ensure_ascii=False, indent=2))



# ============================================================
# FINAL OUTPUT SUMMARY AND CREATION OF SUBMISSION FILE FOR KAGGLE
# ============================================================

import json

print("ğŸ”„ Running JobSecure multi-agent pipeline to regenerate final output...")

# ä½ å�¯ä»¥æŒ‰éœ€è¦�æ”¹æˆ�ä½ å¸Œæœ›å±•ç¤ºçš„â€œå…¸å�‹ç”¨æˆ·ç”»åƒ�â€�
demo_profile = """
My name is Alex. I have 2 years of experience as a Python backend developer.
Tech stack: Python, Flask, Django, REST APIs, PostgreSQL, Docker.
I previously worked on rebuilding an e-commerce backend system with higher scalability.
I'm currently based in Beijing and open to on-site or hybrid roles.
My target roles are: Python Backend Engineer / Backend Developer / Software Engineer (Backend).
"""

demo_prompt = f"""
USER_ID: user_001

BACKGROUND:
{demo_profile}

REQUEST:
1. Recommend 2â€“3 suitable Python backend positions based on my background and target roles.
2. Choose ONE of them and tailor my resume bullet points and a cover letter for that job.
3. If possible, briefly suggest what I should focus on next (e.g., interview prep, application strategy).
"""

# é‡�æ–°å®Œæ•´è·‘ä¸€é��å¤š Agent pipeline
demo_response = await runner.run_debug(demo_prompt)

print("\nâœ… Pipeline run finished. Extracting final output...\n")

# =============== æ��å�–æœ€ç»ˆå�¯è§�æ–‡æœ¬ï¼ˆå°½é‡�é²�æ£’ï¼‰ ===============

# é»˜è®¤å�–æœ€å��ä¸€ä¸ª turnï¼Œå¦‚æ�œæœ€å��ä¸€ä¸ªæ²¡æœ‰ textï¼Œå°±å�‘å‰�å›�æº¯
final_text = None
final_source = None

# ä»�å��å¾€å‰�æ‰¾ï¼Œé˜²æ­¢æœ€å��ä¸€ä¸ªæ˜¯ function_response ä¹‹ç±»æ²¡ç›´æ�¥æ–‡æœ¬çš„
for turn in reversed(demo_response):
    if not hasattr(turn, "content") or turn.content is None:
        continue

    content = turn.content
    text_candidate = None

    # å…¼å®¹ content.text
    if hasattr(content, "text") and content.text:
        text_candidate = content.text

    # å…¼å®¹ content.partsï¼ˆæµ�å¼�/å¤šæ®µè¾“å‡ºï¼‰
    elif hasattr(content, "parts") and content.parts:
        pieces = [p.text for p in content.parts if hasattr(p, "text") and p.text]
        if pieces:
            text_candidate = "\n".join(pieces)

    if text_candidate:
        final_text = text_candidate
        final_source = getattr(turn, "source", None)
        break

# Fallback
if final_text is None:
    final_text = "[No final text produced by the coordinator or sub-agents.]"

print("ğŸ”š Final text source:", final_source or "Unknown")
print("\n==================== FINAL AGENT OUTPUT (PREVIEW) ====================\n")
print(final_text[:2000])  # å�ªæ‰“å�°å‰� 2000 å­—é�¿å…�å¤ªé•¿
print("\n=====================================================================\n")

# =============== å°�è¯•è§£æ��ä¸º JSONï¼Œå¦‚æ�œå¤±è´¥å°±åŒ…è£…ä¸€å±‚ ===============

json_payload = {}
try:
    # æœ‰äº›äººä¼šè®©æœ€ç»ˆè¾“å‡ºæœ¬èº«æ˜¯ä¸€ä¸ª JSONï¼Œæˆ–è€… "```json {...} ```" å½¢å¼�
    # ç®€å�•å�šæ³•ï¼šä»�ç¬¬ä¸€ä¸ª '{' å¼€å§‹æˆªå�–å°�è¯•è§£æ��
    start_idx = final_text.find("{")
    if start_idx != -1:
        json_candidate = final_text[start_idx:]
        json_payload = json.loads(json_candidate)
        print("âœ… Parsed final output as JSON from agent response.")
    else:
        raise ValueError("No JSON object found in final_text")

except Exception as e:
    print("âš ï¸� JSON parsing failed:", e)
    # å…œåº•ï¼šç”¨ä¸€ä¸ªç®€å�• schemaï¼ŒæŠŠå�Ÿå§‹æ–‡æœ¬å¡�è¿›å�»
    json_payload = {
        "user_id": "user_001",
        "agent_pipeline": "JobSecure multi-agent (Coordinator + JobScout + CareerTailor + optional Archivist)",
        "raw_output": final_text,
    }

# =============== ä¿�å­˜ä¸º Kaggle å�¯ä¸‹è½½æ–‡ä»¶ ===============

output_filename = "jobsecure_output.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(json_payload, f, indent=2, ensure_ascii=False)

print(f"ğŸ“� Saved submission/demo file: {output_filename}")
print("ğŸ�‰ You can now download this file from the Kaggle notebook and, if needed, attach it as part of your competition submission / report.")


