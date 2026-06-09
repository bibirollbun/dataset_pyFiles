!pip install -U google-genai


pip install google-adk


!pip install -q nest_asyncio


import logging
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)


import os
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient

# 1. Load your API key from Kaggle Secrets
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

# 2. Export key so tools can use it
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 3. Ensure Gemini API (not Vertex)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

# 4. Initialize client **with API key** (this is REQUIRED on Kaggle)
client = genai.Client(api_key=GOOGLE_API_KEY)

# 5. Test call
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with the text: READY-OK",
)

print(response.candidates[0].content.parts[0].text)



METRICS = {
    "messages_handled": 0,
    "jobs_saved": 0,
    "agent_calls": {}   # dynamically filled, no fixed names
}

def show_metrics():
    print("ğŸ“Š Metrics Dashboard")
    print("====================")
    print(f"Messages handled: {METRICS['messages_handled']}")
    print(f"Jobs saved: {METRICS['jobs_saved']}\n")

    print("Agent Calls:")
    if not METRICS["agent_calls"]:
        print("  (no agent calls recorded yet)")
    else:
        for agent, count in METRICS["agent_calls"].items():
            print(f"  - {agent}: {count}")



import datetime
from typing import Optional, Dict, Any, List

# Simple in-memory "database"
JOB_DB: List[Dict[str, Any]] = []

def add_job_application(
    title: str,
    company: str,
    job_link: Optional[str] = None,
    status: str = "planned",
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add a job application record to the in-memory database.

    Args:
        title: Job title (e.g., 'Senior Data Scientist').
        company: Company name.
        job_link: Optional link to the posting.
        status: Application status ('planned', 'applied', 'interview', 'offer', etc.).
        notes: Any free-text notes.

    Returns:
        The job record that was stored.
    """
    global METRICS
    # Metrics tracking (safe, synchronous, Kaggle-friendly)
    METRICS["jobs_saved"] += 1

    record = {
        "id": len(JOB_DB) + 1,
        "title": title,
        "company": company,
        "job_link": job_link or "",
        "status": status,
        "notes": notes or "",
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    JOB_DB.append(record)
    return record


def list_job_applications() -> Dict[str, Any]:
    """
    List all job applications currently in the in-memory database.
    """
    return {
        "count": len(JOB_DB),
        "jobs": JOB_DB,
    }



from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

APP_NAME = "career_concierge_app"
USER_ID = "user-1"
SESSION_ID = None  # will be set by init_session()

career_agent = Agent(
    name="career_concierge_agent",
    model="gemini-2.5-flash",
    description=(
        "An AI career concierge that helps users analyze their profile, "
        "understand job descriptions, and manage job applications."
    ),
    instruction="""
You are an AI Career Concierge for individual job seekers.

Capabilities:
1. Ask smart questions about the user's background, target roles, and constraints.
2. Help them position their profile for AI/ML/GenAI or data roles, and identify skill gaps.
3. When the user describes a specific job they are interested in, you can:
   - Extract: job title, company, (optional) job link, and status.
   - Call the `add_job_application` tool to store it in the job tracker.
4. When the user asks to review or see their applications, call `list_job_applications`
   and summarize the result clearly for them.

Important:
- Before calling a tool, confirm with the user what you are going to save.
- After using a tool, explain in plain language what was done.
- Keep responses structured with headings and bullet points.

Output style:
- Use headings like: ## Summary, ## Recommendations, ## Saved Jobs.
- Use short paragraphs and bullet lists.
""",
    tools=[
        add_job_application,
        list_job_applications,
    ],
)


# Runner using the orchestrator agent
runner = InMemoryRunner(
    agent=career_agent,
    app_name=APP_NAME,
)

async def init_session():
    """
    Create a new ADK session and store its id in SESSION_ID.
    This is async and must be called with `await init_session()` in a notebook cell.
    """
    global SESSION_ID
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
    )
    SESSION_ID = session.id
    print("âœ… Created session:", SESSION_ID)

def run_agent(user_message: str):
    global SESSION_ID

    METRICS["messages_handled"] += 1

    if SESSION_ID is None:
        print("âš ï¸� SESSION_ID not set. Run: await init_session() first.")
        return

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    print(f"ğŸ‘¤ User: {user_message}\n")

    for event in runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):

        # Normalize the author
        author_raw = getattr(event, "author", "") or ""
        author = author_raw.strip() or "unknown"

        # Update metrics dynamically
        if author not in METRICS["agent_calls"]:
            METRICS["agent_calls"][author] = 0
        METRICS["agent_calls"][author] += 1

        # Print response text
        if event.content and event.content.parts:
            text = event.content.parts[0].text or ""
            if text.strip():
                print(f" {author}: {text}\n")


await init_session()



# 1) Start the convo
run_agent("Hi, I want help with my AI/ML job search.")

# 2) Give some context about yourself (you!)
run_agent(
    "I have over 10 years in SAP Sales Cloud and CRM, and I'm doing a DBA with focus on AI/ML. "
    "Help me position myself for senior AI and data roles."
)

show_metrics()



job_desc = """
There's a role: Senior Principal Data Architect at JPMorgan Chase in Dublin.
It's a hybrid role focusing on data platforms, AI/ML, and employee experience.
Please save this in my job tracker as something I plan to apply for.
"""
run_agent(job_desc)

run_agent("Show me all the jobs you have saved for me so far.")

show_metrics()



from google.adk.agents import Agent

# 1) Profile Agent: CV-like text â†’ JSON profile
profile_agent = Agent(
    name="profile_agent",
    model="gemini-2.5-flash",
    description="Builds a structured user profile from CV-like free text.",
    instruction="""
You are a Profile Agent.

Input:
- Free-text description of the user's background, CV, LinkedIn-like info, or career goals.

Your task:
- Extract a clean, structured representation of the user's profile.

Output JSON exactly in this format:
{
  "name": "string or empty if unknown",
  "headline": "1-line career headline",
  "years_experience": "approx years as integer or string",
  "primary_domains": ["Data Science", "AI/ML", "SAP CRM", ...],
  "core_skills": ["Python", "LLMs", "Databricks", ...],
  "experience_summary": [
    "Short bullet about key achievement/role 1",
    "Short bullet about key achievement/role 2"
  ],
  "education": [
    "Degree, Field, Institution (if known)"
  ],
  "target_roles": [
    "e.g., Senior Data Scientist",
    "e.g., Principal AI Architect"
  ]
}

Rules:
- Always return valid JSON only.
- If a field is unknown, use empty string or empty list.
""",
)

# 2) JD Analyzer Agent
jd_analyzer_agent = Agent(
    name="jd_analyzer_agent",
    model="gemini-2.5-flash",
    description="Analyzes job descriptions and extracts structured insights.",
    instruction="""
You are a Job Description Analyzer Agent.

Your job is to:
1. Extract key responsibilities.
2. Extract required skills.
3. Identify domain knowledge (e.g., AI/ML, cloud, data engineering).
4. Estimate seniority level (Junior/Intermediate/Senior/Principal).
5. Summarize the role in 3-5 bullet points.
6. Compute a fit score between 1â€“10 based on how well the user's profile matches (if profile is provided).

Output JSON exactly in this format:
{
  "title": "...",
  "company": "...",
  "responsibilities": [...],
  "skills_required": [...],
  "domain": "...",
  "seniority": "...",
  "fit_score": 0,
  "summary": "..."
}

Always return valid JSON only.
""",
)

# 3) Resume Agent (recreated)
resume_agent = Agent(
    name="resume_agent",
    model="gemini-2.5-flash",
    description="Generates tailored resume bullets based on the JD analysis and user profile.",
    instruction="""
You are a Resume Tailoring Agent.

Input:
- User profile (skills, experience, achievements)
- JD analysis output (skills_required, responsibilities, domain, seniority)

Output:
- 4â€“8 resume bullet points tailored to the specific job.
- Bullets must be ATS-friendly.
- Each bullet should start with a strong verb.
- Highlight quantifiable achievements where possible.
- Map the user's experience to the JD requirements.

Return JSON:
{
  "tailored_resume_bullets": [
      "â€¦",
      "â€¦"
  ]
}

Always return valid JSON only.
""",
)

# 4) Outreach Agent
outreach_agent = Agent(
    name="outreach_agent",
    model="gemini-2.5-flash",
    description="Generates outreach email and LinkedIn message templates.",
    instruction="""
You are an Outreach Agent.

Input:
- User profile (JSON or natural language summary)
- Job information (company, title, key responsibilities)
- Optional: context such as status (planned/applied/interview) and preferred tone.

Your task:
- Generate two outreach templates:
  1) Email to recruiter / hiring manager
  2) LinkedIn connection / InMail message

Both should:
- Be concise and professional.
- Mention the role title and company explicitly.
- Highlight 1â€“3 relevant strengths from the user's profile.
- End with a polite call to action.

Return JSON in this exact format:
{
  "email_template": "full email text",
  "linkedin_template": "full LinkedIn message text"
}

Always return valid JSON only.
""",
)



from google import genai
from google.genai import types as genai_types

# Reuse or create a Gemini client
client = genai.Client()

def web_search_company(company: str) -> dict:
    """
    Simulated WebSearchTool: fetches a concise overview of a company.

    In a production setting, this would call a web search / company API.
    For this capstone, we use Gemini to generate a short, factual-style overview.
    """
    prompt = f"""
You are a company research assistant.

Provide a concise overview of the company "{company}" in 3-5 bullet points.
Include:
- Industry / domain
- Typical products or services
- 2-3 themes a candidate could mention when explaining why they want to work there.

Keep it under 120 words.
"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    # Safely extract text
    text = ""
    if resp.candidates:
        parts = resp.candidates[0].content.parts
        text = " ".join(
            [p.text for p in parts if hasattr(p, "text") and p.text]
        ).strip()

    return {
        "company": company,
        "overview": text,
    }



import asyncio
import nest_asyncio
nest_asyncio.apply()

from google.adk.runners import InMemoryRunner
from google.genai import types

APP_NAME = "career_concierge_app"
USER_ID = "user-1"
SESSION_ID = None   # will be set by create_session()

# Orchestrator agent with all sub-agents & tools
career_agent = Agent(
    name="career_concierge_agent",
    model="gemini-2.5-flash",
    description="Main orchestrator agent for career tasks.",
    instruction="""
You are the Orchestrator Agent for a multi-agent career concierge system.

You have access to:
- Tools:
  - add_job_application
  - list_job_applications
  - web_search_company  (simulated WebSearchTool)
- Sub-agents:
  - profile_agent: builds structured profiles from CV-like text.
  - jd_analyzer_agent: analyzes job descriptions and returns structured JSON.
  - resume_agent: generates tailored resume bullets in JSON.
  - outreach_agent: generates email and LinkedIn outreach templates.

General behavior:
- If the user shares a long description of their background / CV:
  - Delegate to profile_agent to build a JSON profile.
  - Summarize the profile in 2â€“4 bullets for future use.

- When the user provides a job description and asks for analysis or tailoring:
  1. Ensure you have a profile summary (ask or use existing).
  2. Delegate JD understanding to jd_analyzer_agent.
  3. Delegate resume tailoring to resume_agent,
     passing the profile summary and JD analysis as context.
  4. Optionally call web_search_company(company) to enrich the context.
  5. Combine outputs into a final response with headings:
     - ## Job Summary
     - ## Skills Required
     - ## Fit Score
     - ## Tailored Resume Bullets
     - ## Company Overview (if web_search_company used)

- When the user wants to save a job:
  - Extract title, company, optional link, and status.
  - Call add_job_application.
  - Confirm what you saved in a clear list.

- When the user wants to review applications:
  - Call list_job_applications.
  - Summarize results as a bullet list or mini table.

- When the user asks for outreach / recruiter email:
  - Use profile summary, JD analysis, and (optionally) web_search_company.
  - Delegate to outreach_agent.
  - Return the email and LinkedIn templates with a short explanation.

Always:
- Be explicit (in natural language) when you delegate to a sub-agent or call a tool.
- Keep responses concise but structured with headings and bullet points.
""",
    tools=[
        add_job_application,
        list_job_applications,
        web_search_company,
    ],
    sub_agents=[
        profile_agent,
        jd_analyzer_agent,
        resume_agent,
        outreach_agent,
    ],
)

# Runner using the orchestrator
runner = InMemoryRunner(
    agent=career_agent,
    app_name=APP_NAME,
)


def create_session():
    """
    Create a new ADK session via the runner's session_service and store its id.
    This uses the existing event loop (Colab-safe with nest_asyncio).
    """
    global SESSION_ID

    async def _create():
        return await runner.session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
        )

    loop = asyncio.get_event_loop()
    session = loop.run_until_complete(_create())
    SESSION_ID = session.id
    print("âœ… Created session:", SESSION_ID)


def run_agent(user_message: str):
    """
    Send a message to the orchestrator agent and print streamed response.
    Automatically creates a session the first time it's called.
    """
    global SESSION_ID
    if SESSION_ID is None:
        create_session()

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )
    print(f"ğŸ‘¤ User: {user_message}\n")

    for event in runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):
        if event.content and event.content.parts:
            text = event.content.parts[0].text or ""
            if text.strip():
                print(f"ğŸ¤– {event.author}: {text}\n")



test_jd = """
JPMorgan Chase is hiring a Senior Principal Data Architect in Dublin.
Responsibilities include designing enterprise data platforms, cloud migration,
collaborating with AI/ML engineers, and driving data governance.
Required skills: Python, cloud platforms, data architecture, ML understanding.
"""

run_agent(
    "Here is my profile: 10+ years in SAP CRM and Sales Cloud, "
    "plus AI/ML and DBA studies focused on data science.\n\n"
    "Please analyze this job and generate tailored resume bullets:\n"
    + test_jd
)

show_metrics()



test_jd = """
JPMorgan Chase is hiring a Senior Principal Data Architect in Dublin.
Responsibilities include designing enterprise data platforms, cloud migration,
collaborating with AI/ML engineers, and driving data governance.
Required skills: Python, cloud platforms, data architecture, ML understanding.
"""

run_agent(
    "Please analyze this job, check my fit, and generate tailored resume bullets:\n"
    + test_jd
)

show_metrics()



run_agent(
    "Now, based on my profile and that JPMorgan role, draft a recruiter email and a LinkedIn message "
    "I can send to the hiring manager."
)

# Now show metrics
show_metrics()


