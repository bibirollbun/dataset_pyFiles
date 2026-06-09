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


from google.genai import types

from google.adk.agents import Agent,LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor

print("âœ… ADK components imported successfully.")


# 1. Define the user's profile (the data the agents will use)
user_profile = {
    "name": "ABC",
    "title": "Data Scientist / ML Engineer",
    "skills": [
        "Python", "SQL", "Machine Learning", "Deep Learning",
        "NLP", "Pandas", "NumPy", "Scikit-learn", "TensorFlow",
        "PyTorch", "AWS", "Power BI", "Tableau"
    ],
    "experience_summary": (
        "MS in Data Science. Research Assistant working on "
        "AI agents, NLP, and regulatory data pipelines. Previously worked as a "
        "Software Engineer building ML and data-driven applications."
    ),
    "preferred_roles": ["Data Scientist", "Applied Scientist", "ML Engineer"],
    "tone": "friendly, concise, professional"
}

print("User profile created!")


from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()

print("InMemorySessionService initialized!")


from google.adk.tools import ToolContext

def extract_skills_from_text_fn(jd_text: str):
    known_skills = [
        "Python", "R", "SQL", "Java", "C++",
        "Machine Learning", "Deep Learning", "NLP",
        "Tableau", "Power BI", "AWS", "GCP", "Azure",
        "Spark", "Hadoop", "Docker", "Kubernetes", "Git"
    ]

    jd_lower = jd_text.lower()
    found = [skill for skill in known_skills if skill.lower() in jd_lower]

    return {"jd_skills": sorted(set(found))}



def compute_skill_match_fn(jd_skills: list, user_skills: list):
    jd_set = set(jd_skills)
    user_set = set(user_skills)

    overlap = sorted(jd_set & user_set)
    missing = sorted(jd_set - user_set)

    score = round(100 * len(overlap) / len(jd_set), 1) if jd_set else 0.0

    return {
        "match_score": score,
        "overlap_skills": overlap,
        "missing_skills": missing
    }


print("Tools (functions) created successfully!")


jd_analyst_agent = Agent(
    name="jd_analyst_agent",
    model=Gemini(model="gemini-2.0-flash"),
    instruction="""
You analyze job descriptions and extract structured data.

You MUST ALWAYS do the following:

=====================
STEP 1 â€” Summarize JD
=====================
Read the job description text (jd_text) and create:
- a concise jd_summary
- a list of responsibilities (bullet points)

=====================
STEP 2 â€” Extract Skills with the Tool
=====================
Call the tool extract_skills_from_text_fn using:

{
  "jd_text": <the full job description>
}

This tool returns a list in the field "jd_skills".
You MUST include this field in your final output as "skills".

=====================
FINAL OUTPUT (STRICT)
=====================
Return ONLY a JSON object:

{
  "jd_summary": "...",
  "responsibilities": [...],
  "skills": <jd_skills from tool>
}

Rules:
- NEVER invent skillsâ€”always rely on the tool.
- NEVER add extra fields.
- NEVER produce text outside the JSON.
"""
    ,
    tools=[extract_skills_from_text_fn],
)



company_research_agent = Agent(
    name="company_research_agent",
    model=Gemini(model="gemini-2.0-flash"),
    instruction="""
    You research companies using Google Search

    Return:
    - company summary
    - recent news
    - 2-3 talking points candidate can mention in outreach
    """,
    tools=[google_search],
)


gap_analyzer_agent = Agent(
    name="gap_analyzer_agent",
    model=Gemini(model="gemini-2.0-flash"),
    instruction=f"""
You are the Skill Gap Analysis Agent.

Your job:
1. Receive JD skills from jd_analyst_agent (field: jd_skills).
2. Compare them to the USER'S skills:

USER SKILLS:
{user_profile["skills"]}

3. You MUST call compute_skill_match_fn(jd_skills, user_skills)
   where:
   - jd_skills = list from jd_analyst_agent
   - user_skills = list above

4. Return EXACTLY this JSON:

{{
  "match_score": <number 0â€“100>,
  "overlap_skills": [...],
  "missing_skills": [...],
  "explanation": "2â€“3 sentence explanation summarizing strengths and missing areas."
}}

RULES:
- NEVER invent skills.
- NEVER output 'suggested_skills'.
- ONLY use output from compute_skill_match_fn.
""",
    tools=[compute_skill_match_fn],
)



outreach_writer_agent = Agent(
    name="outreach_writer_agent",
    model=Gemini(model="gemini-2.0-flash"),
    instruction="""
You write two things based on previous agent outputs:

Inputs available in the conversation:
- jd_analysis (role, skills, responsibilities)
- company_research (company_name, summary)
- skill_analysis (match_score, overlap_skills, missing_skills)
- user_profile (candidate background & skills)

Your tasks:

1. Write a personalized LinkedIn message addressed to a recruiter of the company
   detected in the job description.
   The message must:
   - Be <= 150 words.
   - Show genuine interest in the company's mission and work (based on company_research).
   - Highlight overlap between candidate skills and JD skills.
   - Acknowledge missing skills positively (e.g., willingness to learn).


Return JSON:

{
  "outreach_message": "<LinkedIn outreach message>",
}

Always use the company name dynamically from company_research.company_name.
Never hardcode any company.
"""
)



coordinator_agent = Agent(
    name="coordinator_agent",
    model=Gemini(model="gemini-2.0-flash"),
    instruction="""
You are the orchestrator of the job analysis workflow.

You MUST perform the following steps in ORDER:

1. Call jd_analyst_agent with:
   { "jd_text": <full job description> }

2. Extract company name:
   - If mentioned explicitly in the JD, use that.
   - Otherwise, infer it.
   Then call company_research_agent with:
   { "company_name": <name> }

3. Call gap_analyzer_agent with:
   { "jd_skills": <skills from jd_analyst_agent> }

4. Call outreach_writer_agent with ALL data:
   {
     "jd_analysis": ...,
     "company_research": ...,
     "skill_analysis": ...,
     "user_profile": ...
   }

=====================
STRICT FINAL OUTPUT
=====================

{
  "jd_analysis": {...},
  "company_research": {...},
  "skill_analysis": {...},
  "outreach_message": {...}
}

Rules:
- NEVER output anything outside the JSON.
- NEVER skip a step.
""",
    tools=[
        AgentTool(agent=jd_analyst_agent),
        AgentTool(agent=company_research_agent),
        AgentTool(agent=gap_analyzer_agent),
        AgentTool(agent=outreach_writer_agent),
    ],
)



session_service = InMemorySessionService()


runner = Runner(
    agent=coordinator_agent,
    app_name="career_app",
    session_service=session_service
)

print("Runner created successfully!")


job_text = """PhysicsX is a deep-tech company with roots in numerical physics and Formula One, dedicated to accelerating hardware innovation at the speed of software.
We are building an AI-driven simulation software stack for engineering and manufacturing across advanced industries. By enabling high-fidelity, multi-physics simulation through AI inference across the entire engineering lifecycle, PhysicsX unlocks new levels of optimization and automation in design, manufacturing, and operations â€” empowering engineers to push the boundaries of possibility. Our customers include leading innovators in Aerospace & Defense, Materials, Energy, Semiconductors, and Automotive.
Who We're Looking For
As a Data Scientist in Delivery, you are a problem solver and builder who is passionate about creating practical solutions that enable customers to make better engineering decisions. You are someone who can grasp advanced engineering concepts across multiple industries, and you excel at working directly with customers (and often side-by-side with them on-site) to transform cutting edge AI models into tools that are useful and used.
 
Youâ€™ve worked on difficult problems that require strong foundations in data driven modelling and deep learning techniques, with hands-on experience in probabilistic methods and predictive modelling. Expertise in python, along with proficiency in libraries like NumPy, SciPy, Pandas, TensorFlow and PyTorch, is essential, with the ability to deploy scalable, production-ready models and data pipelines.
 
With at least 1 year industry experience (post Masters or PhD) in a commercial, non-research environment, youâ€™re ready to hit the ground running. Youâ€™re truly excited about growing your technical expertise and are naturally inclined to take ownership of data science work streams, continuously improving the systems and solutions you work on to ensure they are practical, impactful and meet the evolving needs of our customers.
 
This Role
In this role, youâ€™ll work closely with our Simulation Engineers, Machine Learning Engineers, and customers to understand and define the engineering and physics challenges we are solving.
Youâ€™ll build the foundations for successful, impactful solutions by:
 
Pre-processing and analyzing data to prepare it for use in predictive modelling, building the foundation for machine learning algorithms to be developed.
Developing and utilizing innovative deep learning models in combination with state-of-the-art optimization methods to predict and control the behaviour of physical systems.
Taking full responsibility for the quality, accuracy and impact of your work.
Designing, building and testing data pipelines that are reliable, scalable and easily deployable in production environments.
Working closely with simulation engineers to ensure seamless integration of data science models with simulations.
Contributing to internal R&D and product development, helping to refine models and identify new areas of application.
Engaging in open communication and presentation with both technical teams and customers, helping onboard users and co-develop with customers.
There is a requirement to travel to customer sites in North America, Europe, Asia, Oceania, an average of 2-3 weeks per quarter, where youâ€™ll collaborate closely with customers to build solutions on site.
 
As the role evolves, there are exciting opportunities for growth as an individual contributor or a technical lead, especially if youâ€™re driven by taking ownership of more complex projects and leading the direction of future solutions.
Please note, this role is based in Manhattan, NYC, working 2-3 days per week in our office.
 
Our delivery teams drive innovation to turn AI models into practical solutions - read our blog to learn more about how youâ€™ll contribute to this exciting journey!
 
What we offer 
Equity options â€“ share in our success and growth.
5% 401(k) match â€“ invest in your future.
Flexible working â€“ balance your work and life in a way that works for you.
Hybrid setup â€“ enjoy our Manhattan office while keeping remote flexibility.
Enhanced parental leave â€“ support for lifeâ€™s biggest milestones.
Private healthcare â€“ comprehensive coverage for you and your family.
Personal development â€“ access learning and training to help you grow.
Work from anywhere â€“ extend your remote setup to enjoy the sun or reconnect with loved ones.
 
We believe diversity fuels innovation, and we're building a culture where everyone belongs. We're proud to be an equal opportunity employer, welcoming talent of all backgrounds, identities, and experiences. Changing the face of tech takes action, which is why we actively encourage individuals from historically underrepresented groups to apply.
 
Salary Range 
 
$120,000 - 240,000 depending on experience 
Seniority will be assessed throughout our interview process 
 
We value diversity and are committed to equal employment opportunity regardless of sex, race, religion, ethnicity, nationality, disability, age, sexual orientation or gender identity. We strongly encourage individuals from groups traditionally underrepresented in tech to apply. To help make a change, we sponsor bright women from disadvantaged backgrounds through their university degrees in science and mathematics. 
 
We collect diversity and inclusion data solely for the purpose of monitoring the effectiveness of our equal opportunities policies and ensuring compliance with UK employment and equality legislation. This information is confidential, used only in aggregate form, and will not influence the outcome of your application. 
"""



from google.genai import types


session = await session_service.create_session(
    app_name="career_app",
    user_id="user123"
)

query = {
    "job_description": job_text,
    "user_profile": user_profile
}



user_message = types.Content(
    role="user",
    parts=[types.Part(text=job_text)]
)

print("Running multi-agent workflow...\n")


events = runner.run_async(
    user_id="user123",
    session_id=session.id,
    new_message=user_message
)

final_response = None

async for event in events:
    if event.is_final_response():
        final_response = event.content.parts[0].text

print("FINAL OUTPUT:")
print(final_response)


