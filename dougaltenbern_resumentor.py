%%writefile resumentor/agent.py

import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

import json
import requests
import subprocess
import time
import uuid

from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent, Memory, Tool

from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types, configure
from google.adk.tools import AgentTool, FunctionTool, google_search

# Hide additional warnings in the notebook
import warnings

warnings.filterwarnings("ignore")

print("âœ… ADK components imported successfully.")

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# Tools

shared_memory = Memory()

memory_service = (
    InMemoryMemoryService()
)  # ADK's built-in Memory Service for development and testing

# Define constants used throughout the notebook
APP_NAME = "resumentor"
USER_ID = "user"

# Create Session Service
session_service = InMemorySessionService()  # Handles conversations

# Create runner with BOTH services
runner = Runner(
    agent=user_agent,
    app_name="resumentor",
    session_service=session_service,
    memory_service=memory_service,  # Memory service is now available!
)

print("âœ… Agent and Runner created with memory support!")

# session = await session_service.get_session(
#     app_name=APP_NAME, user_id=USER_ID, session_id="conversation-01"
# )

# Let's see what's in the session
# print("ğŸ“� Session contains:")
# for event in session.events:
#     text = (
#         event.content.parts[0].text[:60]
#         if event.content and event.content.parts
#         else "(empty)"
#     )
#     print(f"  {event.content.role}: {text}...")
    
def save_resume(content: str):
    shared_memory.set("generated_resume", content)
    return "Resume saved."

def generate_pdf(filename: str = "resume.pdf"):
    resume_md = shared_memory.get("final_resume") or shared_memory.get("generated_resume")
    if not resume_md:
        return "No resume available."

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 40

    for line in resume_content.split("\n"):
        wrapped = textwrap.wrap(line, 100)
        for w in wrapped:
            c.drawString(30, y, w)
            y -= 12
            if y < 40:
                c.showPage()
                y = height - 40
    c.save()
    return f"PDF saved as {filename}"

# Resume research specialized agent
resume_research_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-pro", retry_options=retry_config),
    name="resume_research_agent",
    description="Agent to research general best practices for resume creation",
    instruction="""
    You are a resume research specialist. You are to research general best practices for resumes. 
    Include layout, tone, structure, sections, writing style, and ATS compatibility.
    Save findings in memory under key 'general_best_practices'
    Be professional and helpful.
    """,
    tools=[google_search],
    output_key="general_best_practices",
    tools=[load_memory]
)

print("âœ… Resume Research Agent created successfully!")
print("âœ… Model: gemini-2.5-pro")

# Industry-specific research agent

industry_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-pro", retry_options=retry_config),
    name="industry_agent",
    description="Agent to research industry-specific best practices for resume creation",
    instruction="""
    Ask the user for job titles or industries they want a resume for.
    Then research industry-specific resume standards, keywords,
    recommended skills, and hiring manager preferences.
    Save results into memory under key 'industry_best_practices'.
    """,
    output_key="industry_best_practices",
    tools=[load_memory]
)

print("âœ… Industry Research Agent created successfully!")
print("âœ… Model: gemini-2.5-pro")

# Interview agent

interview_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="interview_agent",
    description="Agent to ask relevant questions for resume creation",
    instruction="""
    Interview the user for all information necessary to build a resume:
    - job history
    - education
    - certifications
    - achievements
    - technical + soft skills
    - contact information
    - project descriptions (if relevant)
    
    Store all user responses in memory under 'user_profile'.
    Ask follow-up questions if details are missing.
    """,
    output_key="user_profile",
    tools=[load_memory]
)

print("âœ… Interview Agent created successfully!")
print("âœ… Model: gemini-2.5-flash-lite")

# Resume writer agent

resume_writer_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="resume_writer_agent",
    description="Agent to create a resume from gathered information",
    instruction="""
    Using:
    - general_best_practices
    - industry_best_practices
    - user_profile

    Generate a clean, professional, ATS-optimized resume.
    Include:
    - Header with contact info
    - Summary
    - Skills (ATS-friendly)
    - Professional experience
    - Education
    - Certifications
    - Projects (if provided)

    Save final resume Markdown text under key 'generated_resume'.
    """,
    output_key="generated_resume",
    tools=[load_memory]
)

print("âœ… Resume Writer Agent created successfully!")
print("âœ… Model: gemini-2.5-flash-lite")

# Resume QA agent

qa_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="qa_agent",
    description="Agent to check resume for mistakes and add user-suggested changes",
    instruction="""
    Retrieve 'generated_resume' from memory.
    Proofread for:
      - grammar
      - spelling
      - clarity
      - formatting
      - ATS risks
      - missing sections

    Suggest improvements and ask the user if they want changes.
    After user feedback, update the resume and save under key 'final_resume'.
    """,
    output_key="final_resume",
    tools=[load_memory]
)

print("âœ… Resume QA Agent created successfully!")
print("âœ… Model: gemini-2.5-flash-lite")

pdf_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="pdf_agent",
    description="Agent to create the resume in PDF format",
    instruction="""
    Call the tool generate_pdf() to produce a PDF of the final resume.
    """,
    tools=[generate_pdf, load_memory],
)

print("âœ… Resume PDF Agent created successfully!")
print("âœ… Model: gemini-2.5-flash-lite")

# Coordinator

root_agent = Agent(
    name="AgentCoordinator",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
     instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.
1. First, you MUST call the `resume_research_agent` tool to research resume best practices.
2. While the `resume_research_agent` is running, you MUST call the `industry_agent` tool to search for industry-specific information.
3. After calling the `industry_agent` tool you MUST call the `interview_agent` tool.
4. Once the `resume_research_agent`, `industry_agent`, and `interview_agent` tools have finished running, you MUST call the `resume_writer_agent` tool to create the resume.
5. Once the `resume_writer_agent` tool has finished running, you MUST call the `qa_agent` tool.
6. Finally, once all other tools have finished running, you MUST call the `pdf_agent` tool
""",
    # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
    tools=[AgentTool(resume_research_agent), AgentTool(industry_agent), AgentTool(interview_agent), AgentTool(resume_writer_agent), AgentTool(qa_agent), AgentTool(pdf_agent), load_memory],
)

print("âœ… root_agent created.")

# await memory_service.add_session_to_memory(session)

# print("âœ… Session added to memory!")


# import json
# import requests
# import subprocess
# import time
# import uuid

# from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent

# from google.adk.models.google_llm import Gemini
# from google.adk.runners import Runner, InMemoryRunner
# from google.adk.sessions import InMemorySessionService
# from google.genai import types
# from google.adk.tools import AgentTool, FunctionTool, google_search

# # Hide additional warnings in the notebook
# import warnings

# warnings.filterwarnings("ignore")

# print("âœ… ADK components imported successfully.")



# retry_config = types.HttpRetryOptions(
#     attempts=5,  # Maximum retry attempts
#     exp_base=7,  # Delay multiplier
#     initial_delay=1,
#     http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
# )


# # Tools

# def save_resume(content: str):
#     shared_memory["generated_resume"] = content
#     return "Resume saved."

# def generate_pdf(filename: str = "resume.pdf"):
#     resume_content = shared_memory.get("final_resume") or shared_memory.get("generated_resume")
#     if not resume_content:
#         return "No resume available."

#     c = canvas.Canvas(filename, pagesize=letter)
#     width, height = letter
#     y = height - 40

#     for line in resume_content.split("\n"):
#         wrapped = textwrap.wrap(line, 100)
#         for w in wrapped:
#             c.drawString(30, y, w)
#             y -= 12
#             if y < 40:
#                 c.showPage()
#                 y = height - 40
#     c.save()
#     return f"PDF saved as {filename}"



# # Resume research specialized agent
# resume_research_agent = LlmAgent(
#     model=Gemini(model="gemini-2.5-pro", retry_options=retry_config),
#     name="resume_research_agent",
#     description="Agent to research general best practices for resume creation",
#     instruction="""
#     You are a resume research specialist. You are to research general best practices for resumes. 
#     Include layout, tone, structure, sections, writing style, and ATS compatibility.
#     Save findings in memory under key 'general_best_practices'
#     Be professional and helpful.
#     """,
#     tools=[google_search],
#     output_key="general_best_practices"
# )

# print("âœ… Resume Research Agent created successfully!")
# print("âœ… Model: gemini-2.5-pro")


# # Industry-specific research agent

# industry_agent = LlmAgent(
#     model=Gemini(model="gemini-2.5-pro", retry_options=retry_config),
#     name="industry_agent",
#     description="Agent to research industry-specific best practices for resume creation",
#     instruction="""
#     Ask the user for job titles or industries they want a resume for.
#     Then research industry-specific resume standards, keywords,
#     recommended skills, and hiring manager preferences.
#     Save results into memory under key 'industry_best_practices'.
#     """,
#     output_key="industry_best_practices"
# )

# print("âœ… Industry Research Agent created successfully!")
# print("âœ… Model: gemini-2.5-pro")


# # Interview agent

# interview_agent = LlmAgent(
#     model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
#     name="interview_agent",
#     description="Agent to ask relevant questions for resume creation",
#     instruction="""
#     Interview the user for all information necessary to build a resume:
#     - job history
#     - education
#     - certifications
#     - achievements
#     - technical + soft skills
#     - contact information
#     - project descriptions (if relevant)
    
#     Store all user responses in memory under 'user_profile'.
#     Ask follow-up questions if details are missing.
#     """,
#     output_key="user_profile"
# )

# print("âœ… Interview Agent created successfully!")
# print("âœ… Model: gemini-2.5-flash-lite")


# # Resume writer agent

# resume_writer_agent = LlmAgent(
#     model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
#     name="resume_writer_agent",
#     description="Agent to create a resume from gathered information",
#     instruction="""
#     Using:
#     - general_best_practices
#     - industry_best_practices
#     - user_profile

#     Generate a clean, professional, ATS-optimized resume.
#     Include:
#     - Header with contact info
#     - Summary
#     - Skills (ATS-friendly)
#     - Professional experience
#     - Education
#     - Certifications
#     - Projects (if provided)

#     Save final resume Markdown text under key 'generated_resume'.
#     """,
# )

# print("âœ… Resume Writer Agent created successfully!")
# print("âœ… Model: gemini-2.5-flash-lite")


# # Resume QA agent

# qa_agent = LlmAgent(
#     model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
#     name="qa_agent",
#     description="Agent to check resume for mistakes and add user-suggested changes",
#     instruction="""
#     Retrieve 'generated_resume' from memory.
#     Proofread for:
#       - grammar
#       - spelling
#       - clarity
#       - formatting
#       - ATS risks
#       - missing sections

#     Suggest improvements and ask the user if they want changes.
#     After user feedback, update the resume and save under key 'final_resume'.
#     """,
# )

# print("âœ… Resume QA Agent created successfully!")
# print("âœ… Model: gemini-2.5-flash-lite")


# pdf_agent = LlmAgent(
#     model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
#     name="pdf_agent",
#     description="Agent to create the resume in PDF format",
#     instruction="""
#     Call the tool generate_pdf() to produce a PDF of the final resume.
#     """,
#     tools=[generate_pdf],
# )

# print("âœ… Resume PDF Agent created successfully!")
# print("âœ… Model: gemini-2.5-flash-lite")


# # Coordinator

# root_agent = Agent(
#     name="AgentCoordinator",
#     model=Gemini(
#         model="gemini-2.5-flash-lite",
#         retry_options=retry_config
#     ),
#      instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.
# 1. First, you MUST call the `resume_research_agent` tool to research resume best practices.
# 2. While the `resume_research_agent` is running, you MUST call the `industry_agent` tool to search for industry-specific information.
# 3. After calling the `industry_agent` tool you MUST call the `interview_agent` tool.
# 4. Once the `resume_research_agent`, `industry_agent`, and `interview_agent` tools have finished running, you MUST call the `resume_writer_agent` tool to create the resume.
# 5. Once the `resume_writer_agent` tool has finished running, you MUST call the `qa_agent` tool.
# 6. Finally, once all other tools have finished running, you MUST call the `pdf_agent` tool
# """,
#     # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
#     tools=[AgentTool(resume_research_agent), AgentTool(industry_agent), AgentTool(interview_agent), AgentTool(resume_writer_agent), AgentTool(qa_agent), AgentTool(pdf_agent)],
# )

# print("âœ… root_agent created.")


# Define helper functions that will be reused throughout the notebook

from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")




url_prefix = get_adk_proxy_url()


!adk web --url_prefix {url_prefix}

