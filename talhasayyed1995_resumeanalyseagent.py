import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


import os
import json
# import subprocess
# import time
# import uuid
import requests
from typing import List, Dict, Any

from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
# from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
# from google.adk.a2a.utils.agent_to_a2a import to_a2a
# from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
# from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools.tool_context import ToolContext
# from google.adk.tools.function_tool import FunctionTool
from google.genai import types

print("All import Done")



retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)
print("retry_config created!\n")


# Initialize Session Service for the client Runner
session_service = InMemorySessionService()
memory_service = (
    InMemoryMemoryService()
)  # ADK's built-in Memory Service for development and testing
print('session_service created!')
print('memory_service created!')


cv_content = """
# **AARAV KAPOOR**

Email: [aarav.kapoor.dev@example.com](mailto:aarav.kapoor.dev@example.com)
Phone: +91 98765 43210
Location: Bengaluru, India
GitHub: github.com/aaravk-dev
LinkedIn: linkedin.com/in/aaravk-dev

---

## **PROFESSIONAL SUMMARY**

Detail-oriented Python Software Developer with 3+ years of experience building scalable backend systems, automation tools, and AI-driven workflows. Strong focus on clean code, API development, GenAI integrations, and end-to-end workflow automation. Experienced working with Flask, FastAPI, Docker, and modern model-context agent frameworks. Passionate about creating efficient, reliable, and maintainable solutions.

---

## **SKILLS**

### **Programming & Backend**

* Python (Core, OOP, Asyncio)
* Flask, FastAPI, Django (basic)
* REST API design, webhooks
* SQL, PostgreSQL, MongoDB
* Celery, Redis
* Pandas, NumPy

### **DevOps & Tools**

* Git, GitHub Actions, CI/CD
* Docker, Linux, Nginx
* Bash scripting

### **GenAI, Agents & Automation**

* LangChain, LangGraph (basic)
* OpenAI / Azure OpenAI APIs
* RAG Pipelines (FAISS, Chroma)
* Prompt Engineering
* Lightweight Agent Systems (model-context workflows)
* LLM-powered automation & GPT-based task orchestration

### **Others**

* Unit testing (pytest)
* Agile/Scrum
* System design basics

---

## **PROFESSIONAL EXPERIENCE**

### **Python Software Developer**

**TechNova Systems Pvt. Ltd., Bengaluru**
**June 2022 â€“ Present**

* Developed and maintained backend services using **Flask** and **FastAPI**, improving API response times by ~30%.
* Built internal automation tools using **Python** to reduce manual workload, cutting processing time by 40%.
* Integrated **OpenAI APIs** and **LangChain** to build intelligent summarization and classification pipelines.
* Created a mini RAG workflow using **FAISS** to support internal knowledge-base search.
* Implemented **CI/CD pipelines** using GitHub Actions and Docker-based deploys.
* Worked with cross-functional teams to design scalable backend architecture for new features.
* Wrote unit tests and improved code coverage to 85% using **pytest**.

---

## **PROJECTS**

### **AI-Powered Document Assistant**

* Built a document Q&A assistant using **LangChain**, **FAISS**, and **Azure OpenAI**.
* Supported PDF ingestion, chunking, embeddings, and query answering.
* Reduced document lookup time from minutes to seconds.

### **Automation Engine for Internal Teams**

* Designed Python scripts to automate repetitive reporting tasks.
* Implemented scheduling & logging with Celery + Redis.
* Saved 6+ hours/week for the operations team.

### **FastAPI Microservice for User Analytics**

* Built a lightweight analytics service using FastAPI + PostgreSQL.
* Implemented `/metrics` endpoints for client applications.
* Deployed using Docker + Nginx reverse proxy.

---

## **EDUCATION**

**B.Tech in Computer Science & Engineering**
Indian Institute of Information Technology (IIIT), Pune
2018 â€“ 2022

---

## **CERTIFICATIONS**

* Python for Everybody â€“ University of Michigan
* Azure OpenAI Developer Fundamentals
* Google Cloud Associate Engineer (In progress)

---

## **PERSONAL PROJECTS**

* **GitHub Bot using GPT Models** â€“ a bot that reviews PR descriptions and suggests fixes.
* **Local AI Playground** â€“ experimented with running models via Ollama and local embeddings.

---

## **INTERESTS**

* Open-source contributions
* AI automation and agent design
* Performance optimization in Python
"""


# Save to file
with open('/kaggle/working/resume_data.txt', 'w') as file:
    file.write(cv_content)

print("file created")


import ipywidgets as widgets
from IPython.display import display
import os

# Create text area widget
cv_textarea = widgets.Textarea(
    value='',
    placeholder='Enter your full CV data here...',
    description='CV Data:',
    layout=widgets.Layout(width='100%', height='300px')
)

# Create save button
save_button = widgets.Button(description="Save CV", button_style='success')

# Output widget for messages
output = widgets.Output()

def on_save_button_clicked(b):
    with output:
        output.clear_output()
        if cv_textarea.value.strip():
            # Save to file
            with open('/kaggle/working/resume_data.txt', 'w') as file:
                file.write(cv_textarea.value)
            
            # Verify file was created
            if os.path.exists('/kaggle/working/resume_data.txt'):
                print("âœ… CV successfully saved as 'resume_data.txt'")
                print(f"ğŸ“Š Characters saved: {len(cv_textarea.value)}")
            else:
                print("â�Œ Error: File was not created")
        else:
            print("âš ï¸� Please enter some CV data before saving")





# save_button.on_click(on_save_button_clicked)
# # Display widgets
# print("Enter your CV data in the text box below and click 'Save CV':")
# display(cv_textarea)
# display(save_button)
# display(output)


# Read the entire file content
try:
    with open('/kaggle/working/resume_data.txt', 'r') as file:
        cv_content = file.read()
    
    print("âœ… CV data successfully loaded!")
    print("=" * 50)
    print("CV CONTENT:")
    print("=" * 50)
    print(cv_content[:100])
    
except FileNotFoundError:
    print("â�Œ File 'resume_data.txt' not found. Please make sure the file exists in /kaggle/working/")
except Exception as e:
    print(f"â�Œ Error reading file: {str(e)}")


JD_list = [
    "Python Software Engineer - Backend Focus: We are seeking a skilled Python Software Engineer to join our backend development team. The ideal candidate will have 3+ years of experience building scalable web applications using Python and Django/Flask frameworks. Responsibilities include designing and implementing RESTful APIs, optimizing database performance with PostgreSQL, developing microservices architecture, and writing comprehensive unit tests. You'll work on our core platform, handling high-volume data processing and real-time analytics. Required skills: Python, Django, REST APIs, PostgreSQL, Docker, AWS. Bonus points for experience with Celery, Redis, Kubernetes, and message queue systems. This role offers the opportunity to solve complex distributed systems problems in a fast-growing tech environment.",
    
    "Python Software Engineer - Data & ML Platform: Join our data engineering team as a Python Software Engineer focused on building robust data pipelines and machine learning infrastructure. You'll be responsible for developing ETL processes, implementing data validation frameworks, and creating scalable data processing systems using PySpark and Pandas. The role involves working closely with data scientists to productionize ML models, building monitoring tools for data quality, and optimizing data storage solutions. Requirements include 4+ years of Python development experience, strong knowledge of SQL, and familiarity with big data technologies. Preferred qualifications: experience with Airflow, MLflow, Scikit-learn, TensorFlow/PyTorch, cloud data warehouses (Snowflake/BigQuery), and distributed computing systems.",
    
    "Python Software Engineer - Full Stack Development: We're looking for a versatile Python Software Engineer who can contribute across our full stack. In this role, you'll develop backend services with FastAPI, create responsive frontend components with React/TypeScript, and build internal tools that enhance team productivity. Daily tasks include feature development across our web application, performance optimization, code reviews, and collaborating with product managers on technical specifications. The perfect candidate has 2+ years of Python experience plus modern JavaScript framework knowledge. You should be comfortable working in agile environments, have experience with CI/CD pipelines, and understand software design patterns. This position offers exposure to the complete software development lifecycle in a product-driven company."
  ]
print("Job Descriptions loaded")


# Agent 1: JD Score Finder Agent 
# JD Score Finder Agent: Its job is to find the similarity score between JD and Resume.
jdscore_calculator_agent = Agent(
    name="JdScoreCalculatorAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
Compare the resume and job description. Provide ONLY a match score (0-10) as a string.
Do not include any other text or explanation.
""",
    output_key="match_score",  # Changed to match your desired JSON key
)
print("âœ… jdscore_calculator_agent agents created.")


# Agent 2: Resume Bullet Agent
# Resume Bullet Agnet: Its job is to missing to point to be added in Resume.
resume_bullet_agent = Agent(
    name="ResumeBulletAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
Compare the Resume with JD and suggest 3-4 specific bullet points to add to the Resume.
Focus on missing skills and experiences mentioned in the JD.
Format as a bulleted list.
""",
    output_key="resume_bullets",
)
print("âœ… resume_bullet_agent agents created.")


# Agent 3: Cover Letter Agent
cover_letter_agent = Agent(
    name="CoverLetterAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
Create a professional custom cover letter that highlights the candidate's relevant 
skills and addresses key requirements from the job description.
Keep it concise and tailored to the specific role.
""",
    output_key="custom_cover_letter",
)
print("âœ… cover_letter_agent agents created.")


# Agent 4: Job Info Extractor Agent (NEW - to extract job_title and company)
job_info_agent = Agent(
    name="JobInfoAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
Extract the job title and company name from the job description.
Return as a simple JSON: {"job_title": "...", "company": "..."}
""",
    output_key="job_info",
)
print("âœ… job_info_agent agents created.")
print("âœ… All individual agents created.")


# Create Parallel Agent with all sub-agents
parallel_analysis_team = ParallelAgent(
    name="ParallelResumeAnalysisTeam",
    sub_agents=[
        jdscore_calculator_agent, 
        resume_bullet_agent, 
        cover_letter_agent,
        job_info_agent  # Added to get job_title and company
    ],
)
print("âœ… parallel_analysis_team created.")

print("create AgentTool from ParallelAgent")

# Wrap the parallel agent as a tool
parallel_tool = AgentTool(parallel_analysis_team)
print("âœ… parallel_tool created.")



# Root Agent that uses the parallel team
root_agent = Agent(
    name="ResumeAnalyzerRootAgent",
    model="gemini-2.5-flash-lite",
    instruction="""You are a comprehensive CV analysis system. 

Use the `ParallelResumeAnalysisTeam` tool to analyze the resume and job description in parallel.

After receiving all the parallel results, compile them into a final JSON response with this exact structure:
{
    "job_title": "extracted job title from JD",
    "company": "extracted company name from JD", 
    "match_score": "score from 0-10",
    "resume_bullets": "suggested bullet points",
    "custom_cover_letter": "generated cover letter"
}

Make sure the output is valid JSON format.
""",
    tools=[parallel_tool],  # Use the wrapped parallel agent tool
)

print("âœ… root_agent created successfully!")



# Correct way to create and run the runner
runner = Runner(
    agent=root_agent,  # You need to pass the agent here
    app_name="ResumeAnalyzerApp",
    session_service=session_service,
    memory_service=memory_service
)
print("âœ… Runner created.")


# After running the agent
# response = await runner.run_debug(root_prompt)

import json

# show formatted response and write in dictionary
def response_formatter(response, formatted_data_list):
    
    # Extract the final output from the response
    if hasattr(response, 'final_output') and response.final_output:
        print("Final Result:", response.final_output)
    else:
        # If final_output is empty, let's extract from the state delta
        for event in response:
            if hasattr(event, 'actions') and event.actions and event.actions.state_delta:
                state_delta = event.actions.state_delta
                # print("State Delta:", state_delta)
                
                # Check if we have the compiled JSON in the state
                if any(key in state_delta for key in ['job_title', 'company', 'match_score', 'resume_bullets', 'custom_cover_letter']):
                    # Extract individual components
                    compiled_result = {}
                    job_info_str = dict(state_delta).get('job_info').replace("```json", "").replace("```", "").strip()
                    job_info = json.loads(job_info_str)
                    compiled_result['job_title'] = job_info.get('job_title', 'Not found')
                    compiled_result['company'] = job_info.get('company', 'Not found')
                    compiled_result['match_score'] = state_delta.get('match_score', 'Not found')
                    compiled_result['resume_bullets'] = state_delta.get('resume_bullets', 'Not found')
                    compiled_result['custom_cover_letter'] = state_delta.get('custom_cover_letter', 'Not found')
                    
                    print("Compiled JSON Result:")
                    print('-'*30)
                    print(json.dumps(compiled_result, indent=2))
                    print('-'*30)
                    formatted_data_list.append(compiled_result)
                else:
                    print('-'*30)
                    # Extract from individual agent outputs
                    print("Individual Agent Outputs:")
                    print(f"Match Score: {state_delta.get('match_score', 'Not found')}")
                    print(f"Job Info: {state_delta.get('job_info', 'Not found')}")
                    print(f"Resume Bullets: {state_delta.get('resume_bullets', 'Not found')}")
                    print(f"Cover Letter: {state_delta.get('custom_cover_letter', 'Not found')}")
                    print('-'*30)
                print('\n\n')
    return formatted_data_list
print("response_formatter created")


# Your existing prompt
root_prompt = f"""Analyze this Resume and JD to give insights:

Resume
---
{cv_content}
---

Job Description
---
{JD_list[0]}
---
"""
# prompt for single response
response = await runner.run_debug(root_prompt)


testdata = response_formatter(response, [])


result_list = []
for jd in JD_list[:-1]:
    root_prompt = f"""Analyze this Resume and JD to give insights:
    
    Resume
    ---
    {cv_content}
    ---
    
    Job Description
    ---
    {jd}
    ---
    """
    # prompt for single response
    response = await runner.run_debug(root_prompt)
    result = response_formatter(response, result_list)

    


# from pprint import pprint
for result in result_list:
    print(result)
    print()


result_list[0].get('job_title')


result_list[0].get('company')


result_list[0].get('match_score')


for line in result_list[0].get('resume_bullets').splitlines():
    print(line)




