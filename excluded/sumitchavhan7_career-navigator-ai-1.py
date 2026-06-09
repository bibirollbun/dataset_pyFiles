# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# -----------------------------
import json
import math
from typing import List, Dict, Any, Tuple
from pprint import pprint
import random
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import json
import re
import asyncio
from typing import List, Dict, Any

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool, google_search, AgentTool
from google.adk.runners import InMemoryRunner
from google.genai import types


try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except ImportError:
    print("âš ï¸� Kaggle Secrets not available. Ensure you're in a Kaggle Notebook.")
except KeyError:
    print("ğŸ”‘ Authentication Error: Add 'GOOGLE_API_KEY' to Kaggle secrets.")

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)


!pip install google-adk>=1.18 --force-reinstall --quiet --upgrade


def load_user_profile(dataset_path: str) -> Dict[str, Any]:
    """
    Loads a user profile from a text file and extracts labeled sections.
    Now includes 'Job Location' field.
    """
    profile = {
        "preferences": "",
        "dietary_restrictions": "",
        "travel_destinations": "",
        "budget": "",
        "location": "",
        "job_location": ""
    }

    try:
        with open(dataset_path, "r", encoding="utf-8") as file:
            raw_text = file.read()

        # Helper: extract text after each label until the next label appears
        def extract_section(label: str) -> str:
            pattern = rf"{label}:\s*(.*?)(?=\n[A-Za-z ]+:|$)"
            match = re.search(pattern, raw_text, re.DOTALL)
            return match.group(1).strip() if match else ""

        # Extract all fields including Job Location
        for key, label in {
            "preferences": "User Preferences",
            "dietary_restrictions": "Dietary Restrictions",
            "travel_destinations": "Travel Destinations",
            "budget": "Budget",
            "location": "Location",
            "job_location": "Job Location"
        }.items():
            profile[key] = extract_section(label)

        print("âœ¨ Profile Loaded Successfully (Job Location Included)")

    except FileNotFoundError:
        print(f"â�Œ File not found: {dataset_path}")
    except Exception as error:
        print(f"â�Œ Unexpected error while loading profile: {error}")
        raise

    return profile


# === Run Loader === #
DATASET_PATH = "/kaggle/input/agents-intensive-capstone-project/Hackathon dataset.txt"
user_profile = load_user_profile(DATASET_PATH)

print("\nğŸ“Œ Loaded User Profile:\n", user_profile)



def get_job_recommendation(skills: str, job_location: str = None) -> str:
    """
    Recommend job roles based on skills and (optional) preferred location.
    """
    skills_lower = skills.lower()

    role_map = {
        "python": "Python Developer",
        "data": "Data Analyst",
        "ml": "Machine Learning Engineer",
        "frontend": "Frontend Developer",
        "backend": "Backend Developer",
        "cloud": "Cloud Engineer",
        "sql": "Database Engineer",
    }

    recommended = "Job Role Not Found"
    for key, value in role_map.items():
        if key in skills_lower:
            recommended = value
            break

    response = f"ğŸ”� Recommended Job Role: **{recommended}**\n"
    response += f"ğŸ§  Based on skills: {skills}"

    if job_location:
        response += f"\nğŸ“� Preferred Location: {job_location}"

    return response


def find_companies(location: str, job_role: str) -> str:
    """
    Suggest companies hiring in a given location for a specific job role.
    """
    location_lower = location.lower()

    companies_db = {
        "mumbai": ["TCS", "Accenture", "CitiusTech", "LTI"],
        "pune": ["Infosys", "Persistent", "Wipro", "Tech Mahindra"],
        "bangalore": ["Google", "Microsoft", "Flipkart", "Swiggy"],
        "delhi": ["Deloitte", "EY", "KPMG", "HCL"],
    }

    selected_companies = companies_db.get(location_lower, ["Local Startups", "Nearby IT Hubs"])
    company_list = ", ".join(selected_companies)

    return (
        f"ğŸ�¢ Companies hiring for **{job_role}** in **{location}**:\n"
        f"{company_list}\n"
        "âœ” Many roles available on Naukri / LinkedIn."
    )


def build_job_itinerary(job_role: str, salary_range: str, preferences: str = None) -> str:
    """
    Creates a job search plan (itinerary) like a Job Hunt Roadmap.
    """
    steps = [
        "Update resume & LinkedIn profile",
        "Apply to 10â€“15 relevant openings daily",
        "Target companies matching your role",
        "Practice interviews for 45â€“60 minutes",
        "Track responses & schedule interviews"
    ]

    journey = " â†’ ".join(steps)

    response = (
        f"ğŸ§­ **Job Search Plan for {job_role}**\n"
        f"ğŸ’° Expected Salary Range: {salary_range}\n"
        f"ğŸ“Œ Step-by-Step Roadmap: {journey}"
    )

    if preferences:
        response += f"\nğŸ�¯ Preferences Considered: {preferences}"

    response += "\nğŸš€ Your job hunting journey is ready!"

    return response



# --- JOB AGENTS SETUP ---

# 1. Recommend job roles based on skills
job_role_agent = Agent(
    name="JobRoleAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Expert job role recommender based on skills and user background.",
    tools=[FunctionTool(get_job_recommendation)],
    output_key="job_recommendation"
)

# 2. Find companies hiring in a given location
company_finder_agent = Agent(
    name="CompanyFinderAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Find companies hiring in a specific location for a given role.",
    tools=[FunctionTool(find_companies)],
    output_key="company_list"
)

# 3. Build job search plan / roadmap
job_planner_agent = Agent(
    name="JobPlannerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Career advisor that creates job search strategies and interview preparation plans.",
    tools=[FunctionTool(build_job_itinerary)],
    output_key="job_plan"
)

# 4. Master agent combining all job sub-agents
career_planner_agent = Agent(
    name="CareerPlannerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Expert career concierge agent. Coordinates job role recommendations, company search, and job roadmap.",
    tools=[
        AgentTool(job_role_agent),
        AgentTool(company_finder_agent),
        AgentTool(job_planner_agent)
    ],
    output_key="final_response"
)

# 5. Root agent (Sequential Execution)
root_agent = SequentialAgent(
    name="CareerAssistantAgent",
    sub_agents=[career_planner_agent]
)



# Runner for Job / Career Assistant
runner = InMemoryRunner(agent=root_agent)
print("ğŸ’¼ Job Career Assistant Runner Initialized Successfully")


async def run_career_assistant(user_request: str):
    # Build session state for job-related data
    session_state = {
        "skills": user_profile.get("skills", ""),
        "preferred_job_location": user_profile.get("job_location", ""),
        "experience": user_profile.get("experience", ""),
        "expected_salary": user_profile.get("expected_salary", ""),
        "preferred_roles": user_profile.get("preferred_roles", ""),
    }

    print("\nSession State:", session_state)

    # Calling the career job agent
    response = await runner.run_debug({
        "user_request": user_request,
        "session_state": session_state
    })

    print("Final Response:", response)



async def main():
    print("ğŸš€ Starting Career Assistant...")

    # Example job-related queries
    await run_career_assistant("I have skills in Python, SQL, and ML. What job roles suit me?")
    await run_career_assistant("Find companies hiring Data Analysts in Bangalore.")
    await run_career_assistant("Create a job search plan for me with a salary range of 8-12 LPA.")

    print("âœ… Career Assistant session completed.")


import asyncio

def run_async(coro):
    """
    Run an async coroutine for the Career/Job Assistant system.
    Handles nested event loops (e.g., in Jupyter/Colab environments).
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Fix for environments where the event loop is already running
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(coro)



run_async(main())


from typing import List, Dict, Any

class CareerA2A:
    """
    Orchestrates multiple job/career agents in a conversational flow
    using the CareerPlanner SequentialAgent setup.
    """
    def __init__(self, runner: InMemoryRunner):
        self.runner = runner

    async def handle_request(self, request: str, session_state: Dict[str, Any]):
        """
        Sends user requests through the Career/Job agent system and returns responses.
        """
        print(f"\nğŸ’¬ User Request: {request}")
        response = await self.runner.run_debug({
            "user_request": request,
            "session_state": session_state
        })
        print(f"ğŸ¤– Career Assistant Response: {response}")
        return response

    async def multi_request_flow(self, requests: List[str]):
        """
        Processes multiple job/career-related requests sequentially using session_state.
        """
        # Job-related session state
        session_state = {
            "skills": user_profile.get("skills", ""),
            "preferred_job_location": user_profile.get("job_location", ""),
            "experience": user_profile.get("experience", ""),
            "expected_salary": user_profile.get("expected_salary", ""),
            "preferred_roles": user_profile.get("preferred_roles", ""),
        }

        results = []
        for req in requests:
            result = await self.handle_request(req, session_state)
            results.append(result)
        return results



async def career_a2a_demo():
    # Initialize Career Assistant A2A
    career_a2a = CareerA2A(runner)

    # Job-related user requests
    requests = [
        "I have skills in Python, SQL, and ML. Which roles suit me?",
        "Find companies hiring Data Analysts in Bangalore.",
        "Create a job search roadmap with a salary range of 8-12 LPA.",
        "Suggest alternative roles matching my experience and preferred locations."
    ]

    # Run all requests sequentially
    responses = await career_a2a.multi_request_flow(requests)

    # Print all collected responses
    print("\nğŸ“Œ All Career Assistant Responses:")
    for i, resp in enumerate(responses, 1):
        print(f"{i}. {resp}")



career_a2a = CareerA2A(runner)


requests = [
    "I have skills in Python, SQL, and ML. Which roles suit me?",
    "Find companies hiring Data Analysts in Bangalore.",
    "Create a job search roadmap with a salary range of 8-12 LPA.",
    "Suggest alternative roles matching my experience and preferred locations."
]



!pip install --upgrade tornado ipykernel


responses = await career_a2a.multi_request_flow(requests)


import os
import json
import random

print("\nğŸš€ 14: Pre-Deployment Testing & Agent Engine Deployment Prep")

# 14.1: Test CareerA2A Flow
career_a2a_test_requests = [
    "I have skills in Python, SQL, and ML. Which job roles suit me?",
    "Find companies hiring Data Analysts in Bangalore.",
    "Create a job search roadmap with a salary range of 8-12 LPA."
]

print("\nğŸ“Œ Running pre-deployment CareerA2A tests...")
responses_before_deploy = run_async(CareerA2A(runner).multi_request_flow(career_a2a_test_requests))


print("\nğŸ“Œ Responses Before Deployment:")
for i, resp in enumerate(responses_before_deploy, 1):
    print(f"{i}. {resp}")

print("\nâœ… Pre-deployment CareerA2A flow tested successfully!")

# 14.2: Agent Engine Deployment Example
PROJECT_ID = os.environ.get("PROJECT_ID", "your-gcp-project-id")

# Deployment configuration
agent_engine_config = {
    "min_instances": 0,
    "max_instances": 1,
    "resource_limits": {"cpu": "1", "memory": "1Gi"}
}
config_path = "/tmp/.agent_engine_config.json"
with open(config_path, "w") as f:
    json.dump(agent_engine_config, f)
print("âœ… Agent Engine deployment config created.")

# Choose region randomly for demo
regions_list = ["europe-west1", "europe-west4", "us-east4", "us-west1"]
deployed_region = random.choice(regions_list)
print(f"âœ… Selected deployment region: {deployed_region}")

# Example deployment command (commented, requires CLI & GCP)
# !adk deploy agent_engine --project=$PROJECT_ID --region=$deployed_region sample_agent --agent_engine_config_file=$config_path

print("âœ… Deployment example ready. Use ADK CLI to deploy your Career Assistant agent to Agent Engine.")


