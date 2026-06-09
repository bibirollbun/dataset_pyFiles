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


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner,Runner
from google.adk.tools import google_search, AgentTool
from google.genai import types
from google.adk.sessions import InMemorySessionService

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


# research a companies core values
company_core_value_agent = LlmAgent(
    name="company_core_value_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that researches a companies core values",
    instruction="You are a helpful assistant. Use google search to research any given company's core values and come up with a summary in bullet point format of what is more important for any given company",
    output_key="company_core_values",
    tools=[google_search],
)
# summarize resume
resume_skills_summarizer_agent = LlmAgent(
    name="resume_skills_summarizer_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    output_key="resume_summary",
    description="A simple agent that summarizes a resume",
    instruction="You are a text summarizer. your task is to create two lists based on the given resume. First you come up with a list of all the tech skills mentioned in the resume, and then a second list for all the soft skills mentioned in the resume. output two separate lists, consiting of comma separated values. example: TechSkills : [\"java\",\"python\"], SoftSkills: [\"leadership\",\"time management\"]",
)

#summarize job description
job_desc_skills_summarizer_agent = LlmAgent(
    name="job_desc_skills_summarizer_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    output_key="job_desc_summary",
    description="A simple agent that summarizes a job description",
    instruction="You are a text summarizer. your task is to create two lists based on the given job description. First you come up with a list of all the tech skills mentioned in the job description, and then a second list for all the soft skills mentioned in the job description. output two separate lists, consiting of comma separated values. example: TechSkills : [\"java\",\"python\"], SoftSkills: [\"leadership\",\"time management\"]",
)

skills_aggeregator_agent = LlmAgent(
    name="skills_aggeregator_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    output_key="skills_comparison",
    description="A simple agent that compares skills mentioned in the job description and a resume",
    instruction= """
    Job description skills required: {job_desc_summary},
    Resume skills: {resume_summary}.

    Compare the skills mentioned above, and come up with a list of what is missing in resume in each category. be very concise and show bullet points only.
    display in bellow format:
    TechSkills requested: ["Java","Python"]
    TechSkills possessed: ["Java"]
    You need to learn ["Java"] for this position.

    SoftSkills requested: ["Management","leadership"]
    SoftSkills possessed: ["leadership"]
    You need to demonstrate ["Management"] skills for this position.
    """
)

resume_analyser_agent = SequentialAgent(
    name="resume_analyser_agent",
    sub_agents=[resume_skills_summarizer_agent, job_desc_skills_summarizer_agent, skills_aggeregator_agent],
)

resume_update_suggestion_agent =LlmAgent(
    name="company_core_value_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that suggests improvement for resume",
    instruction="""You have to first use the resume_analyser_agent to analyse a resume and a job description. Then use the company_core_value_agent to research the company.
    show a summary of all the technical skills needed, and what are missing. Then show a summary of the matching softskills I have and the softskills i need to learn. Then based on the softskills I have and the company's core value, suggest to add a paragraph that shows how my softskills aligns with one or more of the company's core values""",
    tools=[AgentTool(resume_analyser_agent), AgentTool(company_core_value_agent)],
)


# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("âœ… Helper functions defined.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


APP_NAME = "resume_analyzer"
USER_ID = "user1"
SESSION = "session1"

MODEL_NAME = "gemini-2.5-flash-lite"

session_service = InMemorySessionService()
 
runner = Runner(agent=resume_update_suggestion_agent, app_name=APP_NAME, session_service=session_service)
await run_session(
    runner,
    [
            """here is a job description from Cisco:We need people who know java and kubernetes, and also can handle stress, and can manage large teams with strong leadership skills. and here is my resume: I know java, python, and I have managed large teams""", 
    ],
    "stateful-agentic-session",
)

 




