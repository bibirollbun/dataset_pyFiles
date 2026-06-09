# Install necessary libraries
!pip install -q crewai
!pip install -q crewai_tools
!pip install -q duckduckgo-search
!pip install -q langchain_community
print("âœ… Libraries installed successfully.")


import os
from kaggle_secrets import UserSecretsClient
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from IPython.display import Markdown, display

# --- 1. SETUP ENVIRONMENT (The Fix) ---
# We redirect "OpenAI" calls to Groq to stop Auth Errors.
try:
    user_secrets = UserSecretsClient()
    GROQ_KEY = user_secrets.get_secret("GROQ_API_KEY")
except:
    GROQ_KEY = "MISSING_KEY"
    print("âš ï¸� Warning: Check Kaggle Secrets.")

os.environ["OPENAI_API_KEY"] = GROQ_KEY
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama-3.1-8b-instant"

print("âœ… Environment configured.")

# --- 2. DEFINE TOOL ---
# Custom tool wrapper to satisfy CrewAI requirements
class SimpleSearchTool(BaseTool):
    name: str = "Search"
    description: str = "Search the web."
    def _run(self, query: str) -> str:
        return DuckDuckGoSearchRun().run(query)

search_tool = SimpleSearchTool()

# --- 3. DEFINE AGENTS ---
news_agent = Agent(
    role='News Finder',
    goal='Find exactly 8 interesting AI news headlines from the last 24 hours.',
    backstory="You are a focused researcher who finds facts quickly.",
    tools=[search_tool],
    verbose=True
)

writer_agent = Agent(
    role='List Writer',
    goal='Format the news into a clean list.',
    backstory="You write simple, clean bullet points. No fluff.",
    verbose=True
)

# --- 4. DEFINE TASKS ---
task_find = Task(
    description="Search for 8 distinct AI news headlines from today.",
    expected_output="A raw list of 8 headlines.",
    agent=news_agent
)

task_list = Task(
    description="Return a list of exactly 8 headlines. Use bullet points. No intro text.",
    expected_output="A markdown formatted list.",
    agent=writer_agent,
    context=[task_find]
)

# --- 5. RUN CREW ---
my_crew = Crew(
    agents=[news_agent, writer_agent],
    tasks=[task_find, task_list],
    verbose=True
)

print("ğŸš€ Getting Headlines... (Please wait)")
result = my_crew.kickoff()

# --- 6. DISPLAY RESULT (The Display Fix) ---
print("\n\n" + "="*40)
print("       âœ¨ LATEST AI HEADLINES âœ¨")
print("="*40 + "\n")

# We use .raw to get the plain text string, fixing the TypeError
display(Markdown(result.raw))

