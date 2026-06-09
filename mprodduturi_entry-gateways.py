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


# -------------------------------
# Part 1: Setup & Installation
# -------------------------------
!pip install --quiet google-adk --upgrade --no-deps
!pip install --quiet google-generativeai --upgrade --no-deps


import json
import requests
import time
import uuid

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import warnings
warnings.filterwarnings("ignore")

print("âœ… ADK public components imported successfully.")


from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import shutil
from pathlib import Path
DATA_DIR = Path("/kaggle/working/fallbackdata")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_or_fallback(url, local_filename, fallback_data):
    """
    Download dataset from URL. If failed, use fallback_data.
    """
    try:
        print(f"[bold cyan]Downloading {url} ...[/bold cyan]")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            f.write(r.content)
        print(f"[green]Downloaded and saved to {local_filename}[/green]")
    except Exception as e:
        print(f"[yellow]Download failed: {e}. Using fallback.[/yellow]")
        with open(local_filename, "w", encoding="utf-8") as f:
            json.dump(fallback_data, f, indent=2)
        print(f"[green]Fallback saved to {local_filename}[/green]")


# -------------------------------
# Example fallback data
# -------------------------------
OECD_COUNTRIES = [
    "Australia","Austria","Belgium","Canada","Chile","Czech Republic","Denmark",
    "Estonia","Finland","France","Germany","Greece","Hungary","Iceland",
    "Ireland","Israel","Italy","Japan","Korea","Latvia","Lithuania",
    "Luxembourg","Mexico","Netherlands","New Zealand","Norway","Poland",
    "Portugal","Slovak Republic","Slovenia","Spain","Sweden","Switzerland",
    "Turkey","United Kingdom","United States"
]

# Minimal fallback JSON per pillar
fallback_immigration = {c: {"policies": "Fallback immigration info"} for c in OECD_COUNTRIES}
fallback_jobs = {c: {"jobs": "Fallback jobs & economy info"} for c in OECD_COUNTRIES}
fallback_qol = {c: {"quality_of_life": "Fallback QoL info"} for c in OECD_COUNTRIES}
fallback_health = {c: {"healthcare_education": "Fallback health/edu info"} for c in OECD_COUNTRIES}
fallback_safety = {c: {"safety_governance": "Fallback safety info"} for c in OECD_COUNTRIES}
fallback_culture = {c: {"cultural_integration": "Fallback culture info"} for c in OECD_COUNTRIES}


import shutil
from pathlib import Path

# Source folder (read-only)
input_dir = Path("/kaggle/input/fallbackdata")  # <-- your uploaded dataset

# Destination folder (writable)
working_dir = Path("/kaggle/working/fallbackdata")
working_dir.mkdir(parents=True, exist_ok=True)

# Copy all JSON files
for file_path in input_dir.glob("*.json"):
    dest_file = working_dir / file_path.name
    shutil.copy(file_path, dest_file)

print(f"âœ… Copied {len(list(input_dir.glob('*.json')))} files to {working_dir}")


DATA_DIR = Path("/kaggle/working/fallbackdata")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# URLs for real datasets (example placeholders, replace with real endpoints if needed)
OECD_IMMIGRATION_URL = "https://stats.oecd.org/restsdmx/sdmx.ashx/GetData/MIG/ALL/OECD?format=csv"
IOM_MIGRATION_URL = "https://migrationdataportal.org/api/test_endpoint"
WORLD_BANK_QOL_URL = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?format=json"

# Download or fallback
#download_or_fallback(OECD_IMMIGRATION_URL, DATA_DIR/"immigration_policies.json", fallback_immigration)
#download_or_fallback(IOM_MIGRATION_URL, DATA_DIR/"jobs_and_economy.json", fallback_jobs)
#download_or_fallback(WORLD_BANK_QOL_URL, DATA_DIR/"quality_of_life.json", fallback_qol)

# Save remaining fallback JSONs
with open(DATA_DIR/"healthcare_education.json","w") as f:
    json.dump(fallback_health, f, indent=2)
with open(DATA_DIR/"safety_governance.json","w") as f:
    json.dump(fallback_safety, f, indent=2)
with open(DATA_DIR/"culture_integration.json","w") as f:
    json.dump(fallback_culture, f, indent=2)
with open(DATA_DIR/"immigration_policies.json","w") as f:
    json.dump(fallback_immigration, f, indent=2)
with open(DATA_DIR/"jobs_and_economy.json","w") as f:
    json.dump(fallback_jobs, f, indent=2)
with open(DATA_DIR/"quality_of_life.json","w") as f:
    json.dump(fallback_qol, f, indent=2)


def load_json_tool(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Load all pillar tools
immigration_tool = load_json_tool(DATA_DIR/"immigration_policies.json")
jobs_tool = load_json_tool(DATA_DIR/"jobs_and_economy.json")
qol_tool = load_json_tool(DATA_DIR/"quality_of_life.json")
health_tool = load_json_tool(DATA_DIR/"healthcare_education.json")
safety_tool = load_json_tool(DATA_DIR/"safety_governance.json")
culture_tool = load_json_tool(DATA_DIR/"culture_integration.json")


# Example tool query
def query_tool(tool, country):
    return tool.get(country, {"info": "No data available"})


from typing import Dict

class LLM_Agent:
    def __init__(self, name, model="gemini-2.0-flash", tools=None):
        self.name = name
        self.model = model
        self.tools = tools or {}
        self.memory = []

    def query_tool(self, tool_name, country):
        tool = self.tools.get(tool_name)
        if tool:
            return query_tool(tool, country)
        return {"info": f"No tool named {tool_name}"}

    def run(self, country, user_query):
        # Simple reasoning: query all tools and append memory
        result = {"agent": self.name, "country": country, "response": {}}
        for tool_name in self.tools.keys():
            data = self.query_tool(tool_name, country)
            result["response"][tool_name] = data
        self.memory.append({"query": user_query, "result": result})
        return result


# Each agent is specialized in one pillar
immigration_agent = LLM_Agent("ImmigrationAgent", tools={"immigration_tool": immigration_tool})
jobs_agent = LLM_Agent("JobsAgent", tools={"jobs_tool": jobs_tool})
qol_agent = LLM_Agent("QoLAgent", tools={"qol_tool": qol_tool})
health_agent = LLM_Agent("HealthAgent", tools={"health_tool": health_tool})
safety_agent = LLM_Agent("SafetyAgent", tools={"safety_tool": safety_tool})
culture_agent = LLM_Agent("CultureAgent", tools={"culture_tool": culture_tool})


class OrchestratorAgent:
    def __init__(self, agents, search_tool=None):
        self.agents = agents
        self.search_tool = search_tool
        self.memory = []

    def run_all(self, country, user_query):
        results = {}
        for agent in self.agents:
            agent_result = agent.run(country, user_query)
            results[agent.name] = agent_result
        self.memory.append({"query": user_query, "results": results})
        return results


# Example: Google Search Tool (placeholder)
def google_search_tool(query):
    return f"[Search results for '{query}']"

# Orchestrator with agents
orchestrator = OrchestratorAgent(
    agents=[immigration_agent, jobs_agent, qol_agent, health_agent, safety_agent, culture_agent],
    search_tool=google_search_tool
)


country = "Germany"
user_query = "I want to migrate for work and life. Evaluate key pillars."

results = orchestrator.run_all(country, user_query)

# Display results nicely
for agent_name, agent_output in results.items():
    print(f"\n[bold cyan]{agent_name} Response:[/bold cyan]")
    print(agent_output["response"])


def evaluate_results(results):
    evaluation = {}
    for agent_name, agent_output in results.items():
        responses = agent_output["response"]
        non_empty = {k:v for k,v in responses.items() if v}
        evaluation[agent_name] = {
            "pillars_returned": list(non_empty.keys()),
            "status": "OK" if non_empty else "MISSING"
        }
    return evaluation

evaluation = evaluate_results(results)
print("\n[bold magenta]Evaluation Summary:[/bold magenta]")
print(evaluation)


print("\n[bold green]Orchestrator Memory:[/bold green]")
for mem in orchestrator.memory:
    print(mem)


import json
from pathlib import Path

DATA_DIR = Path("/kaggle/working/fallbackdata")
json_files = DATA_DIR.glob("*.json")

# Load all JSONs into a single dictionary
migration_data = {}
for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        migration_data[file.stem] = json.load(f)

print("âœ… Loaded data for pillars:", list(migration_data.keys()))

#Example access of the data
print(migration_data.keys())  # List all loaded JSON file stems
print(migration_data["immigration_policies"]["Canada"].keys())  # show country names



from google.adk.sessions import InMemorySessionService

# Setup session management (required by ADK)
session_service = InMemorySessionService()

# Session identifiers
app_name = "EntryGatewaysApp"
user_id = "demo_user"
# Use unique session ID for each test to avoid conflicts
session_id = f"demo_session_{uuid.uuid4().hex[:8]}"

# CRITICAL: Create session BEFORE running agent (synchronous, not async!)
# This pattern matches the deployment notebook exactly
session = await session_service.create_session(
    app_name=app_name, user_id=user_id, session_id=session_id
)

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner

# Create LLM agent
agent = LlmAgent(name="MigrationAssistant", model="gemini-2.0-flash")

# Runner must have the SAME app_name
runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

from google.genai import types

prompt_text = "What visa options are available for skilled workers in Canada?"

content = types.Content(
    role="user",
    parts=[types.Part(text=prompt_text)]
)

events = runner.run(
    user_id=user_id,
    session_id=session_id,
    new_message=content
)

# Extract final text
final_text = None
for ev in events:
    if ev.content and ev.content.parts:
        final_text = ev.content.parts[0].text

print("Agent Response:\n", final_text)


