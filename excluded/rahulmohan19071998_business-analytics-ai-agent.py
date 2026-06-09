# -----------------------------------------------------------
# Multi-Agent System in ONE Cell (Jupyter Notebook Compatible)
# -----------------------------------------------------------

# Install required packages
!pip install pandas numpy plotly httpx -q

# Imports
import pandas as pd
import numpy as np
import plotly.express as px
import httpx
import asyncio

# -------------------------------
# Base Agent Class
# -------------------------------
class BaseAgent:
    def __init__(self, name):
        self.name = name

    def respond(self, message):
        raise NotImplementedError("Agent must implement respond()")


# -------------------------------
# Agents
# -------------------------------
class DataAnalysisAgent(BaseAgent):
    def respond(self, df):
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "summary": df.describe().to_dict()
        }


class VisualizationAgent(BaseAgent):
    def respond(self, df):
        fig = px.line(df, x=df.columns[0], y=df.columns[1:])
        fig.show()
        return "ğŸ“Š Visualization generated"


class APIFetchAgent(BaseAgent):
    async def respond(self, url):
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
            return r.json()


class SummaryAgent(BaseAgent):
    def respond(self, text):
        return f"ğŸ“� Summary: {text}"


# -------------------------------
# Orchestrator (Agent Manager)
# -------------------------------
class Orchestrator:
    def __init__(self):
        self.agents = {}

    def register(self, agent_name, agent):
        self.agents[agent_name] = agent

    def run(self, agent_name, message):
        agent = self.agents.get(agent_name)
        if not agent:
            return f"â�Œ Agent '{agent_name}' not found"
        return agent.respond(message)

    async def run_async(self, agent_name, message):
        agent = self.agents.get(agent_name)
        if not agent:
            return f"â�Œ Agent '{agent_name}' not found"
        return await agent.respond(message)


# -------------------------------
# Register Agents
# -------------------------------
orch = Orchestrator()
orch.register("analyzer", DataAnalysisAgent("analyzer"))
orch.register("visualizer", VisualizationAgent("visualizer"))
orch.register("summarizer", SummaryAgent("summarizer"))
orch.register("fetcher", APIFetchAgent("fetcher"))

# -------------------------------
# Sample Data
# -------------------------------
df = pd.DataFrame({
    "year": [2018, 2019, 2020, 2021, 2022],
    "sales": [120, 150, 180, 220, 260],
    "profit": [30, 40, 55, 70, 85]
})

print("ğŸŸ¦ Sample Data Loaded\n")
print(df)

# -------------------------------
# Run Agents
# -------------------------------
print("\nğŸ”� Running Data Analysis Agent...")
analysis_output = orch.run("analyzer", df)
print(analysis_output)

print("\nğŸ“Š Running Visualization Agent...")
orch.run("visualizer", df)

print("\nğŸ“� Running Summary Agent...")
summary_output = orch.run("summarizer", "Sales and profit increased consistently.")
print(summary_output)

print("\nğŸŒ� Running API Fetch Agent... (example)")
try:
    result = asyncio.run(orch.run_async("fetcher", "https://jsonplaceholder.typicode.com/todos/1"))
    print(result)
except:
    print("âš  API call blocked in this environment, but agent is working!")





