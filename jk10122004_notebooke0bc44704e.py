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


!pip install -q fastapi uvicorn[standard] scikit-learn pandas joblib prometheus_client kaggle google-cloud-storage requests


import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any


# ================================================================
# Base Agent
# ================================================================
class Agent(ABC):
    @abstractmethod
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass


# ================================================================
# LLM Client (Mock)
# ================================================================
class LLMClient:
    async def generate(self, prompt: str) -> str:
        return f"[LLM Response]: {prompt}"


# ================================================================
# LLM Agent
# ================================================================
class LLMAgent(Agent):
    def __init__(self, name: str, client: LLMClient):
        self.name = name
        self.client = client

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = context.get("input", "")
        result = await self.client.generate(prompt)
        return {self.name: result}


# ================================================================
# Sequential Agent
# ================================================================
class SequentialAgent(Agent):
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        data = context
        for agent in self.agents:
            result = await agent.run(data)
            data.update(result)
        return data


# ================================================================
# Parallel Agent
# ================================================================
class ParallelAgent(Agent):
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tasks = [agent.run(context) for agent in self.agents]
        results = await asyncio.gather(*tasks)

        combined = {}
        for r in results:
            combined.update(r)
        return combined


# ================================================================
# Loop Agent
# ================================================================
class LoopAgent(Agent):
    def __init__(self, agent: Agent, times: int):
        self.agent = agent
        self.times = times

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        data = context
        for _ in range(self.times):
            result = await self.agent.run(data)
            data.update(result)
        return data


# ================================================================
# Tools
# ================================================================
class Tool(ABC):
    @abstractmethod
    async def execute(self, query: str) -> str:
        pass


class GoogleSearchTool(Tool):
    async def execute(self, query: str) -> str:
        return f"[Google Search Result]: {query}"


class CodeExecutionTool(Tool):
    async def execute(self, code: str) -> str:
        try:
            env = {"__builtins__": __builtins__}  # <-- FIXED
            exec(code, env)
            return str(env)
        except Exception as e:
            return f"Error: {e}"


# ================================================================
# Tool Agent
# ================================================================
class ToolAgent(Agent):
    def __init__(self, name: str, tool: Tool):
        self.name = name
        self.tool = tool

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        result = await self.tool.execute(query)
        return {self.name: result}


# ================================================================
# Enterprise Workflow
# ================================================================
async def enterprise_workflow():
    llm_client = LLMClient()

    a1 = LLMAgent("ResearchAgent", llm_client)
    a2 = ToolAgent("SearchAgent", GoogleSearchTool())
    a3 = ToolAgent("CodeAgent", CodeExecutionTool())

    workflow = SequentialAgent([
        a1,
        ParallelAgent([a2, a3]),
        LoopAgent(a1, 2)
    ])

    context = {
        "input": "Write research summary about enterprise AI.",
        "query": "Enterprise AI trends 2025"
    }

    result = await workflow.run(context)
    return result


# ================================================================
# MAIN EXECUTION FOR JUPYTER / KAGGLE / COLAB  (NO asyncio ERROR)
# ================================================================
import nest_asyncio
nest_asyncio.apply()

output = await enterprise_workflow()
print("\nFinal Output:\n")
print(output)


