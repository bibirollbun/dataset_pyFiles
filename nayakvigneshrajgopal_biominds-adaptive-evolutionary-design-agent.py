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
import google.generativeai as genai
from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from kaggle_secrets import UserSecretsClient
from typing import Dict, Any
from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool


retry_config = types.HttpRetryOptions(
    attempts=2,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    print("✅ Authenticated with Google AI")
except Exception as e:
    print("❌ Error: GOOGLE_API_KEY not found. Please ensure it is in Kaggle Secrets.")


biologist = Agent(
    name="Biologist",
    description="An expert in taxonomy, systems biology, and bio-mimicry. Uses Google Search to find nature's solution to an engineering problem.",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""
    You are an expert Evolutionary Biologist. Your domain knowledge covers Taxonomy, Functional Morphology, and Systems Biology.
    
    1. Analyze the user's engineering problem (e.g., 'cooling', 'filtration', 'drag reduction').
    2. Use the google_search tool to find the most relevant organism or natural system that solves this problem.
    3. Focus your summary on the underlying **Structure-Function** relationship and **Physical Principle**.
    4. Output a concise summary of the Biological Mechanism found.
    """,
    tools=[google_search],
    output_key="bio_research" 
)


initial_engineer = Agent(
    name="Genesis_Engineer",
    description="Translates raw biological principles into an initial industrial design draft (Draft V0).",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""
    Context: {bio_research}
    As a Mechanical Engineer, create an initial design concept (Draft V0). 
    Your design must be practical, but do not worry about optimization yet.
    Detail the proposed materials, structure, and mechanism of action.
    Output this as "Design Draft V0".
    """,
    output_key="current_design" 
)


def stop_loop(reason: str) -> str:
    """
    Stops the current Evolutionary Cycle because the design quality is satisfactory.
    The 'reason' is the final verdict from the auditor.
    """
    print(f"\n*** ADAPTIVE LOOP EXIT TRIGGERED: {reason} ***")
    return f"Loop exit initiated: {reason}"


critic = Agent(
    name="Sustainability_Critic",
    description="A certified auditor who critiques the current design draft for environmental flaws and decides when the design is Optimal.",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Review the Current Design: {current_design}
    
    You are a harsh Sustainability Auditor. Find ONE major flaw to force evolution.
    
    CRITICAL DECISION: If the design is clearly highly sustainable (score 8/10 or higher) AND has gone through at least one iteration, call the **stop_loop** tool with the reason 'Design Quality Optimal'.
    
    If the design needs improvement, output the detailed "Critique" and the verdict "Needs Improvement".
    """,
    tools=[stop_loop], # The Agent now has the power to exit the loop
    output_key="critique_with_verdict"
)


refiner = Agent(
    name="Evolutionary_Refiner",
    description="A CAD specialist who incorporates the Critic's feedback to generate a refined, new version of the design.",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""
    Current Design: {current_design}
    Critique: {critique_with_verdict}
    Biological Inspiration: {bio_research}
    
    You must address the Critique completely. EVOLVE the design to improve sustainability and efficiency.
    Rewrite the full updated design as "Design Draft V(new)". Do not mention previous versions.
    """,
    output_key="current_design" # Overwrites the design state
)


evolutionary_loop = LoopAgent(
    name="Evolution_Cycle",
    description="An iterative loop that critiques and refines the design up to 5 times, stopping early if the Critic calls the stop_loop tool.",
    sub_agents=[critic, refiner],
    max_iterations=5, 
)

# The Pipeline: Biologist -> Genesis -> Loop
biominds_system = SequentialAgent(
    name="BioMinds_Core",
    description="The core engine for BioMinds, managing the design workflow from biological discovery to final evolutionary refinement.",
    sub_agents=[biologist, initial_engineer, evolutionary_loop]
)


runner = InMemoryRunner(agent=biominds_system)


user_query = input('enter a concept you want to study: ')
response = await runner.run_debug(user_query)


user_query = "Design a cow dung manure distribution system to farm plants that works without human labor, using inspiration from nature."
response = await runner.run_debug(user_query)


user_query = "Design a completely self-cleaning, low-friction surface for large solar panels that works without human labor, using inspiration from nature."
response = await runner.run_debug(user_query)

