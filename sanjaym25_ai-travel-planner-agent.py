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


pip install langchain langchain-openai openai



import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool

# ---------------------------------------------------
# 1. Set your API key
# ---------------------------------------------------
os.environ["OPENAI_API_KEY"] = "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"

# ---------------------------------------------------
# 2. Create a simple tool (optional but useful)
# ---------------------------------------------------
def budget_tool(budget: str) -> str:
    return f"Okay! I will plan a trip within a budget of {budget}."

budget_calc = Tool(
    name="BudgetCalculator",
    func=budget_tool,
    description="Use this tool when the user mentions their budget."
)

# ---------------------------------------------------
# 3. Create the LLM and Agent
# ---------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

agent = initialize_agent(
    tools=[budget_calc],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# ---------------------------------------------------
# 4. Run the Agent (Travel Planner)
# ---------------------------------------------------
user_query = """
Plan a 3-day trip to Goa for a budget of 15000 INR.
Include: places to visit, food suggestions, travel tips.
"""

response = agent.run(user_query)
print(response)


