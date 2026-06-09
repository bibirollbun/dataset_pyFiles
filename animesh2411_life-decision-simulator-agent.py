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


# Remove conflicting packages from the Kaggle base environment.
!pip uninstall -qqy kfp jupyterlab libpysal thinc spacy fastai ydata-profiling google-cloud-bigquery google-generativeai
# Install langgraph and the packages used in this lab.
!pip install -qU 'langgraph==0.3.21' 'langchain-google-genai==2.1.2' 'langgraph-prebuilt==0.1.7'


import os
from kaggle_secrets import UserSecretsClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from typing import TypedDict, List


GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3)

response = llm.invoke("Say hello in one sentence.")
print(response.content)


class DecisionState(TypedDict):
    context: str
    options: List[str]
    priorities: str
    analysis: str
    recommendation: str


def analyze_scenarios(state: DecisionState):
    prompt = f"""
You are a decision analysis agent.

Context:
{state['context']}

Options:
{state['options']}

User Priorities:
{state['priorities']}

Analyze each option across:
- Career growth
- Financial stability
- Stress & lifestyle
- Risk & uncertainty

Simulate outcomes for 1, 3, and 5 years.
"""
    response = llm.invoke(prompt)
    return {"analysis": response.content}


def recommend_option(state: DecisionState):
    prompt = f"""
Based on the analysis below, recommend the best option.

Analysis:
{state['analysis']}

Explain:
- Why this option is best
- Who should NOT choose it
- Hidden risks
"""
    response = llm.invoke(prompt)
    return {"recommendation": response.content}


graph = StateGraph(DecisionState)

graph.add_node("analyze", analyze_scenarios)
graph.add_node("recommend", recommend_option)

graph.set_entry_point("analyze")
graph.add_edge("analyze", "recommend")

decision_agent = graph.compile()


input_data = {
    "context": "Currently working as a backend engineer in India with 4 years experience.",
    "options": [
        "Stay in current job",
        "Join an early-stage startup",
        "Move abroad for a stable role"
    ],
    "priorities": "Career growth > Financial stability > Work-life balance"
}

result = decision_agent.invoke(input_data)

print("=== ANALYSIS ===")
print(result["analysis"])

print("\n=== RECOMMENDATION ===")
print(result["recommendation"])


import pandas as pd

df = pd.DataFrame({
    "submission": [
        "Life Decision Simulator Agent capstone submission"
    ]
})

df.to_csv("/kaggle/working/submission.csv", index=False)

print("submission.csv created")

