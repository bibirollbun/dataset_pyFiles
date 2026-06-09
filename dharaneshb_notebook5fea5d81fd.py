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


import json
import time
from typing import Callable, Dict, Any, List

# ---------------------------------------
# 1. DEFINE SIMPLE TOOLS (FAKE TOOLS)
# ---------------------------------------
def search_tool(query: str) -> str:
    return f"[search results for '{query}']"

def calculator(expr: str) -> str:
    try:
        return str(eval(expr))
    except:
        return "Error evaluating expression"

TOOLS = {
    "search": search_tool,
    "calculator": calculator
}

# ---------------------------------------
# 2. LLM INTERFACE (USE ANY MODEL)
# ---------------------------------------
class LLM:
    """
    Replace the contents of generate() with:
    - Google Gemini API
    - OpenAI GPT
    - Local LLM
    Current version: Mock LLM for teaching.
    """

    def generate(self, prompt: str) -> str:
        # Fake minimal "model" that detects intentions
        if "calculate" in prompt.lower():
            return json.dumps({
                "action": "calculator",
                "input": prompt.split("calculate")[-1].strip()
            })
        elif "search" in prompt.lower():
            return json.dumps({
                "action": "search",
                "input": prompt.split("search")[-1].strip()
            })
        else:
            return json.dumps({
                "action": "respond",
                "input": "I cannot perform that action yet."
            })


# ---------------------------------------
# 3. AGENT ENGINE
# ---------------------------------------
class Agent:
    def __init__(self, llm: LLM, tools: Dict[str, Callable[[str], str]]):
        self.llm = llm
        self.tools = tools
        self.memory: List[str] = []

    def think(self, user_input: str):
        self.memory.append(f"USER: {user_input}")

        # Ask LLM what to do
        model_output = self.llm.generate(user_input)

        try:
            parsed = json.loads(model_output)
        except:
            return "Model output invalid", None

        action = parsed.get("action")
        action_input = parsed.get("input")

        # Use a tool
        if action in self.tools:
            result = self.tools[action](action_input)
            self.memory.append(f"TOOL RESULT: {result}")
            return result

        # Or respond normally
        if action == "respond":
            return action_input

        return "Unknown action"

    def run(self):
        print("AI Agent Ready. Type 'exit' to stop.")
        while True:
            msg = input("You: ")
            if msg.lower() == "exit":
                break
            answer = self.think(msg)
            print("Agent:", answer)


# ---------------------------------------
# 4. RUN AGENT
# ---------------------------------------
if __name__ == "__main__":
    agent = Agent(LLM(), TOOLS)
    agent.run()


