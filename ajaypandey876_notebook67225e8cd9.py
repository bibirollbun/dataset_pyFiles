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


!pip install anthropic python-dotenv -q

import anthropic
import json
import time
from datetime import datetime



API_KEY = "AIzaSyBVZ3XivYdOCQG9JMX-y2ddkSHAn4IFgNM"  
client = anthropic.Anthropic(api_key=API_KEY)



memory = {
    "tasks": [],
    "history": []
}

def remember(role, msg):
    memory["history"].append({"role": role, "msg": msg})



def simple_web_search(query):
    return f"[Fake Search] Results for '{query}'"

def simple_calculator(expr):
    try:
        return f"Result = {eval(expr)}"
    except:
        return "Error in expression"

def simple_task_manager(action, task=None):
    if action == "add":
        memory["tasks"].append(task)
        return f"Added task: {task}"
    elif action == "list":
        return "\n".join(memory["tasks"]) if memory["tasks"] else "No tasks"



def research_agent(query):
    return simple_web_search(query)



def task_agent(message):
    if "add" in message:
        task = message.replace("add", "").strip()
        return simple_task_manager("add", task)
    return simple_task_manager("list")



def calculator_agent(message):
    expr = message.lower().replace("calculate", "").strip()
    return simple_calculator(expr)



def coordinator(message):
    remember("user", message)
    msg = message.lower()

    if any(w in msg for w in ["search", "find", "what is"]):
        reply = research_agent(message)
    elif "task" in msg or "todo" in msg or "add" in msg:
        reply = task_agent(message)
    elif "calculate" in msg:
        reply = calculator_agent(message)
    else:
        reply = research_agent(message)

    remember("assistant", reply)
    return reply



print("TEST 1: Research")
print(coordinator("What is AI?"))
print("\nTEST 2: Task")
print(coordinator("Add buy groceries"))
print("\nTEST 3: List Tasks")
print(coordinator("show tasks"))
print("\nTEST 4: Calculator")
print(coordinator("calculate 12*8"))


