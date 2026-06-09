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


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
openai_key = user_secrets.get_secret("OPENAI_API_KEY")


!pip install --quiet openai aiohttp tenacity


import os, json, time, pickle, logging, uuid, textwrap
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from tenacity import retry, wait_exponential, stop_after_attempt
import requests
import openai

# ---------- Load secrets ----------
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    OPENAI_API_KEY = user_secrets.get_secret("OPENAI_API_KEY")
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    
except:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
   
if not OPENAI_API_KEY:
    raise RuntimeError("â�Œ Set OPENAI_API_KEY in Kaggle Secrets before running.")

openai.api_key = OPENAI_API_KEY



from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)





from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------
# Simple In-Memory Memory System
# -----------------------------------------------------------
class SimpleMemory:
    def __init__(self):
        self.messages = []

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get(self):
        return self.messages[-10:]  # last 10 messages


memory = SimpleMemory()

# -----------------------------------------------------------
# OpenAI Chat Wrapper (new API)
# -----------------------------------------------------------

def call_openai(role, content):
    memory.add(role, content)

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=memory.get(),
            max_tokens=300
        )
        reply = completion.choices[0].message.content
        memory.add("assistant", reply)
        return reply

    except Exception as e:
        print("â�Œ API Error:", e)
        return "Error calling model."


# -----------------------------------------------------------
# Multi-Agent Classes
# -----------------------------------------------------------

class StudyPlannerAgent:
    def generate_plan(self, topic):
        prompt = (
            f"You are a study planning agent. Create a structured, simple "
            f"3-step study plan for the topic: {topic}."
        )
        return call_openai("user", prompt)


class StudyExplainerAgent:
    def explain_topic(self, topic):
        prompt = (
            f"You are an explanation agent. Explain '{topic}' in simple "
            f"language with examples."
        )
        return call_openai("user", prompt)


class QuizGeneratorAgent:
    def generate_quiz(self, topic):
        prompt = (
            f"You are a quiz agent. Generate 5 easy MCQs for: {topic}."
        )
        return call_openai("user", prompt)


planner = StudyPlannerAgent()
explainer = StudyExplainerAgent()
quizzer = QuizGeneratorAgent()

print("âœ… Multi-Agent System Loaded with NEW OpenAI API!")



topic = "Machine Learning basics"

print("ğŸ“˜ Study Plan:")
print(planner.generate_plan(topic))

print("\nğŸ“˜ Explanation:")
print(explainer.explain_topic(topic))

print("\nğŸ“� Quiz:")
print(quizzer.generate_quiz(topic))


