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


pip install google-generativeai


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



import google.generativeai as genai
import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
genai.configure(api_key=GOOGLE_API_KEY)
MODEL = "gemini-2.5-flash" 

@dataclass
class TaskEstimate:
    module: str
    task: str
    complexity: str  # S, M, L
    hours: float

@dataclass
class ProjectEstimate:
    summary: str
    assumptions: List[str]
    clarifications: List[str]
    tasks: List[TaskEstimate]
    total_hours: float
    base_cost: float
    low_cost: float
    high_cost: float
    duration_weeks: float

# Configurable parameters
HOURLY_RATE = 45.0  # USD/hour
TEAM_SIZE = 4
HOURS_PER_WEEK = 35
UTILIZATION = 0.85
RISK_LOW = 1.1
RISK_HIGH = 1.4

def call_gemini(system_prompt: str, user_prompt: str) -> str:
    model = genai.GenerativeModel(MODEL)
    response = model.generate_content([system_prompt, user_prompt])
    return response.text

def analyze_requirements(description: str) -> Dict[str, Any]:
    system_prompt = (
        "You are an expert project estimator. Analyze the project description. "
        "Return ONLY valid JSON: {'summary': str, 'assumptions': [], 'clarifications': []}"
    )
    user_prompt = f"Project: {description}"
    raw = call_gemini(system_prompt, user_prompt)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except:
        return {"summary": "Generic project", "assumptions": [], "clarifications": []}

def generate_tasks(summary: str) -> List[TaskEstimate]:
    system_prompt = (
        "Break project into tasks by module. Each task: complexity S(4-8h)/M(12-24h)/L(32-80h), hours estimate. "
        "Return ONLY JSON array: [{'module': str, 'task': str, 'complexity': str, 'hours': float}]"
    )
    user_prompt = f"Summary: {summary}"
    raw = call_gemini(system_prompt, user_prompt)
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        tasks_data = json.loads(raw[start:end])
        return [TaskEstimate(**t) for t in tasks_data]
    except:
        return []

def calculate_costs(tasks: List[TaskEstimate]) -> Dict[str, float]:
    total_hours = sum(t.hours for t in tasks)
    base_cost = total_hours * HOURLY_RATE
    effective_hours_week = TEAM_SIZE * HOURS_PER_WEEK * UTILIZATION
    duration = total_hours / effective_hours_week if effective_hours_week > 0 else 0
    return {
        "total_hours": total_hours,
        "base_cost": base_cost,
        "low_cost": base_cost * RISK_LOW,
        "high_cost": base_cost * RISK_HIGH,
        "duration_weeks": duration
    }

def estimate_project(description: str) -> ProjectEstimate:
    analysis = analyze_requirements(description)
    tasks = generate_tasks(analysis["summary"])
    costs = calculate_costs(tasks)
    
    return ProjectEstimate(
        summary=analysis["summary"],
        assumptions=analysis["assumptions"],
        clarifications=analysis["clarifications"],
        tasks=tasks,
        total_hours=costs["total_hours"],
        base_cost=costs["base_cost"],
        low_cost=costs["low_cost"],
        high_cost=costs["high_cost"],
        duration_weeks=costs["duration_weeks"]
    )

def print_estimate(estimate: ProjectEstimate):
    print("=== PROJECT ESTIMATION  ===\n")
    print(f"Summary: {estimate.summary}\n")
    print(f"Total Hours: {estimate.total_hours:.1f}")
    print(f"Cost Range: ${estimate.low_cost:,.0f} - ${estimate.high_cost:,.0f} (Base: ${estimate.base_cost:,.0f})")
    print(f"Duration: {estimate.duration_weeks:.1f} weeks\n")
    
    if estimate.assumptions:
        print("Assumptions:")
        for a in estimate.assumptions:
            print(f"- {a}")
        print()
    
    if estimate.clarifications:
        print("Clarifications Needed:")
        for c in estimate.clarifications:
            print(f"- {c}")
        print()
    
    print("Task Breakdown:")
    for task in estimate.tasks:
        print(f"[{task.module}] ({task.complexity}) {task.task}: {task.hours:.1f}h")




# Example usage
project_desc = """
Develop a web app for e-commerce with user auth, product catalog, cart/checkout, 
payment gateway (Stripe), admin dashboard, and email notifications. MVP in 8 weeks.
"""
result = estimate_project(project_desc)
print_estimate(result)

