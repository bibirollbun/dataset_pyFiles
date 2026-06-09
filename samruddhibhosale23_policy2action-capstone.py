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
import re
from typing import List, Dict

# Placeholder for Gemini or LLM calls
def call_llm(prompt):
    return "LLM response placeholder"


def load_policy_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

#text = load_policy_text('/mnt/data/sample_policy.txt')


def policy_parsing_agent(text: str) -> Dict:
    prompt = f"Extract key sections from this policy:\n{text}"
    response = call_llm(prompt)
    return {"parsed": response}

# parsed = policy_parsing_agent(text)


def action_extraction_agent(parsed_obj: Dict) -> Dict:
    prompt = f"Convert the following into steps, eligibility, documents, deadlines: {parsed_obj}"
    response = call_llm(prompt)
    return {"actions": response}


def form_generator_agent(actions: Dict) -> str:
    prompt = f"Generate a fillable form for: {actions}"
    response = call_llm(prompt)
    return response


def reminder_scheduler(actions: Dict) -> Dict:
    return {"scheduled": "Reminder scheduled (placeholder)"}


def policy2action_pipeline(text: str):
    parsed = policy_parsing_agent(text)
    actions = action_extraction_agent(parsed)
    forms = form_generator_agent(actions)
    reminders = reminder_scheduler(actions)
    return {"parsed": parsed, "actions": actions, "forms": forms, "reminders": reminders}

# Example run:
# result = policy2action_pipeline(text)

