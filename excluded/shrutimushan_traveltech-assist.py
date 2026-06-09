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


# =========================================
# TravelTech Assist - Kaggle Capstone (Single Cell)
# =========================================

import asyncio
import nest_asyncio
import logging
import json
from datetime import date
from typing import Dict, List

nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)

# ===== Memory Bank =====
memory_bank: List[Dict[str, str]] = []

def save_customer_issue(issue_type: str, text: str) -> str:
    entry = {"type": issue_type, "text": text, "date": str(date.today())}
    memory_bank.append(entry)
    return f"Saved to memory: {text[:60]}..."

# ===== Agents =====
def understanding_agent(question: str) -> str:
    travel_keywords = ["flight", "booking", "ticket", "baggage", "refund"]
    tech_keywords = ["app", "device", "software", "update", "crash", "password"]
    if any(word in question.lower() for word in travel_keywords):
        return "travel"
    elif any(word in question.lower() for word in tech_keywords):
        return "tech"
    else:
        return "general"

async def research_agent(issue_type: str, question: str) -> str:
    if issue_type == "travel":
        response_text = f"Travel info: For '{question}', check flight status, contact airline, or request refund."
    elif issue_type == "tech":
        response_text = f"Tech info: For '{question}', restart device, update software, or reset password."
    else:
        response_text = f"General info: For '{question}', please provide more details."
    
    save_customer_issue(issue_type, response_text)
    return response_text

async def parallel_research(issue_type: str, question: str) -> str:
    return await research_agent(issue_type, question)

def response_agent(issue_type: str, research_text: str) -> str:
    return f"Dear Customer, based on your question, here is the suggested solution:\n{research_text}"

async def handle_customer_question(question: str) -> Dict[str, str]:
    issue_type = understanding_agent(question)
    print(f"\nğŸ’¬ Customer Question: {question}")
    print(f"ğŸ”� Identified Issue Type: {issue_type}")
    research_text = await parallel_research(issue_type, question)
    final_reply = response_agent(issue_type, research_text)
    print(f"âœ… Agent Response: {final_reply}\n")
    return {"question": question, "issue_type": issue_type, "reply": final_reply, "date": str(date.today())}

# ===== Demo & Simulated Questions =====
customer_questions = [
    "My flight is delayed. How can I get a refund?",
    "My app keeps crashing when I open it.",
    "How do I reset my password?",
    "Can I change my booking date?"
]

simulated_questions = [
    "My flight is delayed. Can I get a refund?",
    "How do I reset my device password?",
    "My app crashed after the update.",
    "Can I change my booking to next month?"
]

# ===== Main Async Runner =====
async def main():
    final_results = []

    all_questions = customer_questions + simulated_questions
    
    for q in all_questions:
        result = await handle_customer_question(q)
        final_results.append(result)
        
        # ===== Dashboard Update After Each Question =====
        print("ğŸ“Š Updated Dashboard:")
        print(f"{'No.':<4} {'Question':<50} {'Issue Type':<10} {'Date':<12}")
        print("-"*80)
        for i, entry in enumerate(final_results, start=1):
            q_display = (entry['question'][:47] + '...') if len(entry['question']) > 50 else entry['question']
            print(f"{i:<4} {q_display:<50} {entry['issue_type']:<10} {entry['date']:<12} âœ…")
        print("\n")

    # ===== Save JSON (nicely formatted) =====
    output_file = "traveltech_assist_results.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=4, sort_keys=True)
    
    # ===== Success Message =====
    print("\n=======================================")
    print("âœ… TravelTech Assist Notebook Executed Successfully!")
    print(f"ğŸ“„ All results saved in a neatly formatted JSON file: {output_file}")
    print("ğŸ�‰ Ready for Kaggle submission!")
    print("=======================================\n")

# Run the notebook safely
asyncio.run(main())


