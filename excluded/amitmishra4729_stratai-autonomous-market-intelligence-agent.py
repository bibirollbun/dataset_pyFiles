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

# Safer installation command for Save & Run
!pip install -q -U google-generativeai ddgs colorama

# --- IMPORTS ---
import time
import google.generativeai as genai
from ddgs import DDGS  # <--- NEW IMPORT
from colorama import Fore, Style
from IPython.display import Markdown, display
from kaggle_secrets import UserSecretsClient

# --- SETUP API KEY ---
user_secrets = UserSecretsClient()
MY_GOOGLE_KEY = user_secrets.get_secret("GOOGLE_API_KEY") 
genai.configure(api_key=MY_GOOGLE_KEY)

# --- OBSERVABILITY ---
class AgentLogger:
    @staticmethod
    def log(agent, action, content):
        timestamp = time.strftime("%H:%M:%S")
        color = Fore.GREEN if agent == "Scout" else Fore.CYAN if agent == "Analyst" else Fore.MAGENTA
        print(f"{Style.DIM}[{timestamp}]{Style.RESET_ALL} {color}{Style.BRIGHT}[{agent}]{Style.RESET_ALL} {action}")
        # Only print first 500 chars to keep logs clean
        clean_content = str(content).replace("\n", " ")
        print(f"{Fore.WHITE}{clean_content[:300]}..." if len(clean_content) > 300 else clean_content)
        print("-" * 40)

# --- UPDATED TOOL (More Robust) ---
def web_search_tool(query):
    AgentLogger.log("Tool", "Searching Web", query)
    try:
        # Using the new DDGS library structure
        results = DDGS().text(query, max_results=5)
        
        # Check if results exist
        if not results:
            return "No search results found. The company might be too niche or the search API is blocked."
            
        # Debug print to confirm data flow
        print(f"{Fore.YELLOW}DEBUG: Found {len(results)} search results.{Style.RESET_ALL}")
        
        # Format the results
        return "\n".join([f"Source: {r['title']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        return f"Search Error: {e}"

# --- AGENT CLASS ---
class AI_Agent:
    def __init__(self, name, role, model):
        self.name = name
        self.role = role
        self.model = model

    def work(self, task, context=""):
        # STRICT PROMPT to stop "chatting"
        prompt = f"""
        You are {self.name}, a {self.role}.
        
        INPUT CONTEXT:
        {context}
        
        YOUR TASK:
        {task}
        
        INSTRUCTIONS:
        1. Analyze the INPUT CONTEXT immediately.
        2. Do NOT say "I am ready" or "Please provide data".
        3. Perform the task on the available context.
        4. If the context is empty, make a reasonable guess based on your internal knowledge but mention it is a guess.
        """
        AgentLogger.log(self.name, "Thinking", "Processing task...")
        try:
            response = self.model.generate_content(prompt)
            AgentLogger.log(self.name, "Finished", response.text)
            return response.text
        except Exception as e:
            return f"Agent Error: {e}"

# --- MAIN WORKFLOW ---
def run_competitor_intel(target_company):
    # Use the specific model version to avoid 404 errors
    model = genai.GenerativeModel('gemini-2.5-flash') 
    
    display(Markdown(f"# ğŸš€ Launching Enterprise Intel Agent for: **{target_company}**"))
    print("Beginning multi-agent sequential workflow...\n")

    # STEP 1: SCOUT
    scout = AI_Agent("Scout", "Data Gatherer", model)
    raw_news = web_search_tool(f"Latest strategic news and financial results for {target_company} 2024 2025")
    
    # STEP 2: ANALYST
    analyst = AI_Agent("Analyst", "Strategic Filter", model)
    key_points = analyst.work(
        task="Extract top 3 critical strategic moves. Ignore marketing fluff.",
        context=raw_news
    )

    # STEP 3: REPORTER
    reporter = AI_Agent("Reporter", "Executive Briefing Writer", model)
    final_report = reporter.work(
        task="Write a professional Executive Briefing in Markdown. Include 'Strategic Verdict'.",
        context=key_points
    )

    print("\nâœ… Workflow Complete.")
    return final_report

# --- EXECUTION ---
target_company = "NVIDIA" 
report_output = run_competitor_intel(target_company)
display(Markdown(report_output))


import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# 1. Setup API Key
user_secrets = UserSecretsClient()
my_key = user_secrets.get_secret("GOOGLE_API_KEY")
genai.configure(api_key=my_key)

# 2. List Valid Models
print("Checking available models for your key...")
try:
    found_any = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"âœ… AVAILABLE: {m.name}")
            found_any = True
    
    if not found_any:
        print("â�Œ No text generation models found. Check if 'Generative Language API' is enabled in Google Cloud Console.")

except Exception as e:
    print(f"â�Œ Error connecting: {e}")

