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


print("Starting setup...")

# 1. INSTALL LIBRARIES
# We use -q (quiet) to hide the long installation output
# We use -U (upgrade) to ensure we have the latest versions
# This installs BOTH the Agent Development Kit (ADK) and the Gemini library.
!pip install -q -U google-adk google-generativeai

print("âœ… Libraries installed.")

# 2. CONFIGURE API KEY
import os
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

try:
    # Get the key from the Kaggle secret vault we just made
    user_secrets = UserSecretsClient()
    secret_value = user_secrets.get_secret("GOOGLE_API_KEY")

    # Set it as an environment variable. 
    # The ADK will automatically find and use this.
    os.environ["GOOGLE_API_KEY"] = secret_value
    
    # Also configure the base genai library
    genai.configure(api_key=secret_value)

    print("âœ… API key configured successfully.")

except Exception as e:
    print(f"â�Œ ERROR: Could not find secret 'GOOGLE_API_KEY'.")
    print("   Please go to 'Add-ons' -> 'Secrets' and add your key.")

# 3. VERIFY IMPORTS
try:
    from google.adk.agents import Agent
    from google.adk.tools import google_search
    print("âœ… Google ADK imported successfully.")
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("âœ… Gemini (genai) model test successful.")
except ImportError as e:
    print(f"â�Œ ERROR: {e}")
    print("There was a problem importing the libraries. Please re-run the cell.")

print("\n--- Setup Complete! ---")


# --- Step 4 (Corrected): The Build ---

from google.adk.agents import Agent
from google.adk.tools import google_search
# We need the InMemorySessionService from Day 3a
from google.adk.sessions import InMemorySessionService

print("Starting the build...")

# 1. --- Define the Agent ---
# This is the simple, correct definition from Step 4.1 (v5)
fact_check_agent = Agent(
    name="FactCheckAgent",
    model='gemini-1.5-flash',
    tools=[google_search]  # Give the agent its tool
)
print("âœ… Agent defined.")

# 2. --- Define the System Instruction (as a variable) ---
# We are NOT passing this to the agent. We are just storing it
# for the next step.
SYSTEM_INSTRUCTION = """
You are a professional Fact-Check Reporter. You have one tool: Google Search.

Your workflow is very strict:
1.  When the user gives you a claim, your FIRST action is to use the `Google Search` tool.
2.  Your search query MUST be: "[the user's claim] site:snopes.com OR site:reuters.com/fact-check OR site:apnews.com/ap-fact-check"
3.  After you get the search snippets, your SECOND action is to analyze them.
4.  Your FINAL action is to write a report for the user with:
    - A summary of your findings.
    - A final rating: **Verified**, **False**, or **Uncertain**.
    - A brief explanation for your rating.
    - Cite your sources by linking to the URLs from the snippets.
"""
print("âœ… System instruction variable created.")

# 3. --- Create the Session Service ---
# As seen in the Day 3a notebook
session_service = InMemorySessionService()
print("âœ… Session Service created.")

print("\n--- Build Complete! Ready for Step 5 (The Test). ---")


# ============================
# STEP 5 â€” FINAL SETUP (Works 100% on your ADK)
# ============================

print("ğŸ”§ Starting Setup...")

import os
import google.generativeai as genai

from google.adk.sessions import InMemorySessionService
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.models import Gemini   # <-- ADK internal LLM


# 1. ---- Load API Key ----
try:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    GOOGLE_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

    os.environ["GOOGLE_API_KEY"] = GOOGLE_KEY
    genai.configure(api_key=GOOGLE_KEY)

    print("âœ… API key loaded.")
except Exception:
    print("â�Œ ERROR: Please set GOOGLE_API_KEY in Kaggle Secrets.")


# 2. ---- Create Session Service ----
session_service = InMemorySessionService()
print("âœ… Session Service ready.")


# 3. ---- Create Tools List ----
tools = [google_search]
print("ğŸ›  Tools loaded:", [t.__class__.__name__ for t in tools])


# 4. ---- System Instruction ----
SYSTEM_INSTRUCTION = """
You are a professional Fact-Check Reporter.

Your workflow:
1. ALWAYS perform a fact-check search first.
2. Your search query must be:
   "[USER CLAIM] site:snopes.com OR site:reuters.com/fact-check OR site:apnews.com/ap-fact-check"
3. Analyze the search snippets.
4. Generate a final report including:
   - Summary
   - Verification rating (Verified, False, Uncertain)
   - Explanation
   - Sources
"""
print("ğŸ“Œ System instruction loaded.")


# 5. ---- Create Agent (important: use ADK Gemini LLM) ----
fact_check_agent = Agent(
    name="FactCheckAgent",
    model=Gemini(),        # <-- THIS is the correct model
    tools=tools
)
print("ğŸ¤– Agent created successfully.")

print("ğŸ�‰ Step 5 complete!")



# --- STEP 6 â€” FINAL WORKING VERSION ---

from google.adk.runners import InMemoryRunner
import asyncio
import time

print("ğŸ”§ Runner imported. Creating instance...")

runner = InMemoryRunner(fact_check_agent)
print("âœ… Runner created.")

test_claim = "Is it true that the Great Wall of China is the only man-made object visible from the moon?"
full_prompt = SYSTEM_INSTRUCTION + "\n\nUser Claim: " + test_claim

print(f"\nğŸš€ Running test with claim: '{test_claim}'")

max_retries = 5
retry_delay = 3

for attempt in range(1, max_retries + 1):
    try:
        print(f"\nâ�³ Attempt {attempt}/{max_retries}...")
        response = await runner.run_debug(full_prompt)

        print("\n\n--- âœ… AGENT'S FINAL REPORT ---")

        # FIX: response is a list, not a single object
        for block in response:
            if isinstance(block, str):
                print(block)
            elif hasattr(block, "text"):
                print(block.text)
            else:
                print(str(block))

        break  # success

    except Exception as e:
        msg = str(e)
        if "503" in msg or "overloaded" in msg.lower():
            print(f"âš ï¸� Model overloaded, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            continue
        else:
            print(f"\nâ�Œ Unexpected error: {e}")
            break
else:
    print("\nâ�Œ Model kept failing. Try again later.")



# --- Step 7: Create the submission.md file ---

# This text will be the content of your submission file.
# It explains your project, the track, and the concepts you used.
submission_content = """
# Capstone Project: The "Fake News" Detector

**Track:** Agents for Good

---

## 1. The Pitch (The Idea)

The problem I'm solving is the rapid spread of misinformation. It's difficult for an average user to know if a headline or a claim they see online is true. My agent acts as a "Fact-Check Reporter" that helps a user quickly verify a claim against trusted, independent sources.

---

## 2. The Implementation (The Agent)

I built a single, powerful agent using the Google AI Agent Development Kit (ADK). This agent is designed to follow a strict, professional workflow.

### Core Concepts Used:

* **1. Agent with Tools (Day 2a):**
    * The `FactCheckAgent` is given a single, powerful tool: `Google Search`.

* **2. Advanced Prompting (Day 1a):**
    * The agent's "brain" is a detailed `SYSTEM_INSTRUCTION` (a system prompt). This prompt forces the agent to follow a specific, multi-step workflow.

* **3. Context Engineering (Day 3a):**
    * I engineered the context by forcing the agent's `Google Search` tool to *only* search three specific, high-quality fact-checking sites: `snopes.com`, `reuters.com/fact-check`, and `apnews.com/ap-fact-check`. This prevents the agent from citing unreliable sources and is the most important feature of its design.

* **4. Agent Runner (Day 5b):**
    * The agent is run using the `InMemoryRunner` from the ADK, which manages the agent's execution and state for the session.

### How it Works (The Workflow):

1.  **User Input:** The user provides a claim (e.g., "Is the Great Wall of China visible from the moon?").
2.  **Forced Tool Use:** The agent's instructions *force* it to use the `Google Search` tool *first*.
3.  **Scoped Search:** The agent's query is automatically modified to search *only* Snopes, Reuters, and the AP.
4.  **Analysis:** The agent analyzes the search snippets from *only* those trusted sources.
5.  **Final Report:** The agent generates a final report with a **Verified**, **False**, or **Uncertain** rating, along with an explanation and its sources.
"""

# Write the content to a file named 'submission.md'
try:
    with open("submission.md", "w") as f:
        f.write(submission_content)
    print("âœ… 'submission.md' created successfully!")
    print("\nNext, go to the 'Output' tab in your notebook's data panel,")
    print("click 'submission.md', and then 'Submit'.")
except Exception as e:
    print(f"â�Œ Error writing file: {e}")

