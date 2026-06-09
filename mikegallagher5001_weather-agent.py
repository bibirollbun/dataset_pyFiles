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


# ============================================================
# 1. Install & Imports
# ============================================================


!pip install google-genai -q


import time
import json
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient


# ============================================================
# 2. Initialize Gemini Client from Kaggle Secrets
#    (Make sure you set a secret called "GEMINI_API_KEY")
# ============================================================


user_secrets = UserSecretsClient()
GEMINI_API_KEY = None


try:
    GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
except Exception as e:
    print(f"Error loading secret: {e}")


if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("âœ… Gemini Client initialized successfully using Kaggle Secrets.")
else:
    client = None
    print("â�Œ Client initialization failed due to missing API Key. Please add GEMINI_API_KEY in Kaggle Secrets.")


# ============================================================
# 3. System Instruction: Northeast Corridor Weather Agent
# ============================================================


root_system_instruction = (
    "You are the Northeast Corridor Weather-Performance Insight Agent. "
    "You focus on passenger rail service between New York City and Philadelphia, "
    "including intermediate locations such as Newark, Trenton, and other nearby stops. "
    "Your job is to provide highly accurate and up-to-date information on weather and "
    "how it may impact rail operations and passenger experience. "
    "Always use the 'google_search' tool to look up real-time conditions and forecasts. "
    "When asked about a corridor or location, you MUST provide both the CURRENT weather "
    "and a short FUTURE outlook (e.g., next 24â€“72 hours). "
    "Summarize any potential operational risks (e.g., heavy rain, strong winds, heat, snow) "
    "in simple language suitable for executive summaries or passenger communications. "
    "Highlight all temperatures, key conditions, and risk levels using emojis and bold text "
    "where helpful (e.g., ğŸŒ¡ï¸�, â˜”, â�„ï¸�, âš ï¸�). "
    "You may be called multiple times in the same session; make use of prior context when useful."
)


print("âœ… System instruction defined.")


# ============================================================
# 4. Simple Session Memory (for Capstone 'Sessions & Memory')
# ============================================================


SESSION = {
    "history": []  # each item: {"user": str, "agent": str}
}


def build_memory_snippet(max_turns: int = 3) -> str:
    """
    Returns a short textual snippet summarizing the last few interactions.
    This is enough to demonstrate simple session memory for the Capstone.
    """
    if not SESSION["history"]:
        return "No prior conversation context."


    recent = SESSION["history"][-max_turns:]
    lines = []
    for turn in recent:
        lines.append(f"User asked: {turn['user']}")
        lines.append(f"Agent replied (summary): {turn['agent'][:200]}...")
    return "\n".join(lines)


print("âœ… Basic in-memory session state initialized.")


# ============================================================
# 5. Simple Observability (Logging + Timing)
# ============================================================


def log_event(message: str):
    """Tiny helper for observability/logging."""
    # In a more advanced setting this could be structured logging.
    print(f"[LOG] {message}")


# ============================================================
# 6. Core Agent Function
#    - Uses tools (google_search)
#    - Uses memory (SESSION)
#    - Logs events (observability)
# ============================================================


def run_corridor_weather_agent(user_prompt: str) -> str:
    """Runs the Northeast Corridor Weather-Performance Insight Agent."""
    global SESSION


    if client is None:
        return "Error: Gemini Client is not initialized due to missing API Key."


    # Build a short memory snippet from previous turns
    memory_snippet = build_memory_snippet(max_turns=3)


    # Combine memory and current user request into a single content string
    combined_prompt = (
        "Here is recent conversation context (may be helpful):\n"
        f"{memory_snippet}\n\n"
        "Now, answer the new request below, focusing on passenger rail weather "
        "and operational risk between New York City and Philadelphia:\n\n"
        f"User request: {user_prompt}"
    )


    # Observability: log what the user asked
    log_event(f"New request: {user_prompt}")


    # Observability: track timing
    t0 = time.time()


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=combined_prompt,
        config=types.GenerateContentConfig(
            system_instruction=root_system_instruction,
            tools=[{"google_search": {}}]  # âœ… Capstone 'Tools' requirement
        )
    )


    t1 = time.time()
    elapsed = t1 - t0


    # Extract text
    answer_text = response.text or "(No response text returned.)"


    # Observability: log response stats
    log_event(f"Response length: {len(answer_text)} characters")
    log_event(f"Elapsed time: {elapsed:.2f} seconds")


    # Update memory
    SESSION["history"].append({
        "user": user_prompt,
        "agent": answer_text
    })


    return answer_text


print("âœ… Core agent function 'run_corridor_weather_agent' defined.")


# ============================================================
# 7. Example Calls (Demonstration for Reviewers)
# ============================================================


if client is not None:
    demo_queries = [
        "Give me a current and 48-hour weather and risk summary for passenger rail between New York City and Philadelphia.",
        "Focus on potential weather-related operational risks for rail service tomorrow afternoon along this corridor.",
        "Summarize this as a short executive-style paragraph I could paste into a monthly performance spotlight."
    ]


    for i, q in enumerate(demo_queries, start=1):
        print("\n" + "="*60)
        print(f"ğŸ”� Demo Query {i}: {q}")
        print("="*60)
        answer = run_corridor_weather_agent(q)
        print("\nğŸ¤– Agent Response:\n")
        print(answer)
else:
    print("âš ï¸� Skipping demo calls because the Gemini client is not initialized.")




