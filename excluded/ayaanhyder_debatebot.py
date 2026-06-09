# Cell 1: Installation
!pip install -q -U google-adk google-generativeai


# Cell 2: API Key & Imports
import os
import json
import asyncio
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search

# 1. Get the API Key from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ['GOOGLE_API_KEY'] = api_key
    genai.configure(api_key=api_key)
    print("âœ… API Key loaded successfully.")
except Exception as e:
    print("â�Œ Error loading API Key. Did you add 'GOOGLE_API_KEY' to Add-ons > Secrets?")
    print(e)


# Cell 3: Simplified Prompts

PRO_PROMPT = """
You are a world-class debater arguing FOR the topic.
Your goal is to win the debate.

INSTRUCTIONS:
1. Read the debate history.
2. Use your internal knowledge (and `Google Search` if needed) to find facts.
3. Write a compelling, 2-sentence argument supporting your side.
4. DO NOT output "Thought" or "Action". Just output your argument directly.
"""

CON_PROMPT = """
You are a world-class debater arguing AGAINST the topic.
Your goal is to win the debate.

INSTRUCTIONS:
1. Read the debate history.
2. Use your internal knowledge (and `Google Search` if needed) to find facts.
3. Write a compelling, 2-sentence argument supporting your side.
4. DO NOT output "Thought" or "Action". Just output your argument directly.
"""


# Cell 4: Build Agents (Using Gemini 2.0)
# 1. The "Pro" Agent
pro_agent = Agent(
    name="pro_agent",
    model="gemini-2.0-flash",  
    instruction=PRO_PROMPT,
    tools=[google_search],
)
pro_agent_runner = InMemoryRunner(agent=pro_agent)

# 2. The "Con" Agent
con_agent = Agent(
    name="con_agent",
    model="gemini-2.0-flash",  
    instruction=CON_PROMPT,
    tools=[google_search],
)
con_agent_runner = InMemoryRunner(agent=con_agent)

print("âœ… Agents initialized with 'gemini-2.0-flash'.")


# Cell 5: Define the Debate Function 

def get_text(events):
    """
    Extracts text from an agent response by checking EVERY possible location.
    """
    # Strategy 1: If it's a list, iterate
    if isinstance(events, list):
        candidates = []
        for event in events:
            # Check .parts[0].text
            if hasattr(event, 'parts') and event.parts:
                for part in event.parts:
                    if hasattr(part, 'text') and part.text:
                        candidates.append(part.text)
            # Check .text
            elif hasattr(event, 'text') and event.text:
                candidates.append(event.text)
        
        # Return the longest text found (likely the Final Answer)
        if candidates:
            return max(candidates, key=len)

    # Strategy 2: If it's a single object
    if hasattr(events, 'text') and events.text:
        return events.text

    return "Error: No text found in response."

async def run_debate(topic: str, rounds: int = 2):
    print(f"=============================================================")
    print(f"ğŸ”¥ DEBATE TOPIC: {topic}")
    print(f"=============================================================\n")
    
    debate_history = [f"The topic is: {topic}"]
    
    # --- Opening Statement ---
    print("--- Opening Statement ---")
    response_events = await pro_agent_runner.run_debug(
        ["Here is the debate history. Give your opening argument.", json.dumps(debate_history)]
    )
    pro_text = get_text(response_events)
    debate_history.append(f"ProAgent: {pro_text}")
    # (run_debug prints automatically)
    
    # --- Rebuttal Rounds ---
    for i in range(rounds):
        print(f"\n--- Round {i+1} ---")
        
        # 1. ConAgent Rebuttal
        response_events = await con_agent_runner.run_debug(
            ["Here is the debate history. Form your rebuttal.", json.dumps(debate_history)]
        )
        con_text = get_text(response_events)
        debate_history.append(f"ConAgent: {con_text}")
        
        # 2. ProAgent Rebuttal
        response_events = await pro_agent_runner.run_debug(
            ["Here is the debate history. Form your rebuttal.", json.dumps(debate_history)]
        )
        pro_text = get_text(response_events)
        debate_history.append(f"ProAgent: {pro_text}")
    
    print("\n=============================================================")
    print("ğŸ”¥ DEBATE CONCLUDED")


# Cell 6: Execution

# 1. Default Topic (Ensures "Save & Run All" works without freezing)
topic = "Will AI make software developers obsolete?"

# 2. Interactive Mode (Uncomment lines below to type your own topic!)
# ------------------------------------------------------------------
# print("ğŸ�™ï¸� ENTER A DEBATE TOPIC (or press Enter for default):")
# user_input = input()
# if user_input.strip():
#     topic = user_input
# ------------------------------------------------------------------

# 3. Start the Debate
await run_debate(topic, rounds=2)


# Cell 7: Fact Check Agent (Using Gemini 2.0)
fact_check_agent = Agent(
    name="fact_checker",
    model="gemini-2.0-flash",  # <--- Valid model from your list
    instruction="You are a fact checker. Use google_search to answer the user's question with ONLY the fact.",
    tools=[google_search],
)
fact_check_runner = InMemoryRunner(agent=fact_check_agent)
print("âœ… Fact Checker ready.")


# Cell 8: Test Data
TEST_SET_DATA = [
    {"prompt": "What is the capital of France?", "golden_answer": "Paris"},
    {"prompt": "Who is the CEO of Google?", "golden_answer": "Sundar Pichai"},
    {"prompt": "What is the boiling point of water in Celsius?", "golden_answer": "100"}
]
print("âœ… Test Data loaded.")


# Cell 9: Manual Evaluation Loop 

def get_text_brute_force(events):
    """
    Converts the entire event object to a string.
    This guarantees we find the answer if it exists anywhere in the data.
    """
    return str(events)

async def run_manual_eval():
    print(f"=============================================================")
    print(f"ğŸ§ª RUNNING AGENT EVALUATION (Custom Pipeline)")
    print(f"=============================================================\n")
    
    score = 0
    total = len(TEST_SET_DATA)
    
    for test_case in TEST_SET_DATA:
        prompt = test_case["prompt"]
        golden = test_case["golden_answer"]
        
        # Run the agent
        response_events = await fact_check_runner.run_debug([prompt])
        
        # FIX: Convert EVERYTHING to a string
        full_dump = get_text_brute_force(response_events)
        
        # Check result
        if golden.lower() in full_dump.lower():
            print(f"âœ… PASS | Expected: '{golden}'")
            score += 1
        else:
            print(f"â�Œ FAIL | Expected: '{golden}'")
            # Optional: Print part of the dump to debug if needed
            # print(f"Debug: {full_dump[:100]}...")
            
    print(f"\n-------------------------------------------------------------")
    print(f"ğŸ�† FINAL SCORE: {score}/{total} ({(score/total)*100:.0f}%)")
    print("=============================================================")


# Cell 10: Execute Evals
await run_manual_eval()

