# Run this first!
!pip install google-adk duckduckgo-search


import os
import asyncio
import logging
from kaggle_secrets import UserSecretsClient
from google.genai import types, Client

# ADK Imports
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool

# Third-party Tools
from duckduckgo_search import DDGS

# Configure Logging
logging.basicConfig(level=logging.ERROR) 
print("âœ… Imports complete.")


try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except Exception:
    print("âš ï¸� Please set 'GOOGLE_API_KEY' in Kaggle Secrets first!")

def get_best_model():
    """Dynamically finds the best model your API key can access."""
    try:
        client = Client(api_key=os.environ["GOOGLE_API_KEY"])
        available_models = [m.name for m in client.models.list(config={"page_size": 100})]
        
        priorities = [
            "models/gemini-2.5-flash-lite", 
            "models/gemini-2.0-flash-lite-preview-02-05",
            "models/gemini-2.0-flash-exp",
            "models/gemini-1.5-flash"
        ]
        
        for model in priorities:
            clean_name = model.replace("models/", "")
            if any(clean_name in m for m in available_models):
                return clean_name
    except Exception as e:
        print(f"Warning during model detection: {e}")
    
    return "gemini-1.5-flash" # Fallback

MODEL_NAME = get_best_model()
retry_policy = types.HttpRetryOptions(attempts=3, exp_base=2, initial_delay=1, http_status_codes=[429, 500, 503])

print(f"ğŸš€ SUCCESS: System configured using '{MODEL_NAME}'")


def market_search(query: str) -> str:
    """
    Searches the web for real-time market data, competitors, and pricing.
    """
    print(f"  ğŸ”� [Tool] Searching for: {query}...")
    try:
        # Using DDGS for free search
        results = DDGS().text(query, max_results=3)
        if not results: 
            return "No specific data found. Proceed with general knowledge."
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search Error: {e}"

# Wrap in ADK FunctionTool
research_tool = FunctionTool(func=market_search)
print("âœ… Research Tool initialized.")


# 1. The Scout (The Researcher)
scout_agent = LlmAgent(
    name="Scout",
    model=Gemini(model=MODEL_NAME, retry_options=retry_policy),
    instruction="""
    You are a Market Researcher. 
    Your goal is to find valid market data using the 'market_search' tool.
    If exact data is missing, find the closest similar examples or competitors.
    Output a bulleted list of facts.
    """,
    tools=[research_tool] # Binds the tool to this agent
)

# 2. The Critic (The Evaluator)
critic_agent = LlmAgent(
    name="Critic",
    model=Gemini(model=MODEL_NAME, retry_options=retry_policy),
    instruction="""
    You are a Venture Capitalist.
    Review the User Idea and the Scout's Research.
    Create a detailed SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats).
    
    CRITICAL: End your response with a score in this format:
    "Venture Score: [0-100]"
    """
)

print(f"âœ… Agents 'Scout' and 'Critic' ready on {MODEL_NAME}.")


async def run_agent_step(agent, prompt):
    """
    Runs an ADK agent in debug mode to handle tool execution loops automatically.
    Returns the final text response.
    """
    # InMemoryRunner handles the state for this specific turn
    runner = InMemoryRunner(agent=agent)
    
    # run_debug handles the Model -> Tool -> Model loop
    events = await runner.run_debug(prompt)
    
    # Extract just the final text from the complex event stream
    final_text = ""
    for event in events:
        if hasattr(event, 'content') and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    final_text += part.text
    return final_text

print("âœ… Workflow helper loaded.")


async def run_venture_validation():
    # 1. Input
    # idea = input("ğŸ’¡ Enter Business Idea: ")
    idea = "A mobile app that uses AI to diagnose plant diseases and suggest organic remedies"
    print(f"\nâš¡ STARTING VALIDATION FOR: '{idea}'\n" + "="*50)
    
    # 2. Scout Phase
    print("ğŸ•µï¸� Step 1: Scout is searching the market...")
    scout_results = await run_agent_step(scout_agent, f"Find competitors, risks, and pricing for: '{idea}'")
    print(f"\n[Scout Data Summary]\n{scout_results[:400]}...\n")
    
    # 3. Critic Phase
    print("âš–ï¸� Step 2: Critic is evaluating...")
    critic_prompt = f"""
    IDEA: {idea}
    RESEARCH DATA: {scout_results}
    
    TASK: Provide a SWOT analysis and a Venture Score.
    """
    critic_results = await run_agent_step(critic_agent, critic_prompt)
    
    # 4. Final Output
    print("\n" + "="*50 + "\nğŸ�† FINAL VENTURE REPORT\n" + "="*50)
    print(critic_results)

# Run it
if __name__ == "__main__":
    await run_venture_validation()

