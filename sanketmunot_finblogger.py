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


from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.adk.tools import AgentTool, google_search

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


pitcher_configs = [
    {
        "id": "macro_pitcher",
        "name": "Macro Pitcher",
        "expertise": "Indian macro economy, RBI guidelines, inflation, GDP, fiscal policy",
        "prompt_style": "Analytical, data-driven, concise",
        "idea_count": 5
    },
    {
        "id": "markets_pitcher",
        "name": "Markets Pitcher",
        "expertise": "Equity markets, stocks, earnings, sector trends, FII/DII flows",
        "prompt_style": "Fast-paced, market journalist tone",
        "idea_count": 5
    },
    {
        "id": "scam_pitcher",
        "name": "Scam & Fraud Pitcher",
        "expertise": "Cyber frauds, Ponzi schemes, UPI scams, digital security",
        "prompt_style": "Alert, risk-focused, educational",
        "idea_count": 5
    },
    {
        "id": "global_pitcher",
        "name": "Global Pitcher",
        "expertise": "Geo-political events, global markets, oil, forex, US Fed, China",
        "prompt_style": "Global macro analyst, sharp insights",
        "idea_count": 5
    }
]

voter_config = [
    {
        "id": 1,
        "name": "voter_1"
    },
    {
        "id": 2,
        "name": "voter_2"
    },
    {
        "id": 3,
        "name": "voter_3"
    },
]


def create_pitcher_agent(cfg):
    instruction = f"""
You are {cfg['name']}.
Your expertise: {cfg['expertise']}.
Writing style: {cfg['prompt_style']}.
Generate {cfg['idea_count']} high-quality blog ideas.
use google_search tool according to your expertise to do the research on latest data

ALWAYS GENERATE {cfg['idea_count']} ideas, DO NOT ASK ANY QUESTION TO THE USER

Mention below points for every idea - 
 Title, Angle, Sentiment, SEO Tags
"""
# {", ".join(cfg['data_sources'])}
    return Agent(
        name=cfg["id"],
        instruction=instruction,
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        tools=[ google_search],
        output_key=f"{cfg['id']}_ideas"
    )



def create_voting_agent(cfg):
    instruction = f"""
    You are {cfg['name']}, a specialized editor in Indian finance and media.
    
    INPUT: You will receive a list of blog pitches.
    
    YOUR TASK:
    1. Analyze each pitch using Google Search to verify if the topic is trending/relevant.
    2. Score each idea on a scale of 0-10 based on: Impact, Timeliness, and Uniqueness.
    3. **CRITICAL:** For every vote, provide a "Writer Note".
       - Example: "Score: 8/10. Writer Note: Great topic, but ensure you mention the specific SEBI circular dated yesterday."
    
    OUTPUT FORMAT:
    Return a structured review for every pitch with: [ID, Score, Writer Note].
    """
    
    return Agent(
        name=cfg["name"],
        instruction=instruction,
        model=Gemini(
            model="gemini-2.5-flash", # Updated to stable model
            retry_options=retry_config
        ),
        tools=[google_search],
        output_key=f"{cfg['id']}_vote"
    )


def create_writer_agent():
    instruction = """
    You are an expert financial blog editor and writer (similar to a Bloomberg or Mint columnist).
    
    INPUT: You will receive a "Winning Topic" which includes:
    1. The Idea/Title
    2. The original Angle/Pitch
    3. Feedback/Critiques from the Voting Agents (what was good, what was missing)
    
    YOUR TASK:
    Write a high-impact, production-ready finance blog post based on this input.
    
    GUIDELINES:
    - **Tone:** Authoritative, optimistic, yet risk-aware.
    - **Structure:**
      1. **Headline:** Catchy, click-worthy but not clickbait.
      2. **Sub-headline:** A 1-sentence summary of the value.
      3. **The Hook (Intro):** Why this matters NOW (current market context).
      4. **The Deep Dive (Body):** Use the research provided in the prompt. 
      5. **The Takeaway (Outro):** Actionable advice for the reader.
    
    - **Critical Step:** specifically address any "missing" points highlighted by the Voting Agents in their feedback.
    """
    
    return Agent(
        name="writer_agent",
        instruction=instruction,
        model=Gemini(
            model="gemini-2.5-flash", # Updated to stable model
            retry_options=retry_config
        ),
        tools=[google_search],
        # changed output_key to be static so Root Agent can find it easily
        output_key="final_blog_draft" 
    )


pitcher_agents = [create_pitcher_agent(cfg) for cfg in pitcher_configs]
pitcher_tools = [AgentTool(agent) for agent in pitcher_agents]


voter_agents = [create_voting_agent(cfg) for cfg in voter_config]
voter_tools = [AgentTool(agent) for agent in voter_agents]


writer_agent = create_writer_agent()
writer_tool = [AgentTool(writer_agent)]


# Create a string list of pitcher names to help the Root Agent
pitcher_names = [p['id'] for p in pitcher_configs]
voter_names = [v['name'] for v in voter_config]

root_agent = Agent(
    name="ResearchCoordinator",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    instruction=f"""You are a financial blog publishing coordinator. 
    Your goal is to get high quality, high impact, excellent blogs by orchestrating a workflow.
    
    Step 1: Call these specific Pitcher Agents to get ideas: {', '.join(pitcher_names)}.
    Step 2: Collect all ideas and pass them to these Voting Agents: {', '.join(voter_names)}. 
    Step 3: Pick the top 3 blogs based on the cumulative score from {', '.join(voter_names)} voters.
    Step 4: Pass the top 3 blogs to 'writer_agent' to draft 3 blogs.
    
    Make sure to show the reasoning for your final selection.
    """,
    tools=pitcher_tools + voter_tools + writer_tool,
)


currency_runner = InMemoryRunner(agent=root_agent)
finalBlog = await currency_runner.run_debug(
    """
    Run the finance blog workflow and show the final blog
 
    """
)

