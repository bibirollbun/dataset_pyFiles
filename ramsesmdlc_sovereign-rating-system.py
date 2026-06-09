User Prompt
     â”‚
     â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚ SequentialAgent          â”‚
â”‚ "ResearchSystem"         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
            â”‚
            â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚ ParallelAgent           â”‚  â†� runs 3 researcher agents simultaneously
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
   â–¼        â–¼        â–¼
Political   Economic   Public Debt
Risk Agent  Strength    Burden Agent
Agent       Agent
   â”‚        â”‚        â”‚
   â””â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º (Session State)
                                   {political_risk}
                                   {economic_strength}
                                   {public_debt}
                                   â†“
                  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
                  â”‚ Aggregator Agent                     â”‚
                  â”‚ Combines all research using session  â”‚
                  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                   â”‚
                                   â–¼
                              Final Summary



import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# political_risk_agent
tech_researcher = Agent(
    name="Political_risk_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are the Institutional Strength & Political Risk Research Agent.
Search and summarize public political, institutional, and governance information
about the Republic of Panama. Focus on corruption, rule of law, transparency,
stability, public institutions, and democracy.
Provide a structured summary. Keep the report very concise (100 words).""",
    tools=[google_search],
    output_key="political_risk",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… political_risk created.")


# economic_strength_agent
tech_researcher = Agent(
    name="Economic_strength_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are the Economic Strength & Growth Performance Research Agent.
Analyze Panama's GDP growth, employment, external sector, inflation,
sectoral performance, investment flows, and economic stability.
Provide a structured summary. Keep the report very concise (100 words).""",
    tools=[google_search],
    output_key="economic_strength",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… economic_strength created.")


# public_debt_agent.
tech_researcher = Agent(
    name="Public_debt_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are the Public Debt Burden & Sustainability Research Agent.
Research Panamaâ€™s fiscal deficit, government debt ratio, interest burden,
maturity structure, and IMF/WB assessments.
Provide a structured summary. Keep the report very concise (100 words).""",
    tools=[google_search],
    output_key="public_debt",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… public_debt created.")


# The AggregatorAgent runs *after* the parallel step to synthesize the results.
aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""Combine these three research findings into a single executive summary:

    **Institutional Strength & Political Risk:**
    {political_risk}
    
    **Economic Strength & Growth Performance:**
    {economic_strength}
    
    **Public Debt Burden & Sustainability:**
    {public_debt}
    
    Your summary should highlight common themes, surprising connections, and the most important key takeaways from all three reports. The final summary should be around 200 words.""",
    output_key="executive_summary",  # This will be the final output of the entire system.
)

print("âœ… aggregator_agent created.")


# The ParallelAgent runs all its sub-agents simultaneously.
parallel_research_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[political_risk, economic_strength, public_debt],
)

# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
root_agent = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[parallel_research_team, aggregator_agent],
)

print("âœ… Parallel and Sequential Agents created.")


runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug(
    "Run the daily executive briefing on Tech, Health, and Finance"
)

