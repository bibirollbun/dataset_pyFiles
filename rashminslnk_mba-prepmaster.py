# =========================================================
# 1. Authentication + Environment Setup
# =========================================================
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' "
        f"to your Kaggle secrets. Details: {e}"
    )


# =========================================================
# 2. Import Required ADK Components
# =========================================================
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


# =========================================================
# 3. Retry Configuration
# =========================================================
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# =========================================================
# 4. MBA Exam Research Agent
# =========================================================
# This searches CAT/XAT/SNAP/NMAT/CMAT/IIFT etc. information.
research_agent = Agent(
    name="MBAResearchAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    instruction="""
You are an MBA Entrance Exam Research Agent.

Use ONLY the google_search tool.

Your job:
- Fetch accurate information about MBA exams (CAT, XAT, NMAT, SNAP, CMAT, IIFT, TISSNET, MICAT)
- Gather 2â€“4 relevant facts
- Include citations in the response
""",
    tools=[google_search],
    output_key="research_findings",
)

print("âœ… MBAResearchAgent created.")


# =========================================================
# 5. Summarizer Agent
# =========================================================
summarizer_agent = Agent(
    name="MBASummarizerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    instruction="""
Read the following research findings:

{research_findings}

Your job:
- Produce a clean bulleted summary
- Include 4â€“6 key points
- Keep it simple and helpful for students preparing for MBA entrance exams
""",
    output_key="final_summary",
)

print("âœ… MBASummarizerAgent created.")


# =========================================================
# 6. MBA Exam Concierge Coordinator
# =========================================================
root_agent = Agent(
    name="MBAConciergeCoordinator",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    instruction="""
You are an MBA Exam Concierge Coordinator for exams like:
- CAT
- XAT
- SNAP
- NMAT
- CMAT
- MICAT
- IIFT
- TISSNET

Your responsibilities:
1. ALWAYS call the MBAResearchAgent tool first.
2. After getting research findings, ALWAYS call the MBASummarizerAgent tool.
3. Finally, present the summary as your final answer to the user.

Never skip steps.
""",
    tools=[
        AgentTool(research_agent),
        AgentTool(summarizer_agent),
    ],
)

print("âœ… MBAConciergeCoordinator created.")


# =========================================================
# 7. Runner
# =========================================================
runner = InMemoryRunner(agent=root_agent)
print("ðŸš€ Runner initialized successfully! Ready to query.")


# =========================================================
# 8. Example Query (MBA EXAM CONCIERGE)
# =========================================================
response = await runner.run_debug(
    "Tell me the difference between CAT and XAT exam."
)




