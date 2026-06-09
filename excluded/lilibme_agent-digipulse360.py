import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import LlmAgent
from google.adk.tools import google_search

# --- Analyst Agents ---

tech_analyst = LlmAgent(
    name="TechnologyAnalyst",
    model="gemini-2.5-flash-lite",
    description="Specialist in tracking technological innovations in digital health.",
    instruction=(
        "You are a Technology Analyst for DigiPulse360. "
        "Your goal is to identify and analyze the latest technological advancements, research papers, and patents in the digital health sector, specifically related to the user's query. "
        "Focus on: AI/ML models, remote monitoring devices, telemedicine platforms, and novel algorithms. "
        "IMPORTANT: If the user's query mentions a specific time period (e.g., 'November 2025', 'recent', 'last month'), you MUST focus exclusively on developments within that timeframe. "
        "Use the google_search tool to find recent and relevant information."
        "When using the google_search tool, include temporal constraints in your search queries (e.g., 'AI wound care November 2025', 'telemedicine 2025', etc.)."
        "Provide a detailed summary of your findings with specific dates, citing sources where possible."
    ),
    tools=[google_search],
    output_key="tech_findings"
)

market_analyst = LlmAgent(
    name="MarketAnalyst",
    model="gemini-2.5-flash-lite",
    description="Specialist in tracking market trends and competitor activity in digital health.",
    instruction=(
        "You are a Market Analyst for DigiPulse360. "
        "Your goal is to identify and analyze product relesae, market trend, M&A activity and funding rounds in the digital health sector, specifically related to the user's query. "
        "Focus on: Key players, emerging startups, new products or use cases, and strategic partnerships. "
        "IMPORTANT: If the user's query mentions a specific time period (e.g., 'November 2025', 'recent', 'last month'), you MUST focus exclusively on market activities within that timeframe. "
        "Use the google_search tool to find recent and relevant information. "
        "When using the google_search tool, include temporal constraints in your search queries (e.g., 'digital health funding November 2025', 'healthcare M&A 2025', etc.). "
        "Provide a detailed summary of your findings with specific dates, citing sources where possible."
    ),
    tools=[google_search],
    output_key="market_findings"
)

regulatory_analyst = LlmAgent(
    name="RegulatoryAnalyst",
    model="gemini-2.5-flash-lite",
    description="Specialist in tracking regulations and standards in digital health.",
    instruction=(
        "You are a Regulatory Analyst for DigiPulse360. "
        "Your goal is to monitor new regulations or guidance from medical device regulatory bodies (e.g. Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions from FDA), "
        "new and updated industrial standards (e.g. AAMI TIR34971:2023; Application of ISO 14971 to machine learning in artificial intelligenceâ€”Guide),"
        "and regulatory approvals (e.g., SaMD / AI-enabled medical device clearances) in the digital health sector, specifically related to the user's query. "
        "Focus on: Policy changes, new standards, new device approval and reimbursement codes. "
        "IMPORTANT: If the user's query mentions a specific time period (e.g., 'November 2025', 'recent', 'last month'), you MUST focus exclusively on regulatory changes within that timeframe. "
        "Use the google_search tool to find recent and relevant information. "
        "When using the google_search tool, include temporal constraints in your search queries (e.g., 'FDA approval November 2025', 'HIPAA updates 2025', etc.). "
        "Provide a detailed summary of your findings with specific dates, citing sources where possible."
    ),
    tools=[google_search],
    output_key="regulatory_findings"
)

# --- Newsletter Agents ---

newsletter_writer = LlmAgent(
    name="NewsletterWriter",
    model="gemini-2.5-flash-lite",
    description="Synthesizes research into a newsletter draft.",
    instruction=(
        "You are the Newsletter Writer for DigiPulse360. "
        "Your task is to compile a comprehensive newsletter based on the findings from the analysts. "
        "You will have access to 'tech_findings', 'market_findings', and 'regulatory_findings' in the session state. "
        "Synthesize this information into a cohesive, professional, and actionable newsletter for the R&D team. "
        "Structure the newsletter with clear headings: 'Technological Innovations', 'Market Dynamics', and 'Regulatory Updates'. "
        "IMPORTANT: Ensure all information in the newsletter is from the specified time period mentioned in the user's query. "
        "Include specific dates and timeframes for all developments mentioned. "
        "If you receive feedback from the Reviewer (in 'reviewer_feedback'), incorporate it into your revised draft. "
        "Ensure the tone is professional, concise, and forward-looking."
        "Cite the information source."
    ),
    output_key="newsletter_draft"
)

newsletter_reviewer = LlmAgent(
    name="NewsletterReviewer",
    model="gemini-2.5-flash-lite",
    description="Reviews the newsletter draft for quality and accuracy.",
    instruction=(
        "You are the Newsletter Reviewer for DigiPulse360. "
        "Your task is to critique the 'newsletter_draft' produced by the Writer. "
        "Check for: Clarity, accuracy, coherence, actionable insights, and adherence to the format. "
        "IMPORTANT: Verify that all information is relevant to the time period specified in the user's query. "
        "Ensure dates and timeframes are clearly mentioned for all developments. "
        "If the draft is satisfactory and meets high quality standards, output 'APPROVED' as your response. "
        "If improvements are needed, provide specific, actionable feedback in your response to guide the Writer. "
        "Do NOT rewrite the newsletter yourself; only provide feedback."
    ),
    output_key="reviewer_feedback"
)

print("âœ“ Agents defined successfully")


from google.adk.agents import ParallelAgent, SequentialAgent, LoopAgent, BaseAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from typing import AsyncGenerator

# 1. Parallel Research Phase
analysts_parallel = ParallelAgent(
    name="AnalystsTeam",
    sub_agents=[tech_analyst, market_analyst, regulatory_analyst],
    description="Runs the technology, market, and regulatory analysts in parallel."
)

# 2. Refinement Loop Phase

class ApprovalAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Access the session state to check reviewer feedback
        session = await ctx.session_service.get_session(
            app_name=ctx.app_name, 
            user_id=ctx.user_id, 
            session_id=ctx.session.id
        )
        
        feedback = session.state.get("reviewer_feedback", "")
        
        if "APPROVED" in feedback:
             yield Event(
                 author=self.name,
                 actions=EventActions(escalate=True),
                 content=types.Content(parts=[types.Part(text="Newsletter Approved. Escalating.")])
             )
        else:
             yield Event(
                 author=self.name,
                 content=types.Content(parts=[types.Part(text="Newsletter NOT Approved. Continuing loop.")])
             )

approval_check = ApprovalAgent(
    name="ApprovalCheck",
    description="Checks if the newsletter is approved and escalates if so."
)

draft_review_sequence = SequentialAgent(
    name="DraftAndReview",
    sub_agents=[newsletter_writer, newsletter_reviewer, approval_check],
    description="Writer drafts, Reviewer critiques, ApprovalCheck decides."
)

refinement_loop = LoopAgent(
    name="RefinementLoop",
    sub_agents=[draft_review_sequence],
    max_iterations=3,
    description="Iteratively refines the newsletter until approved."
)

# 3. Root Workflow
root_workflow = SequentialAgent(
    name="DigiPulse360_Workflow",
    sub_agents=[analysts_parallel, refinement_loop],
    description="Main workflow: Research -> Synthesize & Refine."
)

print("âœ“ Workflow orchestration defined successfully")


import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "digipulse360"
USER_ID = "user_default"
SESSION_ID = "session_default"

async def run_digipulse360(query_text):
    print(f"Starting DigiPulse360 with query: '{query_text}'")
    print("-" * 50)

    # 1. Setup Session
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    # 2. Setup Runner
    runner = Runner(agent=root_workflow, app_name=APP_NAME, session_service=session_service)

    # 3. Create User Message
    content = types.Content(role='user', parts=[types.Part(text=query_text)])

    # 4. Run Workflow
    print("Agents are working... (This may take a minute)")
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    
    async for event in events:
        pass  # Process events silently

    # 5. Retrieve Final Result
    final_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    final_state = final_session.state
    
    newsletter = final_state.get("newsletter_draft")
    
    print("-" * 50)
    if newsletter:
        print("FINAL NEWSLETTER:\n")
        print(newsletter)
    else:
        print("Workflow completed, but no 'newsletter_draft' was found in state.")
    
    return newsletter

# Windows event loop fix
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

print("âœ“ Running DigiPulse360.")


# Run the system with a time-specific query
query = "Summarize developments in digital health for wound care during November 2025"
result = await run_digipulse360(query)

