# 1. Imports and API Key Configuration
import os
import uuid
import asyncio 
from kaggle_secrets import UserSecretsClient
from google.adk.models.google_llm import Gemini
from google.genai import types
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext, FunctionTool, load_memory
from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.plugins.logging_plugin import LoggingPlugin

# Retrieve the API key
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… API Key loaded.")
except Exception as e:
    print(f"âš ï¸� Error: {e}. Make sure 'GOOGLE_API_KEY' is in Add-ons > Secrets.")

# 2. Define the Pausing Tool (Unchanged)
def calculate_experience_gap(
    candidate_years: int, 
    required_years: int, 
    tool_context: ToolContext 
) -> dict:
    """Calculates experience gap and pauses for approval if borderline."""
    diff = candidate_years - required_years
    
    if diff >= 0:
        return {"status": "qualified", "gap": diff, "message": "Meets requirement."}
    
    if diff == -1:
        if not tool_context.tool_confirmation:
            tool_context.request_confirmation(
                hint=f"Candidate has {candidate_years} years (Requires {required_years}). Approve exception?",
                payload={"gap": diff}
            )
            return {"status": "pending", "message": "Waiting for manager approval..."}
        
        elif tool_context.tool_confirmation.confirmed:
             return {"status": "qualified", "gap": diff, "message": "Manager approved exception."}
        else:
             return {"status": "underqualified", "gap": diff, "message": "Manager rejected exception."}

    return {"status": "underqualified", "gap": diff, "message": f"Missing {abs(diff)} years."}

# 3. Initialize Core Services
model = Gemini(model="gemini-2.5-flash-lite")
memory_service = InMemoryMemoryService()
session_service = InMemorySessionService()

print("âœ… Step 1 Complete: Services and Pausing Tool Ready.")


extractor_agent = Agent(
    name="Extractor", model=model,
    instruction="You are an expert Resume Parser. Read the resume text and extract exactly two things: 'years_experience' (as an integer) and 'skills' (as a list of strings). Output strictly in JSON format.",
    output_key="candidate_profile" 
)


# --- FINAL AGENT INSTRUCTION ---
evaluator_agent = Agent(
    name="Evaluator", model=model,
    instruction="""You are a Hiring Manager. The required experience is 5 years.
    1. Read 'candidate_profile'.
    2. **CRITICAL:** Call `calculate_experience_gap` using the candidate's `years_experience` and 5 for the `required_years` argument.
    3. **IMPORTANT:** If the tool's status requires a pause ('pending'), you MUST stop immediately. DO NOT proceed to call `load_memory` or generate any final text.
    4. ONLY if the status is 'qualified' or 'underqualified':
        a. Use `load_memory` to retrieve the Job Description (JD) for skill comparison.
        b. Write a FINAL, one-paragraph evaluation. You MUST start the output with **STRICTLY ONE** of these words: **[INTERVIEW]** or **[REJECT]**.
        c. If the tool status is 'underqualified' OR if essential skills (Python, SQL, ADK) are missing, the verdict must be **[REJECT]**.
        d. The justification must clearly cite the experience status (from the tool result) and the skill match against the JD.""",
    tools=[FunctionTool(calculate_experience_gap), load_memory],
    output_key="evaluation_report",
    #input_key="candidate_profile"
)

# 2. Sequential Pipeline and App Setup
resume_pipeline = SequentialAgent(name="ResumeScreenerPipeline", sub_agents=[extractor_agent, evaluator_agent])
screener_app = App(
    name="ResumeScreener",
    root_agent=resume_pipeline,
    resumability_config=ResumabilityConfig(is_resumable=True),
    plugins=[LoggingPlugin()] 
)
# 3. Configure the Runner
runner = Runner(
    app=screener_app,
    session_service=session_service,
    memory_service=memory_service
)

# 4. Memory Ingestion Function
async def teach_job_description(jd_text):
    """Writes the JD into the shared Memory Service."""
    print(f"ğŸ’¾ Learning Job Description...")
    setup_agent = Agent(name="SetupAgent", model=model, instruction="Acknowledge the Job Description.")
    setup_runner = Runner(
        agent=setup_agent, app_name="ResumeScreener", session_service=session_service, memory_service=memory_service
    )
    session_id = "jd_ingestion_session"
    await setup_runner.run_debug(f"Here is the Job Description to remember: {jd_text}", session_id=session_id, user_id="manager")
    session = await session_service.get_session(app_name="ResumeScreener", user_id="manager", session_id=session_id)
    await memory_service.add_session_to_memory(session)
    print("âœ… Job Description saved to Memory!")

JOB_DESCRIPTION = """
We are looking for a Senior Python Developer.
Requirements:
- Minimum 5 years of experience.
- Must have skills: Python, SQL, and ADK.
"""

# Run the teaching step
await teach_job_description(JOB_DESCRIPTION)

print("âœ… Step 2 Complete: Runner configured and Memory Loaded.")


async def screen_candidate_workflow(resume_text, mock_decision="approve"):
    print("\nğŸš€ Screening Candidate...")
    user_id = "manager"
    session_id = f"session_{uuid.uuid4()}"
    
    await session_service.create_session(
        app_name="ResumeScreener", user_id=user_id, session_id=session_id
    )
    
    # 1. Start the Run
    new_content_message = types.Content(role="user", parts=[types.Part(text=f"Process resume: {resume_text}")])
    events = []
    print("\nStarting Agent Run...")
    async for event in runner.run_async(
        new_message=new_content_message, session_id=session_id, user_id=user_id
    ):
        events.append(event)
        if event.content and event.content.parts and event.content.parts[0].text:
             print(f"ğŸ¤– Agent Output: {event.content.parts[0].text}")

    # 2. Check for Confirmation Pause
    confirmation_request = None
    pause_invocation_id = None
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call and part.function_call.name == "adk_request_confirmation":
                    confirmation_request = part.function_call
                    pause_invocation_id = event.invocation_id
                    break
            if confirmation_request:
                break
    
    # 3. Handle the Pause or Final Output
    if confirmation_request:
        # HUMAN INTERVENTION (Borderline Case)
        hint_text = confirmation_request.args.get('hint', 'Confirmation requested for borderline candidate.')
        
        print(f"\n=======================================================")
        print(f"â�¸ï¸� HUMAN INPUT MOCKED: {hint_text}")
        print("   (Interactive input disabled in this environment)")
        print(f"=======================================================")
        
        # --- FIX: Mock the interactive input call ---
        is_approved = mock_decision.lower().startswith('a')
        # -------------------------------------------

        print(f"\nâ–¶ï¸� Resuming with decision: {'Approved' if is_approved else 'Rejected'}...")
        
        confirmation_response = types.FunctionResponse(
            id=confirmation_request.id, name="adk_request_confirmation", response={"confirmed": is_approved}
        )
        
        resume_payload = types.Content(role="user", parts=[types.Part(function_response=confirmation_response)])
        
        final_output_events = []
        async for event in runner.run_async(
            new_message=resume_payload,
            session_id=session_id,
            user_id=user_id,
            invocation_id=pause_invocation_id
        ):
             final_output_events.append(event)
             if event.content and event.content.parts and event.content.parts[0].text:
                 # This will show the final verdict after the pause
                 pass 
        
        final_output = next((event.content.parts[0].text
                             for event in reversed(final_output_events)
                             if event.content and event.content.parts and event.content.parts[0].text),
                             "Error: Final Agent Output not found after resume.")
        print(f"\n=======================================================")
        print(f"âœ… FINAL VERDICT (After Resume):")
        print(final_output)
        print(f"=======================================================")

    else:
        # NO PAUSE (Qualified or Rejected Case)
        final_output = next((event.content.parts[0].text
                             for event in reversed(events)
                             if event.content and event.content.parts and event.content.parts[0].text),
                             "Error: Final Agent Output not found.")
        print(f"\n=======================================================")
        print(f"âœ… FINAL VERDICT (No Pause Required):")
        print(final_output)
        print(f"=======================================================")

# --- Test Case 1: BORDERLINE (Should pause and be approved) ---
BORDERLINE_RESUME = """
Candidate: Junior Dev
Experience: 4 years of Python.
Work History: 2020-2024 (4 years total experience).
Skills: Python, SQL, ADK.
"""
print("Running Borderline Candidate Test (MOCK APPROVE)...")
await screen_candidate_workflow(BORDERLINE_RESUME, mock_decision="approve")

# --- Test Case 2: CLEARLY QUALIFIED (No Pause) ---
QUALIFIED_RESUME = """
Candidate: Senior Expert
Experience: 8 years working with backend systems.
Skills: Python, SQL, ADK, Java.
"""
print("\nRunning Qualified Candidate Test...")
await screen_candidate_workflow(QUALIFIED_RESUME)

# --- Test Case 3: CLEARLY REJECTED (No Pause - ensures the previous fix worked) ---
REJECTED_RESUME = """
Candidate: Beginner
Experience: 2 years.
Skills: JavaScript, HTML.
"""
print("\nRunning Rejected Candidate Test...")
await screen_candidate_workflow(REJECTED_RESUME)

