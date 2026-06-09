# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

agents_intensive_capstone_project_path = kagglehub.competition_download('agents-intensive-capstone-project')
print('Data source import complete.')


%%capture
# Install Google Agent Development Kit (ADK) and GenAI libraries
# Using '-q' flag for clean output
!pip install -q google-adk google-genai

# Core imports for agent orchestration
import os
from kaggle_secrets import UserSecretsClient
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
from IPython.display import display, Markdown


# Configure retry options for robust API calls
# Handles transient failures (rate limits, server errors)
retry_config = types.HttpRetryOptions(
    attempts=5,                              # Maximum retry attempts
    exp_base=7,                              # Exponential backoff multiplier
    initial_delay=1,                         # Initial delay in seconds
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

# Securely retrieve API Key from Kaggle Secrets
# Best practice: Never hardcode credentials
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ” Credentials Loaded. ADK Environment Ready.")
except Exception as e:
    print(f"âœ˜ Authentication Error: Please add 'GOOGLE_API_KEY' to Kaggle secrets. Details: {e}")


# Initialize Gemini model with retry configuration
heron_model = Gemini(
    model="gemini-2.5-flash-lite",  # Fast, efficient model for agent tasks
    retry_options=retry_config       # Ensures reliability under load
)
print("âœ” Cognitive Engine Initialized: Gemini 2.5 Flash Lite")


triage_agent = Agent(
    name="TriageAgent",
    model=heron_model,
    instruction="""
    You are a Cybersecurity Triage Analyst.
    Your goal is to parse the input message and extract structured data.
    
    Step 1: Identify the 'sender_claim' (Who does the message claim to be?).
    Step 2: Extract 'contact_info' (Phone numbers, URLs) or return 'None'.
    Step 3: Assess 'risk_tone' (Is it Urgent, Threatening, or Casual?).
    
    Output strictly in JSON format.
    """,
    output_key="triage_findings"
)
print("âœ” Triage Agent Initialized")


investigator_agent = Agent(
    name="InvestigatorAgent",
    model=heron_model,
    instruction="""
    You are a Senior Security Researcher.
    Read the provided JSON data containing 'sender_claim' and 'contact_info': {triage_findings}
    
    Your Task:
    1. If 'contact_info' is present, use the google_search tool to verify if it belongs to the 'sender_claim'.
    2. Search for fraud reports or 'scam' warnings associated with the contact info.
    3. Determine if the contact channel is official or fraudulent.
    
    Provide a concise summary of your findings based on the search results.
    """,
    tools=[google_search],  # Grounding capability: Live web intelligence
    output_key="investigator_findings"
)
print("âœ” Investigator Agent Initialized")


reporter_agent = Agent(
    model=heron_model,
    name="ReporterAgent",
    instruction="""
    You are Heron, a Security Analyst delivering executive threat briefings.
    
    Your responsibilities:
    1. Synthesize triage assessment and investigation findings from previous agents
    2. Translate technical evidence into clear risk determination
    3. Provide decisive, actionable recommendations
    4. Present findings in a professional security alert format
    
    Always respond in this exact format:
    
    ## ğŸ¦… HERON VERDICT: [âœ” SAFE | âš  SUSPICIOUS | ğŸš¨ MALICIOUS]
    
    **Evidence Summary:**
    â€¢ [Key finding from triage]
    â€¢ [Key finding from investigation]
    â€¢ [Verification status of claimed identity]
    
    **Recommendation:**
    [BLOCK & REPORT | IGNORE & DELETE | SAFE TO PROCEED]
    
    **Reasoning:**
    [One clear paragraph explaining the verdict based on gathered evidence]
    
    Be direct, confident, and base all conclusions strictly on provided evidence.
    """,
    output_key="final_report"
)
print("âœ” Reporter Agent Initialized")
print("âœ” Agent Team Assembled: Triage -> Investigator -> Reporter.")


# Create sequential workflow: Triage â†’ Investigator â†’ Reporter
# Each agent's output becomes the next agent's input
root_agent = SequentialAgent(
    name="HeronPipeline",
    sub_agents=[triage_agent, investigator_agent, reporter_agent]
)
print("âœ” Sequential Agent created.")


# Initialize runner to manage execution loop and session history
# InMemoryRunner enables the system to "remember" context across messages
runner = InMemoryRunner(agent=root_agent)


# Simulated phishing text with urgency tactics
scam_text = "ALERT: Your Wells Fargo account is suspended. Call 800-555-0199 immediately."
result_scam = await runner.run_debug(scam_text)


# Normal casual message from friend/contact
safe_text = "Hey, just checking in to see if you're free for lunch tomorrow?"
result_safe = await runner.run_debug(safe_text)

